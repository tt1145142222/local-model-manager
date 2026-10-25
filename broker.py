"""Single-writer resource broker with durable leases and conservative recovery."""
import copy
import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
import threading
import time
import uuid
from adapters import Adapter, process_identity, same_process
import psutil
from resources import snapshot, pressure_reasons
from broker_adaptive import AdaptiveBroker
from adaptive import normalize_task, resolve_kind
from app_memory import AppMemoryReclaimer
from manager_control import disabled_state, DISABLED_MESSAGE

LIVE = {"starting", "active", "orphaned", "releasing"}
DONE = {"released", "expired", "cancelled", "failed"}


class Broker(AdaptiveBroker):
    def __init__(self, root, config, adapters=None, sample_fn=snapshot, config_path=None):
        self.root, self.config = Path(root), config
        self.config_path = Path(config_path) if config_path else None
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.refresh_lock = threading.RLock()
        self.db = sqlite3.connect(self.root / "state.sqlite3", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, data TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, time REAL, kind TEXT, data TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS samples (id INTEGER PRIMARY KEY, time REAL, data TEXT)")
        row = self.db.execute("SELECT data FROM state WHERE id=1").fetchone()
        self.state = json.loads(row[0]) if row else {"leases": [], "managed": {}, "cleanup": []}
        self.adapters = adapters or {k: Adapter(k, v) for k, v in config["models"].items()}
        self.sample_fn = sample_fn
        self.resources, self.models = {}, {}
        self.last_sample = 0
        self.stop_event = threading.Event()
        self.monitor_wake = threading.Event()
        self.worker_error = None
        self.lifecycle_action = None
        self.init_adaptive()
        # An interrupted start/stop has an unknown outcome. Reconcile, never replay it.
        for lease in self.state["leases"]:
            if lease["state"] in {"starting", "releasing"}:
                lease.update(state="orphaned", reason="manager restarted during lifecycle operation",
                             expires=0, release_requested=True)
        for op in self.state["cleanup"]:
            if op["state"] == "running":
                op.update(state="failed", reason="manager restarted; inspect before retrying")
        self.save()

        app_config = copy.deepcopy(config)
        if self.control_status()['disabled']:
            app_config['app_memory_reclaim'] = {'enabled': False}
        self.app_reclaimer = AppMemoryReclaimer(app_config, self.app_memory_context, self.app_memory_event)

    def control_status(self):
        control = disabled_state(self.root)
        return {**control, 'mode': 'disabled_preserving_existing_tasks' if control['disabled'] else 'enabled',
                'automatic_start_allowed': not control['disabled'],
                'new_requests_allowed': not control['disabled'],
                'background_app_reclaim_allowed': not control['disabled'],
                'new_model_start_allowed': not control['disabled'],
                'existing_task_model_maintenance_only': control['disabled']}

    def disabled_decision(self):
        """Drain existing tasks; no new model starts or unrelated app reclaim.

        Preserve the existing task's original model maintenance so disabling the
        manager cannot strand an already admitted task on its next prepare call.
        """
        now = time.time()
        for lease in self.state['leases']:
            if lease['state'] == 'queued' and lease['expires'] < now:
                lease.update(state='expired', reason='manager disabled; queued request expired', finished=now)
            if lease['state'] not in LIVE:
                continue
            model = self.models.get(lease['model'], {})
            if model.get('state') == 'stopped':
                lease.update(state='failed', reason='model stopped independently while manager disabled', finished=now)
                continue
            if model.get('identity') and lease.get('identity') != model.get('identity'):
                lease.update(state='failed', reason='model identity changed independently; replacement retained', finished=now)
                continue
            if lease['release_requested'] or lease['expires'] < now:
                if lease.get('worker') and same_process(lease['worker']):
                    lease['reason'] = 'manager disabled; retaining interface for existing live worker'
                elif model.get('state') == 'ready' and model.get('busy') is False:
                    self.finish(lease, 'manager disabled; existing task finished; model service retained')
            else:
                lease['reason'] = '; '.join(self.plan_for(lease)['reasons']) or 'manager disabled; existing task interface retained'
                action = self.adaptive_action(lease)
                if action:
                    return action
        return None

    def disabled_tasks_finished(self):
        return (self.control_status()['disabled'] and
                not any(x['state'] in LIVE or x['state'] == 'queued' for x in self.state['leases']))

    def app_memory_event(self, kind, data):
        with self.lock:
            self.event(kind, data)

    def app_memory_context(self):
        """Read-only RAM demand; the reclaimer cannot change admission or leases."""
        with self.lock:
            pids = [m['identity']['pid'] for m in self.models.values() if m.get('identity')]
            pids += [x['worker']['pid'] for x in self.state['leases'] if x.get('worker') and x['state'] in LIVE]
            roots = [str(Path(__file__).resolve().parent)]
            roots += [c['root'] for c in self.config['models'].values() if c.get('root')]
            result = {'allowed': False, 'reason': 'no_model_task', 'protected_pids': pids,
                      'protected_roots': roots, 'required_ram_gib': None}
            if self.control_status()['disabled']:
                return dict(result, reason='disabled_by_user')
            if self.stop_event.is_set() or self.worker_error:
                return dict(result, reason='manager_stopping_or_monitor_error')
            if not self.last_sample or time.time()-self.last_sample > max(20, self.config['sample_seconds']*3):
                return dict(result, reason='model_telemetry_stale')
            lease = next((x for x in self.state['leases'] if x['state'] in LIVE), None)
            if lease is None:
                lease = next((x for x in self.state['leases'] if x['state'] == 'queued' and x['expires'] > time.time()), None)
            if not lease or lease['expires'] <= time.time() or lease['release_requested']:
                return result
            model = lease['model']
            state = self.models.get(model, {})
            if state.get('state') not in {'ready', 'stopped'} or state.get('busy') not in {True, False}:
                return dict(result, reason='model_identity_or_busy_state_unknown')
            if lease['state'] == 'queued' and any(k != model and m.get('state') != 'stopped' for k, m in self.models.items()):
                return dict(result, reason='waiting_for_other_model')
            pending = lease.get('pending_admission')
            admission = lease['state'] in {'queued', 'starting'} or bool(pending and not self.active_work(lease) and state.get('busy') is False)
            task = pending['task'] if pending and admission else lease.get('task', {})
            plan = self.plan_for(lease, cold=state['state'] == 'stopped', task=task, admission=admission)
            if plan.get('retryable') is False:
                return dict(result, reason='model_policy_infeasible')
            return dict(result, allowed=True, reason='model_ram_demand', model=model,
                        required_ram_gib=plan['required']['ram_gib'])

    def app_memory_status(self):
        return self.app_reclaimer.status()

    def save(self):
        self.db.execute("INSERT OR REPLACE INTO state VALUES (1, ?)", (json.dumps(self.state),))
        self.db.commit()

    def event(self, kind, data):
        self.db.execute("INSERT INTO events(time,kind,data) VALUES (?,?,?)", (time.time(), kind, json.dumps(data)))
        self.db.execute("DELETE FROM events WHERE id < (SELECT COALESCE(MAX(id),0)-2000 FROM events)")
        self.db.commit()

    def refresh(self):
        with self.refresh_lock:
            self._refresh()

    def _refresh(self):
        resources = self.sample_fn()
        models = {k: a.inspect() for k, a in self.adapters.items()}
        with self.lock:
            self.resources, self.models, self.last_sample = resources, models, time.time()
            self.db.execute("INSERT INTO samples(time,data) VALUES (?,?)", (self.last_sample, json.dumps(resources)))
            self.db.execute("DELETE FROM samples WHERE id < (SELECT COALESCE(MAX(id),0)-1440 FROM samples)")
            self.adaptive_refresh(resources, models)
            for lease in self.state["leases"]:
                if lease["state"] in LIVE:
                    m = lease.setdefault("measurements", {})
                    m["samples"] = m.get("samples", 0)+1
                    for key in ("ram_available_gib", "commit_free_gib"):
                        value = resources.get(key)
                        if value is not None:
                            m["minimum_"+key] = min(m.get("minimum_"+key, value), value)
                    gpu = next(iter(resources.get("gpus", [])), None)
                    if gpu:
                        m["peak_device_used_gib"] = max(m.get("peak_device_used_gib", 0), gpu["used_gib"])
                    rss = models.get(lease["model"], {}).get("rss_gib")
                    if rss is not None:
                        m["peak_model_rss_gib"] = max(m.get("peak_model_rss_gib", 0), rss)
            self.save()

    def public_lease(self, lease, credential=False):
        value = copy.deepcopy(lease)
        if not credential:
            value.pop("token", None)
        plan = self.plan_for(lease)
        value["plan"] = plan
        value["proceed"] = (lease["state"] == "active" and not lease["release_requested"] and lease["expires"] > time.time() and not self.worker_error and not self.lifecycle_action and
                            time.time()-self.last_sample < max(20, self.config["sample_seconds"]*3) and
                            not plan['reasons'] and
                            self.models.get(lease["model"], {}).get("state") == "ready" and
                            self.models.get(lease["model"], {}).get("identity") == lease.get("identity"))
        value["endpoint"] = self.config["models"][lease["model"]]["url"]
        return value

    def status(self):
        with self.lock:
            return {"version": "1.2.0", "policy": "adaptive headroom and bounded learning; exclusive GPU lease; external services preserved",
                    "control": self.control_status(),
                    "adaptive": self.adaptive_status(),
                    "app_memory_reclaim": self.app_memory_status(),
                    "health_report": self.health_report(),
                    "resources": copy.deepcopy(self.resources), "sample_age_seconds": round(time.time()-self.last_sample, 1) if self.last_sample else None,
                    "monitor_state": "ready" if self.last_sample else "initializing",
                    "models": {k: {**v, "label": self.config["models"][k]["label"],
                                   "manager_owned": v.get("identity") is not None and self.state["managed"].get(k, {}).get("identity") == v.get("identity")}
                               for k, v in self.models.items()},
                    "leases": [self.public_lease(x) for x in self.state["leases"] if x["state"] not in DONE],
                    "cleanup": copy.deepcopy(self.state["cleanup"][-10:]), "worker_error": self.worker_error,
                    "lifecycle_action": self.lifecycle_action}

    def request_model(self, model, owner, request_key, ttl_seconds=3600, keep_loaded=False, task=None):
        if model not in self.adapters:
            raise ValueError("Unknown model; register a supported runtime before use")
        if not isinstance(owner, str) or not owner.strip() or len(owner) > 300:
            raise ValueError("owner must identify this chat and task (1..300 characters)")
        if not isinstance(request_key, str) or not request_key.strip() or len(request_key) > 200:
            raise ValueError("request_key must be a stable task identifier (1..200 characters)")
        if type(ttl_seconds) is not int or not 60 <= ttl_seconds <= 28800 or type(keep_loaded) is not bool:
            raise ValueError("TTL must be 60..28800 seconds and keep_loaded must be boolean")
        task = normalize_task(model, task, resolve_kind(self.config,model))
        if task.get('context_tokens',0) > self.config['models'][model].get('context_capacity',65536):
            raise ValueError('Task context exceeds installed capacity; task is never silently truncated')
        with self.lock:
            if self.stop_event.is_set():
                raise ValueError('Manager is stopping; retry after restart')
            existing = next((x for x in self.state["leases"] if x["owner"] == owner and x["request_key"] == request_key), None)
            if existing:
                if existing["model"] != model or existing["keep_loaded"] != keep_loaded or existing.get('task',{}) != task:
                    raise ValueError("request_key already used for a different request")
                return self.public_lease(existing, True)
            if self.control_status()['disabled']:
                raise ValueError(DISABLED_MESSAGE)
            now = time.time()
            if sum(x['state'] == 'queued' for x in self.state['leases']) >= 64:
                raise ValueError('Pending queue is full; wait for existing work')
            lease = {"id": str(uuid.uuid4()), "token": secrets.token_urlsafe(32), "model": model,
                     "owner": owner, "request_key": request_key, "state": "queued", "reason": "awaiting scheduler",
                     "created": now, "expires": now+ttl_seconds, "ttl_seconds": ttl_seconds,
                     "keep_loaded": keep_loaded, "release_requested": False, "task": task}
            self.state["leases"].append(lease)
            self.save()
            self.event("request", {k: lease[k] for k in ("id", "model", "owner")})
            self.monitor_wake.set()
            return self.public_lease(lease, True)

    def authenticated(self, lease_id, token):
        lease = next((x for x in self.state["leases"] if x["id"] == lease_id), None)
        if not lease or not isinstance(token, str) or not secrets.compare_digest(lease["token"], token):
            raise ValueError("Lease credential mismatch; another chat's lease cannot be changed")
        return lease

    def lease_status(self, lease_id, token, renew=True):
        with self.lock:
            lease = self.authenticated(lease_id, token)
            if renew and lease["state"] in {"queued", "active"} and not lease["release_requested"]:
                lease["expires"] = time.time()+lease["ttl_seconds"]
                self.save()
            return self.public_lease(lease)

    def release(self, lease_id, token):
        with self.lock:
            lease = self.authenticated(lease_id, token)
            if lease["state"] == "queued":
                lease.update(state="cancelled", finished=time.time(), reason="cancelled by owner")
            elif lease["state"] not in DONE:
                lease.update(release_requested=True, cleanup_blocked=False, reason="release requested; waiting for verified idle")
            self.save()
            return self.public_lease(lease)

    def attach_worker(self, lease_id, token, pid):
        if type(pid) is not int or pid <= 0:
            raise ValueError("Invalid worker PID")
        with self.lock:
            lease = self.authenticated(lease_id, token)
            if lease["state"] != "active" or lease["release_requested"]:
                raise ValueError("Attach only to an active lease")
            lease["worker"] = process_identity(psutil.Process(pid))
            self.save()
            return self.public_lease(lease)

    def release_idle(self, model, allow_external=False):
        if model not in self.adapters or type(allow_external) is not bool:
            raise ValueError("Unknown model or invalid allow_external")
        with self.lock:
            if any(x["state"] in LIVE for x in self.state["leases"]):
                raise ValueError("A lease is active; its owner must release it first")
            current = self.adapters[model].inspect()
            if current["state"] == "stopped":
                return {"state": "succeeded", "model": model, "already_stopped": True}
            if current["state"] != "ready" or current["busy"] is not False:
                raise ValueError("Service is busy or unverified; release refused")
            owned = self.state["managed"].get(model, {}).get("identity") == current["identity"]
            if not owned and not allow_external:
                raise ValueError("External service: explicit user authorization is required before allow_external=true")
            op = {"id": str(uuid.uuid4()), "model": model, "identity": current["identity"],
                  "state": "queued", "allow_external": allow_external, "created": time.time()}
            self.state["cleanup"].append(op)
            self.save()
            self.event("cleanup_requested", {k: op[k] for k in ("id", "model", "allow_external")})
            return copy.deepcopy(op)

    def history(self, limit=20):
        limit = min(100, max(1, int(limit)))
        with self.lock:
            rows = self.db.execute("SELECT time,kind,data FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return {"leases": [self.public_lease(x) for x in self.state["leases"][-limit:]],
                    "events": [{"time": t, "kind": k, "data": json.loads(d)} for t, k, d in rows]}

    def finish(self, lease, reason):
        work = self.active_work(lease)
        if work:
            self.record_work(work)
        lease.update(state="expired" if lease["expires"] < time.time() else "released", finished=time.time(), reason=reason)
        self.event("lease_finished", {k: lease[k] for k in ("id", "model", "state")})

    def decision(self):
        """Called with the broker lock. Returns at most one lifecycle action."""
        if self.control_status()['disabled']:
            return self.disabled_decision()
        now = time.time()
        for lease in self.state["leases"]:
            if lease["state"] == "queued" and lease["expires"] < now:
                lease.update(state="expired", reason="queued request expired", finished=now)
        active = next((x for x in self.state["leases"] if x["state"] in LIVE), None)
        if active:
            s = self.models[active["model"]]
            if active["state"] in {"starting", "releasing"}:
                return None
            if s["state"] == "stopped":
                active.update(state="failed", reason="model service stopped", finished=now)
                return None
            if active.get("identity") and s.get("identity") and s["identity"] != active["identity"]:
                active.update(state="failed", reason="model process identity changed; replacement retained", finished=now)
                return None
            if active.get("cleanup_blocked"):
                return None
            if active["release_requested"] or active["expires"] < now:
                active["state"] = "orphaned" if active["expires"] < now else "active"
                active["reason"] = "holding GPU lease until service is verified idle"
                if active.get("worker") and same_process(active["worker"]):
                    active["reason"] = "registered worker is still alive; lease retained"
                    return None
                if s["state"] == "ready" and s["busy"] is False:
                    managed = self.state["managed"].get(active["model"], {})
                    warm_seconds = self.plan_for(active)['warm_seconds']
                    if managed.get("identity") == s["identity"] and (not active["keep_loaded"] or active["expires"] < now or warm_seconds <= 0):
                        active["state"] = "releasing"
                        return ("stop_lease", active["model"], s["identity"], active["id"])
                    if managed:
                        managed["idle_since"] = now
                        managed['warm_seconds'] = warm_seconds
                    self.finish(active, "lease released; service retained")
            else:
                active["reason"] = "; ".join(self.plan_for(active)['reasons']) or "ready"
                action = self.adaptive_action(active)
                if action:
                    return action
            return None
        op = next((x for x in self.state["cleanup"] if x["state"] == "queued"), None)
        if op:
            op["state"] = "running"
            return ("stop_explicit", op["model"], op["identity"], op["id"])
        queued = next((x for x in self.state["leases"] if x["state"] == "queued"), None)
        for model, managed in list(self.state["managed"].items()):
            s = self.models[model]
            if s["state"] == "stopped":
                self.state["managed"].pop(model)
                continue
            if s.get("identity") != managed["identity"]:
                continue
            if managed.get("auto_cleanup_failed"):
                continue
            guard = self.policy.plan(model, {}, self.resources, self.models)
            warm_seconds = min(managed.get('warm_seconds',self.config['idle_grace_seconds']),guard['warm_seconds'])
            idle_timeout = now-managed.get("idle_since", now) >= warm_seconds or bool(guard['reasons'])
            switch = queued and queued["model"] != model
            if (idle_timeout or switch) and s["state"] == "ready" and s["busy"] is False:
                return ("stop_idle", model, s["identity"], None)
        if not queued:
            return None
        target = queued["model"]
        for model, s in self.models.items():
            if model != target and s["state"] != "stopped":
                queued["reason"] = f"waiting for {model}; external or busy service requires explicit release"
                return None
        s = self.models[target]
        if s["state"] == "unknown" or s["busy"] is not False:
            queued["reason"] = "target is busy or state is unverified"
            return None
        if s['state']=='stopped':
            self.refresh_artifact_signature()
        admission = self.plan_for(queued, cold=s["state"] == "stopped", admission=True)
        reasons = admission['reasons']
        if admission.get('retryable') is False:
            queued.update(state='failed', reason='; '.join(reasons), finished=now,
                          admission_state=admission.get('admission_state'), retryable=False)
            self.event('admission_infeasible', {'model': target, 'id': queued['id'],
                                              'reason': queued['reason']})
            return None
        if reasons:
            queued["reason"] = "; ".join(reasons)
            return None
        if s["state"] == "stopped":
            queued.update(state="starting", reason="starting registered model", started=now)
            return ("start", target, None, queued["id"])
        queued.update(state="active", reason="ready; reused existing service", started=now, identity=copy.deepcopy(s["identity"]))
        self.event("lease_granted", {"id": queued["id"], "model": target, "reused": True})
        return None

    def tick(self):
        self.refresh()
        with self.lock:
            action = self.decision()
            self.lifecycle_action = None if not action else {"kind": action[0], "model": action[1]}
            self.save()  # Commit intent BEFORE the external action.
        if not action:
            return
        kind, model, identity, ref = action
        try:
            if kind == "start":
                identity = self.adapters[model].start()
                with self.lock:
                    self.state["managed"][model] = {"identity": copy.deepcopy(identity), "idle_since": time.time(),
                                                     'cache_state':'cold' if self.config['models'][model]['kind']=='comfyui' else 'warm'}
                    lease = next(x for x in self.state["leases"] if x["id"] == ref)
                    lease.update(state="active", reason="ready", identity=copy.deepcopy(identity), expires=time.time()+lease["ttl_seconds"])
                    self.event("lease_granted", {"id": ref, "model": model, "reused": False})
            elif kind in {'runtime','free_cache'}:
                self.perform_adaptive_action(kind,model,identity,ref)
            else:
                self.adapters[model].stop(identity)
                with self.lock:
                    self.state["managed"].pop(model, None)
                    if kind == "stop_lease":
                        self.finish(next(x for x in self.state["leases"] if x["id"] == ref), "managed service stopped")
                    elif kind == "stop_explicit":
                        next(x for x in self.state["cleanup"] if x["id"] == ref).update(state="succeeded", finished=time.time())
                    self.event("service_stopped", {"model": model, "kind": kind})
        except Exception as exc:
            with self.lock:
                reason = type(exc).__name__ + ": " + str(exc)[:300]
                if kind in {'runtime','free_cache'}:
                    managed = self.state['managed'].get(model,{})
                    managed['runtime_error' if kind=='runtime' else 'cache_error'] = reason
                elif kind in {"start", "stop_lease"}:
                    lease = next(x for x in self.state["leases"] if x["id"] == ref)
                    if kind == "start":
                        # Unknown startup outcome remains visible; no auto resubmit.
                        lease.update(state="failed", reason=reason, finished=time.time())
                    else:
                        lease.update(state="orphaned", reason=reason, expires=0, cleanup_blocked=True)
                elif kind == "stop_explicit":
                    next(x for x in self.state["cleanup"] if x["id"] == ref).update(state="failed", reason=reason)
                else:
                    # Stop auto retries until owner asks again; identity is preserved.
                    self.state["managed"][model]["auto_cleanup_failed"] = reason
                self.event("lifecycle_failed", {"model": model, "operation": kind, "reason": reason})
        finally:
            with self.lock:
                self.lifecycle_action = None
                self.save()
        self.refresh()

    def run(self):
        while not self.stop_event.is_set():
            self.monitor_wake.clear()
            started = time.monotonic()
            try:
                self.tick()
                self.worker_error = None
            except Exception as exc:
                self.worker_error = type(exc).__name__ + ": " + str(exc)[:200]
            with self.lock:
                if self.disabled_tasks_finished():
                    self.event('disabled_tasks_finished', {'models': 'retained', 'auto_start': 'blocked'})
                    self.stop_event.set()
                    self.monitor_wake.set()
                    return
                interval = self.sampling_interval()
            self.monitor_wake.wait(max(.05, interval-(time.monotonic()-started)))
