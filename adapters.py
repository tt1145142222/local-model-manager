"""Verified local service adapters; never stop by process name or PID alone."""
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib import request, error, parse
import psutil
from resources import NO_WINDOW


def http_json(url, payload=None, headers=None, timeout=3):
    if parse.urlparse(url).hostname != "127.0.0.1":
        raise ValueError("Only fixed loopback model APIs are supported")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    h = {"Content-Type": "application/json", **(headers or {})}
    opener = request.build_opener(request.ProxyHandler({}))
    with opener.open(request.Request(url, data=data, headers=h), timeout=timeout) as response:
        raw = response.read(2 * 1024 * 1024)
        return json.loads(raw) if raw else {}


def process_identity(p):
    return {"pid": p.pid, "created": p.create_time(), "exe": str(Path(p.exe()).resolve())}


def same_process(identity):
    try:
        p = psutil.Process(identity["pid"])
        return (abs(p.create_time()-identity["created"]) < .01 and
                os.path.normcase(p.exe()) == os.path.normcase(identity["exe"]))
    except (psutil.Error, KeyError):
        return False


class Adapter:
    def __init__(self, model_id, config):
        self.model_id, self.c = model_id, config
        self.root = Path(config["root"])
        # One service identity per adapter keeps sampling and runtime state bounded.
        self._sample = None
        self._runtime_baseline = None

    def _process_telemetry(self, p, identity):
        """Nonblocking process samples, normalized to the whole logical CPU pool."""
        result = {"cpu_percent": None, "cpu_process_percent": None,
                  "read_bytes_per_sec": None, "write_bytes_per_sec": None}
        try:
            cpu = p.cpu_times()
            cpu_seconds = float(cpu.user + cpu.system)
            if not math.isfinite(cpu_seconds):
                return result
            sample = {"identity": identity, "at": time.monotonic(), "cpu": cpu_seconds}
            try:
                io = p.io_counters()
                sample.update(read=int(io.read_bytes), write=int(io.write_bytes))
            except (psutil.Error, AttributeError, TypeError, ValueError):
                pass
            old, self._sample = self._sample, sample
            if old and old["identity"] == identity:
                elapsed = sample["at"] - old["at"]
                if elapsed > .05:
                    process_percent = max(0.0, 100 * (sample["cpu"] - old["cpu"]) / elapsed)
                    result.update(cpu_process_percent=round(process_percent, 2),
                                  cpu_percent=round(min(100.0, process_percent / (psutil.cpu_count() or 1)), 2))
                    for key in ("read", "write"):
                        if key in old and key in sample:
                            result[key + "_bytes_per_sec"] = max(0, round((sample[key] - old[key]) / elapsed))
        except (psutil.Error, AttributeError, TypeError, ValueError):
            pass
        return result

    def process(self):
        expected = os.path.normcase(str(Path(self.c["exe"]).resolve()))
        found = []
        for p in psutil.process_iter(["exe", "cmdline"]):
            try:
                if os.path.normcase(p.info["exe"] or "") != expected:
                    continue
                args = p.info["cmdline"] or []
                marker = self.c.get("process_marker")
                if marker is None and self.model_id == "h3" and self.c["kind"] == "comfyui":
                    marker = str(self.root / "work/setup/comfy_bootstrap.py")
                if marker:
                    if not any(os.path.normcase(a) == os.path.normcase(marker) for a in args):
                        continue
                if "--port" not in args or args[args.index("--port")+1] != str(parse.urlparse(self.c["url"]).port):
                    continue
                found.append(p)
            except (psutil.Error, IndexError):
                continue
        if len(found) > 1:
            raise RuntimeError("Multiple matching service processes; manual inspection required")
        return found[0] if found else None

    def headers(self):
        key_file = self.c.get("api_key_file")
        if key_file is None and self.model_id == "muse" and self.c["kind"] == "llamacpp":
            key_file = self.root / "config/api-key.txt"
        if key_file:
            key = Path(key_file).read_text(encoding="utf-8-sig").strip()
            return {"Authorization": "Bearer " + key}
        return {}

    def inspect(self):
        try:
            p = self.process()
            if p is None:
                self._sample = None
                port = parse.urlparse(self.c["url"]).port
                with socket.socket() as sock:
                    sock.settimeout(.3)
                    occupied = sock.connect_ex(("127.0.0.1", port)) == 0
                return {"state": "unknown" if occupied else "stopped", "busy": None if occupied else False,
                        "identity": None, "reason": "port occupied by unrecognized process" if occupied else ""}
            result = {"state": "unknown", "busy": None, "identity": process_identity(p),
                      "rss_gib": p.memory_info().rss / 1024**3}
            private = getattr(p.memory_info(), "private", None) if os.name == "nt" else None
            result["private_commit_gib"] = (private / 1024**3 if type(private) in (int, float)
                                            and math.isfinite(private) and private >= 0 else None)
            result.update(self._process_telemetry(p, result["identity"]))
            port = parse.urlparse(self.c["url"]).port
            if not any(c.status == psutil.CONN_LISTEN and c.laddr.port == port for c in p.net_connections(kind="tcp")):
                raise ValueError("Matching process does not own the expected listener")
            if self.c["kind"] == "comfyui":
                q = http_json(self.c["url"] + "/queue", headers=self.headers())
                if not isinstance(q.get("queue_running"), list) or not isinstance(q.get("queue_pending"), list):
                    raise ValueError("Invalid ComfyUI queue response")
                result.update(state="ready", busy=bool(q["queue_running"] or q["queue_pending"]),
                              running=len(q["queue_running"]), pending=len(q["queue_pending"]))
            else:
                h = http_json(self.c["url"] + "/health", headers=self.headers())
                slots = http_json(self.c["url"] + "/slots", headers=self.headers())
                if h.get("status") != "ok" or not isinstance(slots, list) or not slots:
                    raise ValueError("Model health or slots unavailable")
                if any(type(x.get("is_processing")) is not bool for x in slots):
                    raise ValueError("Unrecognized slot state")
                busy = any(x["is_processing"] for x in slots)
                agent_busy = False
                agent_online = False
                if self.c.get("agent_url"):
                    port = parse.urlparse(self.c["agent_url"]).port
                    with socket.socket() as sock:
                        sock.settimeout(.3)
                        agent_online = sock.connect_ex(("127.0.0.1", port)) == 0
                if agent_online:
                    sessions = http_json(self.c["agent_url"] + "/session/status")
                    if not isinstance(sessions, dict) or any(not isinstance(x, dict) or "type" not in x for x in sessions.values()):
                        raise ValueError("Agent session state unavailable")
                    agent_busy = any(x["type"] != "idle" for x in sessions.values())
                result.update(state="ready", busy=busy or agent_busy, processing_slots=sum(x["is_processing"] for x in slots),
                              agent_busy=agent_busy)
            return result
        except Exception as exc:
            return {**locals().get("result", {}), "state": "unknown", "busy": None,
                    "identity": locals().get("result", {}).get("identity"),
                    "reason": type(exc).__name__ + ": service state could not be verified"}

    def apply_runtime(self, identity, profile, baseline=None):
        """Adjust scheduling only; caller must prove manager ownership first.

        The broker persists the returned baseline alongside the owned identity so
        a broker restart cannot turn an already restricted mask into the original
        mask and reserve more CPUs on every restart. No quality or model options
        are accepted here. CPU IDs are logical processors, not physical cores.
        """
        if not isinstance(profile, dict) or set(profile) - {"cpu_priority", "reserve_logical_cpus"}:
            raise ValueError("Runtime profile accepts scheduling fields only")
        mode = profile.get("cpu_priority", "normal")
        reserve = profile.get("reserve_logical_cpus", 0)
        if mode not in ("normal", "below_normal", "restore"):
            raise ValueError("Unsupported CPU priority")
        if type(reserve) is not int or reserve < 0:
            raise ValueError("CPU reservation must be a nonnegative integer")
        state = self.inspect()
        if state["state"] != "ready" or state["identity"] != identity or not same_process(identity):
            raise RuntimeError("Service identity or health changed; runtime change refused")
        p = psutil.Process(identity["pid"])
        current_priority, current_affinity = int(p.nice()), sorted(p.cpu_affinity())
        saved = baseline if baseline is not None else self._runtime_baseline
        if baseline is None and saved is not None and saved.get("identity") != identity:
            saved = None
        if saved is not None:
            if (not isinstance(saved, dict) or saved.get("identity") != identity or
                    type(saved.get("priority")) is not int or not isinstance(saved.get("affinity"), list) or
                    not saved["affinity"] or any(type(x) is not int or x < 0 for x in saved["affinity"])):
                raise ValueError("Runtime baseline does not match the service identity")
            saved = {"identity": dict(identity), "priority": saved["priority"],
                     "affinity": sorted(set(saved["affinity"]))}
        else:
            saved = {"identity": dict(identity), "priority": current_priority, "affinity": current_affinity}
        if not saved["affinity"]:
            raise RuntimeError("Original CPU affinity is unavailable")
        # New process identities never reuse a previous process's scheduling mask.
        self._runtime_baseline = saved
        if mode == "restore":
            desired_priority, desired_affinity = saved["priority"], saved["affinity"]
        else:
            # Do not elevate above normal, or above a pre-existing idle/below-normal class.
            normal = int(getattr(psutil, "NORMAL_PRIORITY_CLASS", 0x20))
            below = int(getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 0x4000))
            idle = int(getattr(psutil, "IDLE_PRIORITY_CLASS", 0x40))
            desired_priority = below if mode == "below_normal" else normal
            if saved["priority"] == idle or (mode == "normal" and saved["priority"] == below):
                desired_priority = saved["priority"]
            effective_reserve = min(reserve, len(saved["affinity"]) - 1)
            desired_affinity = saved["affinity"][effective_reserve:]
        try:
            for change, desired, current in ((p.nice, desired_priority, current_priority),
                                             (p.cpu_affinity, desired_affinity, current_affinity)):
                if desired != current:
                    if not same_process(identity):
                        raise RuntimeError("Process identity changed during runtime adjustment")
                    change(desired)
            if not same_process(identity):
                raise RuntimeError("Process identity changed during runtime verification")
            actual_priority, actual_affinity = int(p.nice()), sorted(p.cpu_affinity())
            if actual_priority != desired_priority or actual_affinity != desired_affinity:
                raise RuntimeError("Runtime scheduling change could not be verified")
        except Exception:
            # Roll back a partially applied setting only on the same exact process.
            if same_process(identity):
                for change, value in ((p.nice, current_priority), (p.cpu_affinity, current_affinity)):
                    try:
                        change(value)
                    except (psutil.Error, OSError, ValueError):
                        pass
            raise
        return {"applied": True, "profile": dict(profile), "baseline": saved,
                "priority": actual_priority, "affinity": actual_affinity,
                "reserved_logical_cpus": len(saved["affinity"]) - len(actual_affinity)}

    def free_idle_cache(self, identity):
        """Ask ComfyUI to unload at its worker boundary; never interrupt a task."""
        if self.c["kind"] != "comfyui":
            raise ValueError("This service has no verified idle cache-release API")
        for _ in range(2):
            state = self.inspect()
            if state["state"] != "ready" or state["busy"] is not False or state["identity"] != identity:
                raise RuntimeError("Service changed or is busy/unknown; cache release refused")
            time.sleep(.15)
        if not same_process(identity):
            raise RuntimeError("Process identity changed; cache release refused")
        http_json(self.c["url"] + "/free", {"unload_models": True, "free_memory": True}, headers=self.headers())
        # /free is asynchronous. Telemetry must establish how many bytes were freed.
        return {"requested": True, "identity": dict(identity), "state": "cache_release_requested"}

    def start(self):
        before = self.inspect()
        if before["state"] != "stopped":
            raise RuntimeError("Start requires a verified stopped service")
        started_at = time.time()
        expected_pid = None
        if self.c.get("start_argv"):
            logs = self.root / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            with (logs / "manager-model.out.log").open("ab") as out, (logs / "manager-model.err.log").open("ab") as err:
                p = subprocess.Popen(self.c["start_argv"], cwd=self.root, stdout=out, stderr=err,
                                     stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
            expected_pid = p.pid
        elif self.model_id == "h3" and self.c["kind"] == "comfyui":
            cmd = [self.c["exe"], "-s", str(self.root / "work/setup/launcher.py"), "start", "--no-browser"]
            proc = subprocess.run(cmd, cwd=self.root, capture_output=True, timeout=210, creationflags=NO_WINDOW)
            if proc.returncode:
                raise RuntimeError("H3 official launcher failed; inspect its local logs")
            try:
                pid_file = self.root / "work/logs/comfyui.pid"
                launch_file = self.root / "work/logs/comfyui-launch.json"
                if min(pid_file.stat().st_mtime, launch_file.stat().st_mtime) < started_at:
                    raise ValueError("launcher records predate this start")
                expected_pid = int(pid_file.read_text(encoding="ascii").strip())
                launch = json.loads(launch_file.read_text(encoding="utf-8"))
                if launch.get("pid") != expected_pid or expected_pid <= 0:
                    raise ValueError("launcher PID records disagree")
            except (OSError, ValueError, TypeError):
                raise RuntimeError("H3 launch ownership could not be verified; process retained for inspection") from None
        elif self.model_id == "muse" and self.c["kind"] == "llamacpp":
            model = self.root / "models/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
            vision = self.root / "models/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf"
            if not model.is_file() or not vision.is_file():
                raise RuntimeError("Muse model files are missing")
            args = [self.c["exe"], "-m", str(model), "--mmproj", str(vision), "--alias", "muse-glimmer",
                    "-ngl", "99", "-c", "65536", "-np", "1", "--flash-attn", "on", "--host", "127.0.0.1",
                    "--port", "8085", "--api-key-file", str(self.root / "config/api-key.txt"),
                    "--chat-template-kwargs", '{"reasoning_strength":"low"}', "--temp", "1.0", "--top-p", "0.95", "--top-k", "64"]
            with (self.root / "logs/manager-model.out.log").open("ab") as out, (self.root / "logs/manager-model.err.log").open("ab") as err:
                p = subprocess.Popen(args, cwd=self.root, stdout=out, stderr=err, stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
            expected_pid = p.pid
            (self.root / "data/model.pid").write_text(str(p.pid), encoding="ascii")
        else:
            raise ValueError("Registered model requires an explicit start_argv")
        deadline = time.time()+180
        while time.time() < deadline:
            state = self.inspect()
            if state["state"] == "ready":
                identity = state["identity"]
                if (identity.get("pid") != expected_pid or
                        not isinstance(identity.get("created"), (int, float)) or
                        not math.isfinite(identity["created"]) or identity["created"] < started_at - .1):
                    raise RuntimeError("Ready service does not match this launch; process retained without ownership")
                return state["identity"]
            if state["state"] == "stopped":
                raise RuntimeError("Model exited during startup")
            time.sleep(1)
        raise RuntimeError("Startup timed out; service is retained for inspection")

    def stop(self, identity):
        # Two idle checks and identity revalidation narrow, but cannot eliminate,
        # races with clients bypassing the manager. No arbitrary process tree kill.
        for _ in range(2):
            s = self.inspect()
            if s["state"] == "stopped":
                return {"state": "stopped", "already_stopped": True}
            if s["state"] != "ready" or s["busy"] is not False or s["identity"] != identity:
                raise RuntimeError("Service changed or is busy/unknown; stop refused")
            time.sleep(.15)
        if not same_process(identity):
            raise RuntimeError("Process identity changed; stop refused")
        p = psutil.Process(identity["pid"])
        p.terminate()
        p.wait(timeout=20)
        after = self.inspect()
        if after["state"] != "stopped":
            raise RuntimeError("Service stop could not be verified")
        return after
