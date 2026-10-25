"""Bounded adaptive control and automatic observations for the durable broker.

Quality comes from workflow QA. Timer drift and memory telemetry are pressure
proxies, not a subjective quality/desktop responsiveness certificate.
"""
import copy
import hashlib
import json
import math
import time
import uuid
import os
from pathlib import Path
from adaptive import AdaptivePolicy, normalize_task, workload_key, resolve_kind, _budget_exceeds
import learner
from protection import observe_guard
from health_report import summarize


def free_metrics(resources):
    gpu = next((g for g in resources.get('gpus', []) if g.get('index') == 0), {})
    values={'ram_available_gib': resources.get('ram_available_gib'),
            'commit_free_gib': resources.get('commit_free_gib'),
            'free_vram_gib': gpu.get('free_gib')}
    return {k:v if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0 else None
            for k,v in values.items()}


class AdaptiveBroker:
    def register_model(self, model, manifest, replace=False):
        from registry import register_config
        from adapters import Adapter
        # Registration is a deliberate configuration operation. It never starts
        # a new model, and cannot disrupt another chat's task.
        with self.refresh_lock, self.lock:
            if (self.stop_event.is_set() or self.lifecycle_action or
                any(x['state'] in {'queued','starting','active','orphaned','releasing'} for x in self.state['leases']) or
                any(x['state'] in {'queued','running'} for x in self.state['cleanup'])):
                raise ValueError('Register only while the manager is quiescent')
            if not self.config_path or not self.config_path.is_file():
                raise ValueError('Persistent configuration path is not available')
            if model in self.adapters and self.adapters[model].inspect()['state']!='stopped':
                raise ValueError('Existing model must be stopped before changing its registration')
            saved=json.loads(self.config_path.read_text(encoding='utf-8'))
            updated=register_config(saved,model,manifest,replace=replace)
            adapter=Adapter(model,updated['models'][model])
            if adapter.inspect()['state']!='stopped':
                raise ValueError('Registration port or matching service is already active; do not adopt it implicitly')
            backup=self.root/('config-backup-'+str(time.time_ns())+'.json')
            backup.write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding='utf-8')
            tmp=self.config_path.with_suffix('.json.new')
            tmp.write_text(json.dumps(updated,ensure_ascii=False,indent=2),encoding='utf-8')
            os.replace(tmp,self.config_path)
            self.config=updated
            self.adapters[model]=adapter
            self.models[model]=adapter.inspect()
            self.init_adaptive()
            self.save()
            self.event('model_registered',{'model':model,'kind':updated['models'][model]['kind'],'replaced':replace})
            return {'registered':model,'started':False,'kind':updated['models'][model]['kind'],
                    'learning':'new task/config fingerprint starts with conservative baseline'}

    def init_adaptive(self):
        # Signature metadata is read once, without loading large model weights.
        signature = []
        self.artifact_paths=[]
        for model, c in self.config['models'].items():
            root = Path(c['root'])
            paths = [Path(c['exe'])]
            argv=c.get('start_argv',[])
            for index,arg in enumerate(argv):
                if index and argv[index-1] in {'-m','--model','--mmproj','--lora'}:
                    paths.append(Path(arg))
                elif arg.lower().endswith('.py') and Path(arg).is_absolute():
                    paths.append(Path(arg))
            if c['kind'] == 'llamacpp':
                paths += list((root / 'models').glob('*.gguf'))
            else:
                paths += [root/'work/setup/launcher.py', root/'work/setup/h3-memory-policy.json']
                base = root/'ComfyUI_windows_portable/ComfyUI/models'
                for folder in ('diffusion_models', 'text_encoders', 'vae', 'audio_encoders'):
                    paths += list((base/folder).rglob('*.safetensors'))
            for path in sorted(set(paths)):
                self.artifact_paths.append(path)
                try:
                    st = path.stat()
                    signature.append((str(path), st.st_size, st.st_mtime_ns))
                except OSError:
                    signature.append((str(path), 'missing'))
        self.config = copy.deepcopy(self.config)
        self.config['workload_fingerprint'] = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
        self.policy = AdaptivePolicy(self.config)
        self.state.setdefault('learning', learner.initial_state())
        self.state.setdefault('works', [])
        guard = self.state.setdefault('resource_guard', {})
        guard.pop('previous', None)
        guard['quiet_since'] = None  # A restart is not a continuous quiet interval.
        for work in self.state['works']:
            if work['state']=='observing':
                # A process restart can miss the true peak; retain the job but
                # never calibrate capacity from a discontinuous observation.
                for resource in ('ram','commit','vram'):
                    work['measurement']['peak_extra_'+resource+'_gib']=None
        self.load_state = {}
        self.previous_resources, self.previous_models = {}, {}
        self.db.execute('PRAGMA cache_size=-2048')
        self.db.execute('PRAGMA journal_size_limit=4194304')
        self.db.execute('PRAGMA wal_autocheckpoint=256')

    def refresh_artifact_signature(self):
        signature=[]
        for path in self.artifact_paths:
            try:
                st=path.stat()
                signature.append((str(path),st.st_size,st.st_mtime_ns))
            except OSError:
                signature.append((str(path),'missing'))
        value=hashlib.sha256(json.dumps(signature).encode()).hexdigest()
        self.config['workload_fingerprint']=value
        self.policy.config['workload_fingerprint']=value

    def cache_state(self, model):
        managed = self.state['managed'].get(model, {})
        current = self.models.get(model, {})
        if managed.get('identity') != current.get('identity') or not managed.get('identity'):
            return 'unknown'
        return managed.get('cache_state', 'cold' if self.config['models'][model]['kind']=='comfyui' else 'warm')

    def profile_for(self, model, task, cache=None):
        wk = workload_key(model, task, self.config)
        pk = learner.profile_key(model, wk, self.config['workload_fingerprint'], cache or self.cache_state(model))
        learned = learner.get_policy(self.state['learning'], pk)
        if learned:
            learned['workload_key'] = wk
        return pk, learned

    def plan_for(self, lease, cold=False, task=None, admission=False):
        task = lease.get('task', {}) if task is None else task
        _, learned = self.profile_for(lease['model'], task)
        if not admission:
            # Already allocated working memory must not be requested a second time.
            # Retain learned desktop biases, remove only incremental task budgets.
            task = {k:v for k,v in task.items() if not k.startswith('estimated_peak_')}
            learned = dict(learned, runs=0, samples=0)
            learned['workload_key'] = workload_key(lease['model'],task,self.config)
        fallback = None
        if admission:
            budgets = self.config['models'][lease['model']].get('unprofiled_peak_extra_gib', {})
            cache = 'warm' if not cold and self.cache_state(lease['model']) == 'warm' else 'cold'
            fallback = budgets.get(cache)
        plan = self.policy.plan(lease['model'], task, self.resources, self.models, learned=learned, cold=cold, fallback=fallback,
                                startup_admission=admission and cache == 'cold')
        guard = self.state.get('resource_guard', {})
        if guard.get('blocked'):
            plan['reasons'].append('resource guard: '+guard['reason'])
            plan['proceed'] = False
            if plan.get('retryable') is not False:
                plan['admission_state'] = 'waiting'
        return plan

    def sampling_interval(self):
        base = self.config['sample_seconds']
        if any(w['state']=='observing' for w in self.state['works']):
            return min(base, self.policy._option('short_task_sample_seconds', .5, .25))
        if any(x['state'] in {'starting','active','orphaned','releasing','queued'} for x in self.state['leases']):
            return min(base, self.policy._option('active_sample_seconds', 1, .25))
        return self.policy._option('idle_sample_seconds', 10, .25)

    def health_report(self):
        with self.lock:
            report = summarize(self.state, self.resources, self.last_sample, time.time(), self.worker_error)
            report['sampling'] = {'next_interval_seconds': self.sampling_interval(),
                                  'prepared_interval_seconds': self.policy._option('short_task_sample_seconds', .5, .25),
                                  'active_interval_seconds': self.policy._option('active_sample_seconds', 1, .25),
                                  'idle_interval_seconds': self.policy._option('idle_sample_seconds', 10, .25)}
            row = self.db.execute('SELECT COUNT(*),MIN(time),MAX(time) FROM events').fetchone()
            report['event_window'] = {'count': row[0], 'first_time': row[1], 'last_time': row[2], 'scope': 'bounded retained events'}
            return report

    def adaptive_status(self):
        return {'startup_vram_admission_gib': {m: c.get('startup_vram_admission_gib') for m, c in self.config['models'].items()},
                'mode': 'quality_preserving_adaptive', 'load': copy.deepcopy(self.load_state),
                'observations': len(self.state['works']),
                'manager_rss_mib': self.resources.get('manager_rss_mib'),
                'memory_target_mib': self.config.get('adaptive', {}).get('manager_memory_target_mib', 96),
                'record_limit': 128, 'sample_rows_limit': 1440,
                'quality_assessment': 'workflow QA; unknown is never passed'}

    def active_work(self, lease):
        return next((w for w in reversed(self.state['works'])
                     if w['lease_id'] == lease['id'] and w['state'] == 'observing'), None)

    def start_work_record(self, lease, work_id, task, automatic=False, before=None):
        pk, _ = self.profile_for(lease['model'], task)
        now = time.time()
        w = {'id': work_id, 'lease_id': lease['id'], 'model': lease['model'], 'task': copy.deepcopy(task),
             'profile_key': pk, 'workload_key': workload_key(lease['model'], task, self.config),
             'cache_state': self.cache_state(lease['model']), 'created': now,
             'state': 'observing', 'automatic': automatic, 'seen_busy': False,
             'measurement': {'before': free_metrics(before or self.resources), 'busy_samples': 0,
                             'elapsed_seconds': 0, 'pressure_observed': False, 'pressure_resources': [],
                             'peak_extra_ram_gib': 0., 'peak_extra_commit_gib': 0., 'peak_extra_vram_gib': 0.}}
        w['sampling'] = {'window_samples': 0, 'last_sample_time': (before or self.resources).get('time'),
                         'max_gap_seconds': 0., 'discontinuous': False,
                         'target_interval_seconds': self.policy._option('short_task_sample_seconds', .5, .25)}
        self.state['works'].append(w)
        self.event('work_started', {'id': work_id, 'model': lease['model'], 'automatic': automatic,
                                    'workload_key': w['workload_key']})
        return w

    def record_work(self, w, outcome='unknown', quality='unknown', desktop_responsive=None):
        feedback = {'outcome': outcome, 'quality': quality, 'desktop_responsive': desktop_responsive}
        measurement = copy.deepcopy(w['measurement'])
        measurement['elapsed_seconds'] = max(0, time.time()-w['created'])
        # Lease-scoped IDs prevent callers in separate chats sharing work_id.
        result = learner.record_run(self.state['learning'], w['profile_key'], w['lease_id']+':'+w['id'], measurement, feedback)
        w.update(state='recorded', finished=time.time(), feedback=feedback, measurement=measurement)
        self.event('work_recorded', {'id': w['id'], 'outcome': outcome, 'quality': quality,
                                    'busy_samples': measurement['busy_samples'], 'learning': result})
        return result

    def adaptive_refresh(self, resources, models):
        self.load_state = self.policy.observe(resources, models)
        previous_guard = self.state.get('resource_guard', {})
        floors = self.policy._floors()
        guard = observe_guard(previous_guard, {'time': resources.get('time'), **free_metrics(resources)},
                              {r: floors[r+'_gib'] for r in ('ram','commit','vram')},
                              self.policy._option('protection_recovery_seconds', 10, 0),
                              self.policy._option('forecast_guard_seconds', 10, 0))
        self.state['resource_guard'] = guard
        if guard.get('blocked') != previous_guard.get('blocked', False) or guard.get('breached_resources') != previous_guard.get('breached_resources', []):
            self.event('resource_guard_changed', {k:guard.get(k) for k in ('blocked','reason','breached_resources','forecast_resources','breach_episodes')})
        for lease in self.state['leases']:
            if lease['state'] in {'failed','cancelled','released','expired'}:
                w=self.active_work(lease)
                if w:
                    self.record_work(w, outcome='failed' if lease['state']=='failed' else 'unknown')
            if lease['state'] not in {'active', 'orphaned'}:
                continue
            model = lease['model']
            s = models.get(model, {})
            if s.get('identity') != lease.get('identity') or s.get('state') != 'ready':
                continue
            w = self.active_work(lease)
            if s.get('busy') is True and w is None:
                # Backward-compatible producers are still observed automatically.
                before = self.previous_resources if self.previous_models.get(model, {}).get('busy') is False else resources
                w = self.start_work_record(lease, 'observed-'+str(uuid.uuid4()), lease.get('task', {}), True, before)
            if w:
                sampling = w.setdefault('sampling', {'window_samples': 0, 'max_gap_seconds': 0.})
                at, last = resources.get('time'), sampling.get('last_sample_time')
                fresh = isinstance(at, (int,float)) and (last is None or at > last)
                if fresh:
                    sampling['window_samples'] += 1
                    values = free_metrics(resources)
                    for r, field in (('ram','ram_available_gib'),('commit','commit_free_gib'),('vram','free_vram_gib')):
                        original, current = w['measurement']['before'].get(field), values.get(field)
                        if isinstance(original, (int,float)) and isinstance(current, (int,float)):
                            peak = 'window_peak_extra_'+r+'_gib'
                            sampling[peak] = max(sampling.get(peak, 0.), original-current, 0.)
                    # Window telemetry is useful for host protection/reporting;
                    # only verified native busy samples enter the learner below.
                    if last is not None:
                        gap = at-last
                        sampling['max_gap_seconds'] = max(sampling['max_gap_seconds'], gap)
                        if gap > max(10, self.config['sample_seconds']*2):
                            sampling['discontinuous'] = True
                            for r in ('ram','commit','vram'):
                                w['measurement']['peak_extra_'+r+'_gib'] = None
                    sampling['last_sample_time'] = at
                else:
                    continue  # Never turn duplicate/stale telemetry into busy evidence.
            if w and s.get('busy') is True:
                w['seen_busy'] = True
                m = w['measurement']
                m['busy_samples'] += 1
                values = free_metrics(resources)
                for resource, field in (('ram','ram_available_gib'),('commit','commit_free_gib'),('vram','free_vram_gib')):
                    previous, current = m['before'].get(field), values.get(field)
                    field='peak_extra_'+resource+'_gib'
                    valid=lambda x:isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>=0
                    if valid(previous) and valid(current) and m[field] is not None:
                        m[field] = max(m[field], previous-current, 0)
                    else:
                        m[field] = None
                running_plan = self.plan_for(lease, task=w['task'])
                m['pressure_observed'] |= bool(running_plan['reasons']) or self.load_state.get('pressure_tier',0)>0
                # Learn shortages against fixed desktop floors. Dynamic reserves
                # include prior learned bias and CPU/timer protection; using them
                # as evidence would let those safeguards reinforce themselves.
                # Old in-flight records without attribution keep legacy semantics.
                if 'pressure_resources' in m:
                    attributed = set(m['pressure_resources'])
                    for resource, field in (('ram','ram_available_gib'),('commit','commit_free_gib'),('vram','free_vram_gib')):
                        current = values.get(field)
                        minimum = running_plan['budget_breakdown'][resource+'_gib']['base_floor_gib']
                        if current is not None and _budget_exceeds(minimum, current):
                            attributed.add(resource)
                    m['pressure_resources'] = sorted(attributed)
                    m['pressure_observed'] |= bool(attributed)
            if w and w['seen_busy'] and s.get('busy') is False:
                if model in self.state['managed']:
                    self.state['managed'][model]['cache_state'] = 'warm'
                if w['automatic']:
                    self.record_work(w)  # idle says nothing about successful output/QA
        # A requested asynchronous unload is only considered observed when device
        # use actually falls. It is never reported as exact per-model freed bytes.
        for model, managed in self.state['managed'].items():
            pending = managed.get('cache_release')
            if pending and pending.get('state') == 'requested':
                now_free = free_metrics(resources).get('free_vram_gib')
                before_free = pending.get('before_free_vram_gib')
                if isinstance(now_free, (int,float)) and isinstance(before_free,(int,float)) and now_free > before_free+.25:
                    pending.update(state='device_free_increased', observed_free_increase_gib=now_free-pending['before_free_vram_gib'])
                    managed['cache_state'] = 'unknown'  # whole-device telemetry cannot prove exact cache contents
                elif time.time()-pending['time'] > 20:
                    pending['state'] = 'unconfirmed'
        self.previous_resources, self.previous_models = copy.deepcopy(resources), copy.deepcopy(models)
        # Bound resident records. SQLite events retain summaries separately.
        # A soft footprint budget trims only old terminal metadata; never an
        # active lease or process. Interpreter/MCP RSS is reported, not hard capped.
        target=self.config.get('adaptive',{}).get('manager_memory_target_mib',96)
        compact=resources.get('manager_rss_mib',0)>target
        terminal = [w for w in self.state['works'] if w['state'] != 'observing'][-(32 if compact else 128):]
        self.state['works'] = terminal + [w for w in self.state['works'] if w['state'] == 'observing']
        old = [x for x in self.state['leases'] if x['state'] in {'released','expired','cancelled','failed'}][-(64 if compact else 256):]
        self.state['leases'] = old + [x for x in self.state['leases'] if x['state'] not in {'released','expired','cancelled','failed'}]
        self.state['cleanup'] = self.state['cleanup'][-32:]

    def prepare_work(self, lease_id, token, work_id, task=None):
        if not isinstance(work_id,str) or not work_id or len(work_id)>160:
            raise ValueError('work_id must be 1..160 characters')
        # Fresh sensor and native idle check at each admission boundary.
        self.refresh()
        with self.lock:
            lease = self.authenticated(lease_id, token)
            normalized = normalize_task(lease['model'], lease.get('task', {}) if task is None else task,
                                        resolve_kind(self.config,lease['model']))
            if normalized.get('context_tokens',0)>self.config['models'][lease['model']].get('context_capacity',65536):
                raise ValueError('Task context exceeds installed capacity')
            old = next((w for w in self.state['works'] if w['lease_id']==lease_id and w['id']==work_id),None)
            if old and old['task'] != normalized:
                raise ValueError('work_id already used for different task')
            if old and old['state'] != 'observing':
                return {'proceed':False,'work_id':work_id,'reason':'work already finished; inspect original output before retry'}
            if old and old.get('seen_busy'):
                return {'proceed':False,'work_id':work_id,'reason':'work already observed executing; complete or inspect original task before retry'}
            s = self.models.get(lease['model'], {})
            plan = self.plan_for(lease, task=normalized, admission=True)
            current = self.active_work(lease)
            reasons = list(plan['reasons'])
            if not self.public_lease(lease)['proceed']:
                reasons.append('lease not ready or its resource guard is paused')
            if self.lifecycle_action:
                reasons.append('resource adjustment in progress')
            if current and current['id'] != work_id:
                reasons.append('another work item is still registered')
            if s.get('busy') is not False:
                reasons.append('model service is busy or unverified')
            if reasons:
                # Keep idle cache reclaim responsive to the pending inference's
                # budget, not just to the already-allocated running floor.
                if plan['reasons'] and plan.get('retryable') is not False and not current and s.get('busy') is False:
                    lease['pending_admission'] = {'task': copy.deepcopy(normalized)}
                    self.save()
                    self.monitor_wake.set()
                return {'proceed':False,'work_id':work_id,'reason':'; '.join(reasons),'plan':plan,
                        'retryable':plan.get('retryable',True), 'admission_state':plan.get('admission_state','waiting')}
            if not old:
                self.start_work_record(lease,work_id,normalized)
            lease['task'] = normalized
            lease.pop('pending_admission', None)
            self.save()
            self.monitor_wake.set()
            return {'proceed':True,'work_id':work_id,'plan':plan,'quality_parameters_changed':False,
                    'retryable':True,'admission_state':'ready'}

    def complete_work(self, lease_id, token, work_id, outcome='completed', quality='unknown', desktop_responsive=None):
        self.refresh()  # Capture end-of-window headroom, including short tasks.
        with self.lock:
            lease = self.authenticated(lease_id,token)
            w = next((w for w in self.state['works'] if w['lease_id']==lease_id and w['id']==work_id),None)
            if not w:
                raise ValueError('Unknown work_id')
            if w['state'] == 'recorded':
                return self.feedback(lease_id,token,work_id,'completion',outcome,quality,desktop_responsive)
            # Reinspect rather than infer idleness from an old monitor snapshot.
            s = self.adapters[lease['model']].inspect()
            if s.get('state') != 'ready' or s.get('busy') is not False or s.get('identity') != lease.get('identity'):
                raise ValueError('Service busy, changed or unverified; completion deferred')
            result = self.record_work(w,outcome,quality,desktop_responsive)
            if w['seen_busy'] and lease['model'] in self.state['managed']:
                self.state['managed'][lease['model']]['cache_state']='warm'
            self.save()
            return result

    def feedback(self, lease_id, token, work_id, feedback_id, outcome='completed', quality='unknown', desktop_responsive=None):
        with self.lock:
            self.authenticated(lease_id,token)
            w = next((w for w in self.state['works'] if w['lease_id']==lease_id and w['id']==work_id),None)
            if not w or w['state'] != 'recorded':
                raise ValueError('Feedback requires a recorded work item')
            result = learner.report_feedback(self.state['learning'],w['profile_key'],w['lease_id']+':'+work_id,feedback_id,
                {'outcome':outcome,'quality':quality,'desktop_responsive':desktop_responsive})
            job=self.state['learning'].get('profiles',{}).get(w['profile_key'],{}).get('jobs',{}).get(w['lease_id']+':'+work_id)
            if job:
                w['feedback']=copy.deepcopy(job['feedback'])
            self.event('work_feedback',{'id':work_id,'result':result})
            self.save()
            return result

    def learning_status(self, model=None):
        if model is not None and model not in self.adapters:
            raise ValueError('Unknown model')
        with self.lock:
            recent = [w for w in self.state['works'] if model is None or w['model']==model][-20:]
            keys = [pk for pk in self.state['learning'].get('profiles', {}) if model is None or pk.startswith(model+':')]
            recent_keys = {w['profile_key'] for w in recent}
            return {'automatic':True,'status':'quality labels come from workflow QA; absent labels remain unknown',
                    'profile_scope':'all retained active profiles; recent_work is limited to 20',
                    'recent_work':[ {k:copy.deepcopy(w[k]) for k in ('id','model','task','state','profile_key','measurement','feedback','sampling') if k in w} for w in recent],
                    'profiles':[{ 'profile_key':pk,'policy':learner.get_policy(self.state['learning'],pk),
                                  'revisions':learner.history(self.state['learning'],pk) if pk in recent_keys else []}
                                for pk in keys], 'health_report':self.health_report()}

    def rollback_policy(self, lease_id, token, work_id, target_revision, reason):
        with self.lock:
            self.authenticated(lease_id,token)
            w = next((w for w in self.state['works'] if w['lease_id']==lease_id and w['id']==work_id),None)
            if not w:
                raise ValueError('Unknown work_id')
            result = learner.rollback(self.state['learning'],w['profile_key'],target_revision,reason)
            self.event('policy_rollback',result)
            self.save()
            return result

    def adaptive_action(self, lease):
        model = lease['model']
        managed = self.state['managed'].get(model)
        s = self.models.get(model,{})
        if not managed or managed.get('identity') != s.get('identity') or s.get('state') != 'ready':
            return None
        plan = self.plan_for(lease)
        pending = lease.get('pending_admission')
        if pending and not self.active_work(lease) and s.get('busy') is False:
            plan = self.plan_for(lease, task=pending['task'], admission=True)
        profile = plan['runtime_profile']
        if hasattr(self.adapters[model],'apply_runtime') and managed.get('runtime_profile') != profile and not managed.get('runtime_error'):
            return ('runtime',model,s['identity'],lease['id'])
        if (self.config['models'][model]['kind']=='comfyui' and hasattr(self.adapters[model],'free_idle_cache') and plan['reasons']
                and s.get('busy') is False and not self.active_work(lease)
                and time.time()-managed.get('last_cache_release',0)>120 and not managed.get('cache_error')):
            return ('free_cache',model,s['identity'],lease['id'])
        return None

    def perform_adaptive_action(self, kind, model, identity, ref):
        with self.lock:
            managed = self.state['managed'].get(model,{})
            lease = next(x for x in self.state['leases'] if x['id']==ref)
            if managed.get('identity') != identity:
                raise RuntimeError('Owned process changed')
            if kind=='runtime':
                profile = self.plan_for(lease)['runtime_profile']
                baseline = copy.deepcopy(managed.get('runtime_baseline'))
            elif self.active_work(lease):
                raise RuntimeError('Work started; cache release deferred')
        if kind=='runtime':
            result = self.adapters[model].apply_runtime(identity,profile,baseline)
            with self.lock:
                managed.update(runtime_profile=profile,runtime_baseline=result['baseline'],runtime_applied_at=time.time())
                self.event('runtime_adjusted',{'model':model,'profile':profile,'reserved_logical_cpus':result['reserved_logical_cpus']})
        else:
            self.adapters[model].free_idle_cache(identity)
            with self.lock:
                managed.update(last_cache_release=time.time(),cache_state='unknown',cache_release={
                    'state':'requested','time':time.time(),'before_free_vram_gib':free_metrics(self.resources)['free_vram_gib']})
                self.event('cache_release_requested',{'model':model,'reason':'resource pressure at verified idle boundary'})
