"""Bounded, credential-free health summaries with explicit evidence windows."""
import collections
import copy


def summarize(state, resources, last_sample, now, worker_error=None):
    works = state.get('works', [])
    done = [w for w in works if w.get('state') == 'recorded']
    profiles = state.get('learning', {}).get('profiles', {})
    observed = [w for w in done if w.get('measurement', {}).get('busy_samples', 0) > 0]
    measured = [w for w in done if w.get('feedback', {}).get('outcome') == 'completed'
                and w.get('measurement', {}).get('busy_samples', 0) >= 3
                and all(w['measurement'].get('peak_extra_'+r+'_gib') is not None for r in ('ram','commit','vram'))]
    guard = {k: copy.deepcopy(v) for k, v in state.get('resource_guard', {}).items() if k != 'previous'}
    age = max(0., now-last_sample) if last_sample else None
    old_lows = []
    for lease in state.get('leases', []):
        m = lease.get('measurements', {})
        low = {r: m[f] for r, f, floor in (
            ('ram', 'minimum_ram_available_gib', 12), ('commit', 'minimum_commit_free_gib', 12))
            if isinstance(m.get(f), (int,float)) and m[f] < guard.get('floors', {}).get(r, floor)}
        if low:
            old_lows.append({'model': lease['model'], 'created': lease['created'], 'minimum_free_gib': low})
    issues = []
    if worker_error: issues.append({'severity': 'error', 'code': 'monitor_error'})
    if age is None or age > 20: issues.append({'severity': 'error', 'code': 'stale_monitor'})
    if resources.get('errors'): issues.append({'severity': 'warning', 'code': 'telemetry_errors'})
    if guard.get('blocked'): issues.append({'severity': 'warning', 'code': 'admission_paused', 'reason': guard.get('reason')})
    if old_lows: issues.append({'severity': 'warning', 'code': 'historical_headroom_breach'})
    if done and len(observed)/len(done) < .8:
        issues.append({'severity': 'warning', 'code': 'low_inference_observation_coverage'})
    return {
        'state': 'error' if any(x['severity']=='error' for x in issues) else ('attention' if issues else 'ok'),
        'issues': issues, 'monitor_sample_age_seconds': round(age, 2) if age is not None else None,
        'resource_guard': guard,
        'work_window': {'scope': 'retained terminal work only; not lifetime totals', 'runs': len(done),
            'first_created': min((w['created'] for w in done), default=None),
            'last_created': max((w['created'] for w in done), default=None),
            'native_busy_observed_runs': len(observed), 'measured_completed_runs': len(measured),
            'unobserved_runs': len(done)-len(observed),
            'sampling_window_runs': sum(w.get('sampling', {}).get('window_samples', 0)>0 for w in done),
            'discontinuous_sampling_runs': sum(bool(w.get('sampling', {}).get('discontinuous')) for w in done),
            'inference_observation_percent': round(100*len(observed)/len(done), 1) if done else None,
            'outcomes': dict(collections.Counter(w.get('feedback', {}).get('outcome', 'unknown') for w in done)),
            'quality': dict(collections.Counter(w.get('feedback', {}).get('quality', 'unknown') for w in done))},
        'learning': {'scope': 'all retained active profiles; archived policies counted separately',
            'profiles': len(profiles), 'archived_profiles': len(state.get('learning', {}).get('archives', {})),
            'profiles_with_measurements': sum(p['policy'].get('measured_runs', 0)>0 for p in profiles.values())},
        'historical_low_headroom_leases': old_lows[-16:],
        'limits': ['host/device peaks include other programs; process attribution is not implied',
                   'sampling a prepared window does not prove native inference executed',
                   'unknown quality and desktop responsiveness are never inferred as passed',
                   'admission guard cannot hard-isolate memory or preempt an in-flight kernel']}
