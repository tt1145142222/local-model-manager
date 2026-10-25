"""Admission-only host guard. Never preempts a process or changes model quality."""
import copy
import math
from adaptive import _budget_exceeds


FIELDS = {'ram': 'ram_available_gib', 'commit': 'commit_free_gib', 'vram': 'free_vram_gib'}


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def observe_guard(state, sample, floors, recovery_seconds=10, horizon_seconds=10):
    """Persist bounded breach episodes; recovery needs continuous, fresh telemetry.

    Raw decline forecasts are risk signals, not attribution to any one process.
    A restart/gap cannot count as a quiet observation interval.
    """
    now = sample.get('time')
    if not number(now):
        return copy.deepcopy(state)
    previous = state.get('previous', {})
    if number(previous.get('time')) and now <= previous['time']:
        return copy.deepcopy(state)
    elapsed = now - previous['time'] if number(previous.get('time')) else None
    continuous = elapsed is not None and 0 < elapsed <= 15
    values = {r: sample.get(f) if number(sample.get(f)) else None for r, f in FIELDS.items()}
    missing = [r for r, v in values.items() if not number(v)]
    breached = [r for r, v in values.items() if number(v) and _budget_exceeds(floors[r], v)]
    forecast = []
    for r in ('ram', 'commit'):
        old, current = previous.get(r), values[r]
        if continuous and number(old) and number(current):
            rate = max(0., (old-current)/elapsed)
            if rate >= .15 and current-rate*horizon_seconds < floors[r]:
                forecast.append(r)
    result = copy.deepcopy(state)
    counts = result.setdefault('breach_episodes', {r: 0 for r in FIELDS})
    for r in breached:
        if r not in state.get('breached_resources', []):
            counts[r] = counts.get(r, 0)+1
    result.setdefault('since', now)
    result.update(breached_resources=breached, forecast_resources=forecast, missing_resources=missing)
    if breached:
        result['last_breach_time'] = now
    risky = bool(breached or forecast or missing)
    recovering = bool(state.get('blocked'))
    quiet = all(number(v) and v >= floors[r] + (0.5 if r == 'vram' else 2) for r, v in values.items())
    quiet_since = state.get('quiet_since') if continuous else None
    if risky or not quiet:
        quiet_since = None
    elif recovering and quiet_since is None:
        quiet_since = now
    blocked = risky or (recovering and (quiet_since is None or now-quiet_since < recovery_seconds))
    result.update(blocked=blocked, quiet_since=quiet_since if blocked else None,
                  recovery_seconds=recovery_seconds, forecast_horizon_seconds=horizon_seconds,
                  current=values, floors=copy.deepcopy(floors),
                  previous={'time': now, **values})
    if blocked:
        details = []
        if breached: details.append('desktop floor breached: '+', '.join(breached))
        if forecast: details.append('declining headroom forecasts a floor breach: '+', '.join(forecast))
        if missing: details.append('headroom telemetry missing: '+', '.join(missing))
        if not details: details.append('waiting for stable headroom recovery')
        result['reason'] = '; '.join(details)
    else:
        result['reason'] = 'headroom guard ready'
    return result
