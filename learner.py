"""Bounded, deterministic learning of scheduling safeguards, never generation settings.

The broker owns/persists this JSON-compatible state and supplies measurements.
Reported quality and desktop responsiveness are evidence, not inferred metrics.
"""
import copy
import hashlib
import json
import math

MAX_PROFILES = 32
MAX_JOBS = 16
MAX_RUNS = 16
MAX_REVISIONS = 16
MAX_FEEDBACK_PER_JOB = 4
MAX_ARCHIVES = 32
SEEN_BYTES = 32768
BIAS_MAX = {"ram": 8.0, "commit": 8.0, "vram": 3.0}
BIAS_STEP = {"ram": 2.0, "commit": 2.0, "vram": 0.5}
EASE_STEP = {"ram": 0.5, "commit": 0.5, "vram": 0.125}
RESOURCES = ("ram", "commit", "vram")


def initial_state():
    return {"schema_version": 1, "profiles": {}, "archives": {}, "access_sequence": 0,
            "seen_jobs_hex": "00" * SEEN_BYTES}


def _text(value, name, limit=512):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(name + " must be a nonempty bounded string")
    return value


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not finite JSON") from exc


def profile_key(model, workload_key, config_signature, cache_state):
    """Separate exact tasks, model/config revisions, and cold/warm observations."""
    _text(model, "model", 80)
    if isinstance(workload_key, dict):
        workload_key = _canonical(workload_key)
    _text(workload_key, "workload_key", 8192)
    _text(config_signature, "config_signature", 512)
    if cache_state not in ("cold", "warm", "unknown"):
        raise ValueError("cache_state must be cold, warm or unknown")
    digest = hashlib.sha256(_canonical([model, workload_key, config_signature, cache_state]).encode()).hexdigest()
    return f"{model}:{cache_state}:{digest}"


def _base_policy(key):
    return {"workload_key": key, "runs": 0, "samples": 0, "measured_runs": 0,
            "peak_extra_ram_gib": None, "peak_extra_commit_gib": None,
            "peak_extra_vram_gib": None, "reserve_ram_bias_gib": 0.0,
            "reserve_commit_bias_gib": 0.0, "reserve_vram_bias_gib": 0.0,
            "disable_warm_cache": False, "quarantined": False, "revision": 0}


def get_policy(state, key):
    """Always return a policy dict; uncalibrated peak fields are None, never zero."""
    _text(key, "profile key")
    profile = state.get("profiles", {}).get(key) or state.get("archives", {}).get(key)
    return copy.deepcopy(profile["policy"] if profile else _base_policy(key))


def history(state, key):
    return copy.deepcopy(state.get("profiles", {}).get(key, {}).get("revisions", []))


def _number(value, name, ceiling=1048576):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= ceiling:
        raise ValueError(name + " must be a finite nonnegative number")
    return float(value)


def _measurement(value):
    if not isinstance(value, dict):
        raise ValueError("measurement must be a mapping")
    allowed = {"before", "busy_samples", "elapsed_seconds", "pressure_observed", "pressure_resources"} | {
        "peak_extra_" + r + "_gib" for r in RESOURCES}
    if set(value) - allowed:
        raise ValueError("unknown measurement fields")
    before = value.get("before")
    if before is None:
        before = {}
    if not isinstance(before, dict) or set(before) - {"ram_available_gib", "commit_free_gib", "free_vram_gib"}:
        raise ValueError("invalid before resources")
    cleaned = {"before": {k: _number(v, k) for k, v in before.items() if v is not None}}
    count = value.get("busy_samples", 0)
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 10000000:
        raise ValueError("busy_samples must be a nonnegative integer")
    cleaned["busy_samples"] = count
    elapsed = value.get("elapsed_seconds")
    cleaned["elapsed_seconds"] = None if elapsed is None else _number(elapsed, "elapsed_seconds", 31536000)
    pressure = value.get("pressure_observed", False)
    if not isinstance(pressure, bool):
        raise ValueError("pressure_observed must be boolean")
    cleaned["pressure_observed"] = pressure
    if "pressure_resources" in value:
        affected = value["pressure_resources"]
        if (not isinstance(affected, list) or len(affected) > len(RESOURCES)
                or any(not isinstance(item, str) or item not in RESOURCES for item in affected)):
            raise ValueError("pressure_resources must be a bounded list of ram, commit, vram")
        if affected and not pressure:
            raise ValueError("pressure_resources require pressure_observed=true")
        # Presence matters: [] means observed pressure had no memory attribution;
        # omission preserves the conservative behavior of historic observations.
        cleaned["pressure_resources"] = [resource for resource in RESOURCES if resource in affected]
    for resource in RESOURCES:
        field = "peak_extra_" + resource + "_gib"
        cleaned[field] = None if value.get(field) is None else _number(value[field], field)
    return cleaned


def _feedback(value, partial=False):
    if not isinstance(value, dict) or set(value) - {"outcome", "quality", "desktop_responsive"}:
        raise ValueError("invalid feedback fields")
    result = {} if partial else {"outcome": "unknown", "quality": "unknown", "desktop_responsive": None}
    if "outcome" in value:
        if value["outcome"] not in ("unknown", "completed", "oom", "failed", "cancelled"):
            raise ValueError("invalid outcome")
        result["outcome"] = value["outcome"]
    if "quality" in value:
        if value["quality"] not in ("passed", "failed", "unknown"):
            raise ValueError("invalid quality")
        result["quality"] = value["quality"]
    if "desktop_responsive" in value:
        responsive = value["desktop_responsive"]
        if responsive is not None and not isinstance(responsive, bool):
            raise ValueError("desktop_responsive must be boolean or null")
        result["desktop_responsive"] = responsive
    if partial and not result:
        raise ValueError("feedback must contain evidence")
    return result


def _seen_positions(key, job_id):
    value = hashlib.sha256(_canonical([key, job_id]).encode()).digest()
    return [int.from_bytes(value[i:i+4], "big") % (SEEN_BYTES * 8) for i in (0, 4, 8, 12)]


def _seen(state, key, job_id, add=False):
    # Bounded, no-false-negative tombstones prevent replay after old detail eviction.
    # A very rare false positive only refuses learning; it cannot weaken protection.
    bits = bytearray.fromhex(state.get("seen_jobs_hex", "00" * SEEN_BYTES))
    if len(bits) != SEEN_BYTES:
        raise ValueError("invalid learning deduplication state")
    positions = _seen_positions(key, job_id)
    found = all(bits[p // 8] & (1 << (p % 8)) for p in positions)
    if add:
        for p in positions:
            bits[p // 8] |= 1 << (p % 8)
        state["seen_jobs_hex"] = bits.hex()
    return found


def _profile(state, key):
    if state.get("schema_version") != 1 or not isinstance(state.get("profiles"), dict):
        raise ValueError("invalid learning state")
    if key not in state["profiles"]:
        if len(state["profiles"]) >= MAX_PROFILES:
            evict_key = min(state["profiles"], key=lambda k: state["profiles"][k].get("last_used", 0))
            evicted = state["profiles"].pop(evict_key)
            state.setdefault("archives", {})[evict_key] = {
                "policy": evicted["policy"], "last_relaxation": evicted["last_relaxation"],
                "last_used": evicted.get("last_used", 0)}
        archived = state.setdefault("archives", {}).pop(key, None)
        policy = copy.deepcopy(archived["policy"]) if archived else _base_policy(key)
        state["profiles"][key] = {"policy": policy, "jobs": {}, "run_ids": [],
                                  "revisions": [], "clean_streak": 0, "last_relaxation": None,
                                  "last_adverse_run": policy["runs"]}
        if archived:
            state["profiles"][key]["last_relaxation"] = copy.deepcopy(archived["last_relaxation"])
        while len(state["archives"]) > MAX_ARCHIVES:
            oldest = min(state["archives"], key=lambda k: state["archives"][k]["last_used"])
            del state["archives"][oldest]
    _touch(state, state["profiles"][key])
    return state["profiles"][key]


def _touch(state, profile):
    state["access_sequence"] = state.get("access_sequence", 0) + 1
    profile["last_used"] = state["access_sequence"]


def _revision(profile, before, reason, job_id=None):
    after = profile["policy"]
    compare_before, compare_after = copy.deepcopy(before), copy.deepcopy(after)
    compare_before.pop("revision", None)
    compare_after.pop("revision", None)
    if compare_before == compare_after:
        return False
    after["revision"] = before["revision"] + 1
    profile["revisions"].append({"revision": after["revision"], "reason": reason,
                                  "job_id": job_id, "before": before, "after": copy.deepcopy(after)})
    del profile["revisions"][:-MAX_REVISIONS]
    return True


def _measured(job):
    m = job["measurement"]
    return (job["feedback"]["outcome"] == "completed" and m["busy_samples"] >= 3
            and m["elapsed_seconds"] is not None and m["elapsed_seconds"] > 0
            and len(m["before"]) == 3
            and all(m["peak_extra_" + r + "_gib"] is not None for r in RESOURCES))


def _clean(job):
    f = job["feedback"]
    # Quiet sampled load plus workflow QA can support cautious tuning unattended.
    # None is not a claim that the user experienced a responsive desktop.
    return (_measured(job) and f["quality"] == "passed" and f["desktop_responsive"] is not False
            and not job["measurement"]["pressure_observed"] and not job["adverse_applied"])


def _learn(profile, job_id):
    p, job = profile["policy"], profile["jobs"][job_id]
    feedback, measurement = job["feedback"], job["measurement"]
    reasons = []
    adverse = []
    if measurement["pressure_observed"] or feedback["desktop_responsive"] is False:
        adverse.append("headroom")
    if feedback["outcome"] == "oom":
        adverse.append("oom")
    if feedback["quality"] == "failed":
        adverse.append("quality")
    new_adverse = [reason for reason in adverse if reason not in job["adverse_applied"]]
    if feedback["outcome"] == "oom" or feedback["desktop_responsive"] is False:
        bias_resources = list(RESOURCES)  # Cause is unknown; never invent specificity.
    elif measurement["pressure_observed"]:
        bias_resources = measurement.get("pressure_resources", list(RESOURCES))
    else:
        bias_resources = []
    already_biased = job.get("bias_applied_resources")
    if already_biased is None:
        # Old jobs predate resource attribution and applied all three biases.
        already_biased = list(RESOURCES) if set(job["adverse_applied"]) & {"headroom", "oom"} else []
    new_bias_resources = [resource for resource in bias_resources if resource not in already_biased]
    if new_adverse or new_bias_resources:
        previous = profile["last_relaxation"]
        rollback_resources = list(RESOURCES) if "quality" in new_adverse else bias_resources
        pending_rollback = previous.get("pending_resources", list(RESOURCES)) if previous else []
        rollback_resources = [resource for resource in rollback_resources if resource in pending_rollback]
        if previous and rollback_resources:
            for resource in rollback_resources:
                bias = "reserve_" + resource + "_bias_gib"
                peak = "peak_extra_" + resource + "_gib"
                p[bias] = max(p[bias], previous[bias])
                if previous[peak] is not None:
                    p[peak] = max(p[peak] or 0, previous[peak])
            remaining = [resource for resource in pending_rollback if resource not in rollback_resources]
            previous["pending_resources"] = remaining
            profile["last_relaxation"] = previous if remaining else None
            reasons.append("rollback previous efficiency relaxation")
        # Each affected resource can increase once for this job. Late unknown-cause
        # feedback may add the other resources, never repeat an existing penalty.
        for resource in new_bias_resources:
            field = "reserve_" + resource + "_bias_gib"
            p[field] = min(BIAS_MAX[resource], p[field] + BIAS_STEP[resource])
        if new_bias_resources:
            job["bias_applied_resources"] = [resource for resource in RESOURCES
                                              if resource in already_biased or resource in new_bias_resources]
        if "quality" in new_adverse:
            p["quarantined"] = True
        p["disable_warm_cache"] = True
        profile["clean_streak"] = 0
        profile["last_adverse_run"] = p["runs"]
        job["adverse_applied"].extend(new_adverse)
        if new_adverse:
            reasons.append("adverse evidence: " + ", ".join(new_adverse))
        if new_bias_resources:
            reasons.append("reserve increase attributed to " + ", ".join(new_bias_resources))

    measured_jobs = [profile["jobs"][j] for j in profile["run_ids"] if j in profile["jobs"] and _measured(profile["jobs"][j])]
    p["measured_runs"] = len(measured_jobs)
    p["samples"] = sum(j["measurement"]["busy_samples"] for j in measured_jobs)
    if _measured(job) and not job["measurement_applied"]:
        job["measurement_applied"] = True
        for resource in RESOURCES:
            field = "peak_extra_" + resource + "_gib"
            p[field] = max(p[field] or 0.0, measurement[field])
        reasons.append("completed measured workload peaks")

    if _clean(job) and not job["clean_applied"] and job["record_number"] > profile["last_adverse_run"]:
        job["clean_applied"] = True
        profile["clean_streak"] += 1
        if profile["clean_streak"] >= 3:
            p["quarantined"] = False
            p["disable_warm_cache"] = False
        # Only exact-key runs with measured quiet load and explicit workflow QA.
        clean_jobs = [profile["jobs"][j] for j in profile["run_ids"] if j in profile["jobs"] and _clean(profile["jobs"][j])][-5:]
        durations = [j["measurement"]["elapsed_seconds"] for j in clean_jobs]
        comparable = len(durations) == 5 and max(durations) <= min(durations) * 1.25
        if profile["clean_streak"] >= 5 and comparable:
            prior = copy.deepcopy(p)
            for resource in RESOURCES:
                field = "reserve_" + resource + "_bias_gib"
                p[field] = max(0.0, p[field] - EASE_STEP[resource])
            # Relax learned extra margins only; configured reserve floors and peaks
            # are untouched. There is no quality-setting or model-setting mutation.
            if p != prior:
                profile["last_relaxation"] = prior
                reasons.append("five comparable clean runs: cautiously ease extra reserve")
            profile["clean_streak"] = 0
        else:
            reasons.append("explicit clean run evidence")
    return "; ".join(reasons) or "recorded workload feedback"


def record_run(state, key, job_id, measurement, feedback):
    """Learn once from broker-measured work; replay must be byte-equivalent."""
    _text(key, "profile key")
    _text(job_id, "job_id", 256)
    measured, reported = _measurement(measurement), _feedback(feedback)
    fingerprint = hashlib.sha256(_canonical([measured, reported]).encode()).hexdigest()
    existing = state.get("profiles", {}).get(key, {}).get("jobs", {}).get(job_id)
    if existing:
        if existing["fingerprint"] != fingerprint:
            raise ValueError("conflicting duplicate run; attach feedback with report_feedback")
        return {"changed": False, "duplicate": True, "policy": get_policy(state, key)}
    if _seen(state, key, job_id):
        raise ValueError("run was already learned or its bounded detail expired")
    profile = _profile(state, key)
    before = copy.deepcopy(profile["policy"])
    profile["jobs"][job_id] = {"measurement": measured, "feedback": reported,
                               "fingerprint": fingerprint, "feedback_ids": {},
                               "adverse_applied": [], "measurement_applied": False, "clean_applied": False,
                               "bias_applied_resources": [],
                               "record_number": profile["policy"]["runs"] + 1}
    profile["run_ids"].append(job_id)
    del profile["run_ids"][:-MAX_RUNS]
    profile["policy"]["runs"] += 1
    reason = _learn(profile, job_id)
    changed = _revision(profile, before, reason, job_id)
    _seen(state, key, job_id, add=True)
    while len(profile["jobs"]) > MAX_JOBS:
        del profile["jobs"][next(iter(profile["jobs"]))]
    return {"changed": changed, "duplicate": False, "revision": profile["policy"]["revision"],
            "reason": reason, "policy": get_policy(state, key)}


def report_feedback(state, key, job_id, feedback_id, feedback):
    """Attach late QA/user evidence; IDs are idempotent and failures stay adverse."""
    _text(key, "profile key")
    _text(job_id, "job_id", 256)
    _text(feedback_id, "feedback_id", 256)
    supplied = _feedback(feedback, partial=True)
    profile = state.get("profiles", {}).get(key)
    if not profile or job_id not in profile["jobs"]:
        raise ValueError("run detail unavailable for feedback")
    job = profile["jobs"][job_id]
    fingerprint = _canonical(supplied)
    if feedback_id in job["feedback_ids"]:
        if fingerprint != job["feedback_ids"][feedback_id]:
            raise ValueError("conflicting feedback_id")
        return {"changed": False, "duplicate": True, "policy": get_policy(state, key)}
    if len(job["feedback_ids"]) >= MAX_FEEDBACK_PER_JOB:
        raise ValueError("feedback capacity reached for this run")
    before = copy.deepcopy(profile["policy"])
    merged = copy.deepcopy(job["feedback"])
    for field, value in supplied.items():
        if field == "quality" and merged[field] == "failed":
            continue
        if field == "desktop_responsive" and merged[field] is False:
            continue
        if field == "outcome":
            rank = {"unknown": 0, "completed": 1, "cancelled": 2, "failed": 3, "oom": 4}
            if rank[merged[field]] >= rank[value]:
                continue
        if field == "quality" and value == "unknown":
            continue
        if field == "desktop_responsive" and value is None:
            continue
        merged[field] = value
    job["feedback"] = merged
    job["feedback_ids"][feedback_id] = fingerprint
    _touch(state, profile)
    reason = _learn(profile, job_id)
    changed = _revision(profile, before, reason, job_id)
    return {"changed": changed, "duplicate": False, "revision": profile["policy"]["revision"],
            "reason": reason, "policy": get_policy(state, key)}


def rollback(state, key, target_revision, reason):
    """Restore optimization policy conservatively, retaining observed safeguards."""
    _text(reason, "rollback reason", 256)
    if isinstance(target_revision, bool) or not isinstance(target_revision, int) or target_revision < 0:
        raise ValueError("invalid target_revision")
    profile = state.get("profiles", {}).get(key)
    if not profile:
        raise ValueError("profile unavailable")
    target = None
    for entry in profile["revisions"]:
        if entry["after"]["revision"] == target_revision:
            target = entry["after"]
        elif entry["before"]["revision"] == target_revision:
            target = entry["before"]
    if target is None:
        raise ValueError("revision outside retained audit history")
    before = copy.deepcopy(profile["policy"])
    p = profile["policy"]
    for resource in RESOURCES:
        bias = "reserve_" + resource + "_bias_gib"
        peak = "peak_extra_" + resource + "_gib"
        p[bias] = max(p[bias], target[bias])
        if target[peak] is not None:
            p[peak] = max(p[peak] or 0.0, target[peak])
    p["disable_warm_cache"] = p["disable_warm_cache"] or target["disable_warm_cache"]
    p["quarantined"] = p["quarantined"] or target["quarantined"]
    profile["clean_streak"] = 0
    profile["last_relaxation"] = None
    changed = _revision(profile, before, f"conservative rollback to {target_revision}: {reason}")
    return {"changed": changed, "target_revision": target_revision, "policy": get_policy(state, key)}
