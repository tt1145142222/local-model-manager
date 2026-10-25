"""Deterministic admission policy; never edits inference/quality parameters.

The policy protects desktop headroom through admission, scheduling hints and
cache lifetime. It cannot reclaim allocations from an executing CUDA kernel.
Measured workload records must come from completed *inference* windows, not
server startup. No dimensions-to-memory formula is presented as calibration.
"""
import copy
import hashlib
import json
import math


ESTIMATES = {
    "estimated_peak_ram_gib", "estimated_peak_commit_gib", "estimated_peak_vram_gib"
}
TASK_FIELDS = {
    "video": {"width", "height", "frames", "batch_size", "workflow_signature"},
    "llm": {"input_tokens", "max_output_tokens", "context_tokens", "images", "workflow_signature"},
}
REQUIRED_SHAPE = {
    "video": {"width", "height", "frames", "batch_size", "workflow_signature"},
    "llm": {"input_tokens", "max_output_tokens", "context_tokens", "workflow_signature"},
}
DEFAULT_KINDS = {"h3": "video", "muse": "llm"}
ADAPTER_KINDS = {"comfyui": "video", "llamacpp": "llm"}


def _number(value, *, minimum=0):
    try:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= minimum)
    except OverflowError:
        return False


def _budget_exceeds(required, available):
    # Equivalent floating-point expressions such as E + E*f and E*(1+f)
    # can differ below byte precision. This is numeric equality, not a buffer
    # reduction or admission based on the rounded user-facing numbers.
    return required > available and not math.isclose(required, available, rel_tol=1e-12, abs_tol=1e-12)


def resolve_kind(config, model):
    """Resolve workload semantics independently from registered model names.

Supporting a workload shape does not implement an inference backend. The broker
still needs a registered, validated lifecycle adapter for the configured model.
"""
    model_config = config.get("models", {}).get(model, {})
    kind = (model_config.get("workload_kind")
            or ADAPTER_KINDS.get(model_config.get("kind")) or DEFAULT_KINDS.get(model))
    if kind not in TASK_FIELDS:
        raise ValueError("Model requires a supported workload_kind: video or llm")
    return kind


def normalize_task(model, task, workload_kind=None):
    """Validate task metadata only; an empty/partial spec stays unprofiled.

Explicit estimated_peak_* values are conservative *additional free budgets*
above desktop reserves, not instructions to cap memory or change the workload.
"""
    kind = workload_kind or DEFAULT_KINDS.get(model)
    if kind not in TASK_FIELDS:
        raise ValueError("Model requires a supported workload_kind: video or llm")
    if task is None:
        task = {}
    if not isinstance(task, dict):
        raise ValueError("task must be an object")
    unknown = set(task) - TASK_FIELDS[kind] - ESTIMATES
    if unknown:
        raise ValueError("Unknown task fields: " + ", ".join(sorted(map(str, unknown))))
    result = {}
    for key, value in task.items():
        if key == "workflow_signature":
            if (not isinstance(value, str) or not value or len(value) > 128
                    or value.strip() != value or any(ord(char) < 32 for char in value)):
                raise ValueError("workflow_signature must be a nonblank string of 1..128 characters without control characters")
            result[key] = value
        elif key in ESTIMATES:
            if not _number(value):
                raise ValueError(key + " must be a finite nonnegative number")
            result[key] = float(value)
        else:
            minimum = 0 if key in {"input_tokens", "images"} else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(key + " must be an integer >= " + str(minimum))
            result[key] = value
    if kind == "llm" and "context_tokens" in result:
        if result.get("input_tokens", 0) + result.get("max_output_tokens", 0) > result["context_tokens"]:
            raise ValueError("input_tokens + max_output_tokens exceed context_tokens; quality limits are not silently reduced")
    return dict(sorted(result.items()))


def workload_key(model, normalized, config):
    """Exact shape/config identity; caller can add a verified artifact signature.

workload_fingerprint may include model/launcher file signatures. Changed files
at the same path are otherwise indistinguishable from unchanged configuration.
"""
    normalized = normalize_task(model, normalized, resolve_kind(config, model))
    fingerprint = {
        "policy_version": 1,
        "model": model,
        "task": normalized,
        "model_config": config.get("models", {}).get(model, {}),
        "adaptive_config": config.get("adaptive", {}),
        "artifact_signature": config.get("workload_fingerprint"),
    }
    encoded = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class AdaptivePolicy:
    """State advances only on observe(), once per distinct sample timestamp.

plan() is side-effect free. Repeated MCP status polls cannot accelerate recovery
or increase measured trends. Pressure rises immediately and falls one level
only after a full quiet recovery window of fresh samples.
"""
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.options = self.config.get("adaptive", {})
        if not isinstance(self.options, dict):
            raise ValueError("adaptive config must be an object")
        self.recovery_seconds = self._option("recovery_seconds", 30, 5)
        self.trend_horizon_seconds = self._option("trend_horizon_seconds", 10, 0)
        self.max_trend_headroom_gib = self._option("max_trend_headroom_gib", 8, 0)
        self.measured_margin_fraction = self._option("measured_margin_fraction", 0.15, 0)
        self.measured_margin_gib = self._option("measured_margin_gib", 1, 0)
        self._last = None
        self._last_models = {}
        self._quiet_since = None
        self._tier = 0
        self._held_trends = {"ram": 0.0, "commit": 0.0}
        self._state = {
            "pressure_level": "normal", "pressure_tier": 0,
            "non_model_cpu_percent": None, "scheduler_delay_ms": None,
            "ram_fall_gib_per_second": 0.0, "commit_fall_gib_per_second": 0.0,
            "raw_ram_fall_gib_per_second": 0.0, "raw_commit_fall_gib_per_second": 0.0,
            "observed_model_ram_growth_gib": 0.0, "observed_model_commit_growth_gib": 0.0,
            "trend_basis": "estimated non-model residual; absolute reserves use raw available resources",
            "trend_headroom_ram_gib": 0.0, "trend_headroom_commit_gib": 0.0,
            "signals": [], "sample_time": None,
        }

    def _option(self, name, default, minimum):
        value = self.options.get(name, default)
        if not _number(value, minimum=minimum):
            raise ValueError("adaptive." + name + " must be finite and >= " + str(minimum))
        return float(value)

    def _floors(self):
        values = {
            "ram_gib": self.config.get("min_available_ram_gib", 12),
            "commit_gib": self.config.get("min_commit_free_gib", 12),
            "vram_gib": self.config.get("min_free_vram_gib", 3),
        }
        if any(not _number(value) for value in values.values()):
            raise ValueError("resource floors must be finite nonnegative numbers")
        return {key: float(value) for key, value in values.items()}

    @staticmethod
    def _gpu(resources):
        gpus = resources.get("gpus", [])
        if not isinstance(gpus, list):
            return None
        return next((item for item in gpus if isinstance(item, dict) and item.get("index") == 0), None)

    def _model_memory_growth(self, model_status):
        """Estimate attributable growth using known registered process identities.

        RSS is only an estimate of physical pressure attribution (shared mappings
        and paging affect it). Private commit is used solely for commit growth;
        RSS never stands in for an unavailable private-commit measurement.
        """
        growth = {"ram": 0.0, "commit": 0.0}
        for model in self.config.get("models", {}):
            current = model_status.get(model, {})
            previous = self._last_models.get(model, {})
            if not isinstance(current, dict) or not current.get("identity"):
                continue
            identity = current["identity"]
            same_identity = previous.get("identity") == identity
            new_identity = (previous.get("state") == "stopped"
                            or (previous.get("identity") and not same_identity))
            for resource, field in (("ram", "rss_gib"), ("commit", "private_commit_gib")):
                value = current.get(field)
                if not _number(value):
                    continue
                if same_identity and _number(previous.get(field)):
                    baseline = previous[field]
                elif new_identity:
                    # The previous process is gone or was verified stopped.
                    # This identity's known allocation started from zero.
                    baseline = 0.0
                else:
                    # An unobserved/missing metric is not a zero allocation.
                    continue
                growth[resource] += max(0.0, value - baseline)
        return growth

    def observe(self, resources, model_status):
        timestamp = resources.get("time")
        # Synthetic callers with no timestamp may use policy thresholds, but do
        # not accumulate an apparent recovery period from repeated reads.
        if not _number(timestamp) or (self._last is not None and timestamp <= self._last["time"]):
            return copy.deepcopy(self._state)
        floors = self._floors()
        signals, desired = [], 0
        host_cpu = resources.get("cpu_percent")
        model_cpu = sum(item.get("cpu_percent", 0) for item in model_status.values()
                        if isinstance(item, dict) and _number(item.get("cpu_percent")))
        outside_cpu = None
        if _number(host_cpu) and host_cpu <= 100:
            outside_cpu = max(0.0, host_cpu - model_cpu)
            if outside_cpu >= 65:
                desired = 2
                signals.append("high non-model CPU load")
            elif outside_cpu >= 30:
                desired = 1
                signals.append("elevated non-model CPU load")
        delay = resources.get("scheduler_delay_ms")
        if not _number(delay) and _number(resources.get("scheduler_delay_seconds")):
            delay = resources["scheduler_delay_seconds"] * 1000
        if _number(delay):
            if delay >= 1000:
                desired = 2
                signals.append("scheduler delay >= 1000 ms")
            elif delay >= 250:
                desired = max(desired, 1)
                signals.append("scheduler delay >= 250 ms")
        else:
            delay = None
        trends = {"ram": 0.0, "commit": 0.0}
        raw_trends = {"ram": 0.0, "commit": 0.0}
        model_growth = {"ram": 0.0, "commit": 0.0}
        elapsed = timestamp - self._last["time"] if self._last else None
        # A long monitoring gap is not a useful instantaneous trend.
        if elapsed is not None and 0.1 <= elapsed <= max(30, self.config.get("sample_seconds", 5) * 4):
            model_growth = self._model_memory_growth(model_status)
            for resource, key in (("ram", "ram_available_gib"), ("commit", "commit_free_gib")):
                previous, current = self._last.get(key), resources.get(key)
                if _number(previous) and _number(current):
                    raw_trends[resource] = max(0.0, (previous - current) / elapsed)
                    # Avoid labeling expected model loading as desktop pressure.
                    # Unattributed decline remains protective; absolute floors
                    # below are checked against raw available memory regardless.
                    trends[resource] = max(0.0, previous - current - model_growth[resource]) / elapsed
                    if trends[resource] >= 0.5:
                        desired = 2
                        signals.append("rapidly falling estimated non-model " + resource + " headroom")
                    elif trends[resource] >= 0.15:
                        desired = max(desired, 1)
                        signals.append("falling estimated non-model " + resource + " headroom")
        for key, resource in (("ram_available_gib", "ram"), ("commit_free_gib", "commit")):
            current = resources.get(key)
            if _number(current) and current < floors[resource + "_gib"] + 2:
                desired = 2
                signals.append(resource + " near desktop reserve")
        current_trends = {key: min(self.max_trend_headroom_gib, rate * self.trend_horizon_seconds)
                          for key, rate in trends.items()}
        if desired > self._tier:
            self._tier, self._quiet_since = desired, None
        elif desired == self._tier:
            self._quiet_since = None
        else:
            if self._quiet_since is None:
                self._quiet_since = timestamp
            if timestamp - self._quiet_since >= self.recovery_seconds:
                self._tier -= 1
                self._quiet_since = timestamp if self._tier > desired else None
                self._held_trends = current_trends.copy()
        if self._tier == 0:
            self._held_trends = current_trends.copy()
        else:
            self._held_trends = {key: max(self._held_trends[key], value) for key, value in current_trends.items()}
        self._last = {"time": timestamp, "ram_available_gib": resources.get("ram_available_gib"),
                      "commit_free_gib": resources.get("commit_free_gib")}
        self._last_models = {
            model: {key: copy.deepcopy(status.get(key)) for key in ("state", "identity", "rss_gib", "private_commit_gib")}
            for model, status in model_status.items()
            if model in self.config.get("models", {}) and isinstance(status, dict)
        }
        self._state = {
            "pressure_level": ("normal", "protect", "high")[self._tier],
            "pressure_tier": self._tier, "non_model_cpu_percent": outside_cpu,
            "scheduler_delay_ms": delay,
            "ram_fall_gib_per_second": trends["ram"],
            "commit_fall_gib_per_second": trends["commit"],
            "raw_ram_fall_gib_per_second": raw_trends["ram"],
            "raw_commit_fall_gib_per_second": raw_trends["commit"],
            "observed_model_ram_growth_gib": model_growth["ram"],
            "observed_model_commit_growth_gib": model_growth["commit"],
            "trend_basis": "estimated non-model residual; absolute reserves use raw available resources",
            "trend_headroom_ram_gib": self._held_trends["ram"],
            "trend_headroom_commit_gib": self._held_trends["commit"],
            "signals": signals, "sample_time": timestamp,
        }
        return copy.deepcopy(self._state)

    def plan(self, model, task, resources, model_status, learned=None, cold=False, fallback=None, startup_admission=False):
        kind = resolve_kind(self.config, model)
        normalized = normalize_task(model, task, kind)
        key = workload_key(model, normalized, self.config)
        # Dimensions alone do not identify conditioning, checkpoints, sampling
        # steps or inference mode. A producer-supplied workflow/quality signature
        # is required before any cross-run calibration is used.
        profile_complete = REQUIRED_SHAPE[kind].issubset(normalized)
        tier = self._tier
        floors = self._floors()
        reserves = {
            "ram_gib": floors["ram_gib"] + (0, 2, 4)[tier] + self._held_trends["ram"],
            "commit_gib": floors["commit_gib"] + (0, 2, 4)[tier] + self._held_trends["commit"],
            "vram_gib": floors["vram_gib"] + (0, 0.5, 1)[tier],
        }
        # Failure/desktop-pressure feedback can raise bounded reserves before a
        # successful calibration exists. The learner owns evidence persistence;
        # this boundary still verifies exact workload identity and finite bounds.
        matching_learning = (profile_complete and isinstance(learned, dict)
                             and learned.get("workload_key") == key)
        applied_bias = {}
        if matching_learning:
            for resource, maximum in (("ram", 8), ("commit", 8), ("vram", 3)):
                bias = learned.get("reserve_" + resource + "_bias_gib", 0)
                if _number(bias) and bias <= maximum:
                    reserves[resource + "_gib"] += bias
                    applied_bias[resource + "_gib"] = float(bias)
        # Persistent admission requirements exclude temporary load protection.
        # They distinguish a policy that cannot fit this device from headroom
        # which can recover when background load falls.
        minimum_reserves = {resource: floor + applied_bias.get(resource, 0.0)
                            for resource, floor in floors.items()}
        required = reserves.copy()
        minimum_required = minimum_reserves.copy()
        breakdown = {
            resource: {
                "base_floor_gib": floor,
                "learned_bias_gib": applied_bias.get(resource, 0.0),
                "persistent_reserve_gib": minimum_reserves[resource],
                "dynamic_reserve_extra_gib": reserves[resource] - minimum_reserves[resource],
                "current_reserve_gib": reserves[resource],
                "measured_peak_extra_gib": None,
                "uncertainty_margin_gib": None,
                "shared_headroom_gib": None,
                "minimum_shared_headroom_gib": None,
                "cold_startup_required_gib": None,
                "minimum_cold_startup_required_gib": None,
                "explicit_peak_extra_gib": None,
            } for resource, floor in floors.items()
        }
        baseline = self.config.get("models", {}).get(model, {})
        if cold:
            for resource, name in (("ram", "startup_free_ram_gib"), ("commit", "startup_commit_free_gib"),
                                   ("vram", "startup_free_vram_gib")):
                minimum = baseline.get(name)
                if not _number(minimum):
                    raise ValueError("missing or invalid cold startup threshold: " + name)
                # Add only adaptive desktop increments to the configured startup
                # threshold, which already includes the fixed desktop reserve.
                required[resource + "_gib"] = max(required[resource + "_gib"],
                    minimum + reserves[resource + "_gib"] - floors[resource + "_gib"])
                minimum_required[resource + "_gib"] = max(minimum_required[resource + "_gib"],
                    minimum + minimum_reserves[resource + "_gib"] - floors[resource + "_gib"])
                breakdown[resource + "_gib"].update(
                    cold_startup_required_gib=required[resource + "_gib"],
                    minimum_cold_startup_required_gib=minimum_required[resource + "_gib"])
        confidence = "configured_fallback"
        notes = ["quality parameters unchanged", "partial or unknown shape uses configured reserve floors"] if not profile_complete else [
            "quality parameters unchanged", "exact shape has no accepted completed-inference history"]
        learned_valid = (profile_complete and isinstance(learned, dict) and learned.get("workload_key") == key
                         and learned.get("completed", True) is True
                         and isinstance(learned.get("runs"), int) and not isinstance(learned.get("runs"), bool) and learned["runs"] >= 1
                         and ("measured_runs" not in learned or (isinstance(learned["measured_runs"], int)
                              and not isinstance(learned["measured_runs"], bool) and learned["measured_runs"] >= 1))
                         and isinstance(learned.get("samples"), int) and not isinstance(learned.get("samples"), bool) and learned["samples"] >= 3
                         and all(_number(learned.get("peak_extra_" + resource + "_gib")) for resource in ("ram", "commit", "vram")))
        if learned_valid and not cold:
            for resource in ("ram", "commit", "vram"):
                extra = learned["peak_extra_" + resource + "_gib"]
                margin = max(extra * self.measured_margin_fraction, self.measured_margin_gib)
                shared_headroom = max(reserves[resource + "_gib"], margin)
                minimum_shared_headroom = max(minimum_reserves[resource + "_gib"], margin)
                # Uncertainty and desktop protection size one shared headroom
                # budget; they are not three additive allocations. At the
                # observed high-water mark this retains the full desktop reserve.
                # Future unobserved peaks remain uncertain and are monitored.
                required[resource + "_gib"] = max(required[resource + "_gib"], extra + shared_headroom)
                minimum_required[resource + "_gib"] = max(minimum_required[resource + "_gib"],
                                                          extra + minimum_shared_headroom)
                breakdown[resource + "_gib"].update(
                    measured_peak_extra_gib=extra, uncertainty_margin_gib=margin,
                    shared_headroom_gib=shared_headroom,
                    minimum_shared_headroom_gib=minimum_shared_headroom)
            confidence = "measured_exact"
            notes = ["quality parameters unchanged",
                     "completed exact-workload high water plus max(desktop reserve, absolute margin, fractional margin); no cross-shape scaling",
                     "shared headroom retains desktop reserve at observed peaks; future unobserved peaks remain uncertain"]
        elif learned is not None:
            notes.append("history not used for this admission: startup, incomplete shape or mismatched/invalid record")
        # Uncalibrated work must not be admitted on desktop floors alone. These
        # are configured extra budgets, never a shape-derived capacity claim.
        # They apply at admission only; running allocations are not charged twice.
        if fallback is not None and not learned_valid:
            if not isinstance(fallback, dict) or set(fallback)-{'ram','commit','vram'}:
                raise ValueError('invalid unprofiled peak budgets')
            for resource, extra in fallback.items():
                if not _number(extra):
                    raise ValueError('unprofiled peak budgets must be finite and nonnegative')
                if 'estimated_peak_'+resource+'_gib' in normalized:
                    continue  # A reliable task-specific estimate supersedes fallback.
                name = resource+'_gib'
                required[name] = max(required[name], reserves[name]+extra)
                minimum_required[name] = max(minimum_required[name], minimum_reserves[name]+extra)
                breakdown[name]['configured_unprofiled_extra_gib'] = extra
            confidence = 'configured_unprofiled_budget'
            notes.append('uncalibrated inference uses configured extra free budgets; no inferred quality changes')
        # An explicit operator-selected startup gate replaces configured VRAM
        # prediction only. Preserve resource floors, measured warm admission,
        # and task-specific estimates applied below. Never alter model quality.
        gate = baseline.get("startup_vram_admission_gib")
        if gate is not None:
            if not _number(gate) or gate < floors["vram_gib"]:
                raise ValueError("startup VRAM admission must be finite and at least the VRAM floor")
            if cold or (startup_admission and not learned_valid):
                breakdown["vram_gib"]["automatic_required_before_override_gib"] = required["vram_gib"]
                required["vram_gib"] = max(float(gate), reserves["vram_gib"])
                minimum_required["vram_gib"] = max(float(gate), minimum_reserves["vram_gib"])
                breakdown["vram_gib"]["operator_startup_gate_gib"] = float(gate)
                if cold:
                    breakdown["vram_gib"]["cold_startup_required_gib"] = required["vram_gib"]
                    breakdown["vram_gib"]["minimum_cold_startup_required_gib"] = minimum_required["vram_gib"]
                notes.append("explicit operator startup VRAM gate replaces automatic startup budget")
        for resource in ("ram", "commit", "vram"):
            estimate = normalized.get("estimated_peak_" + resource + "_gib")
            if estimate is not None:
                required[resource + "_gib"] = max(required[resource + "_gib"], reserves[resource + "_gib"] + estimate)
                minimum_required[resource + "_gib"] = max(minimum_required[resource + "_gib"],
                                                          minimum_reserves[resource + "_gib"] + estimate)
                breakdown[resource + "_gib"]["explicit_peak_extra_gib"] = estimate
                if confidence != "measured_exact":
                    confidence = "explicit_estimate"
        if confidence == "explicit_estimate":
            notes.append("explicit peak estimate treated as extra free budget above desktop reserve")
        if any(applied_bias.values()):
            notes.append("bounded learned reserve increases applied for this exact task/config")
        gpu = self._gpu(resources)
        available = {"ram_gib": resources.get("ram_available_gib"), "commit_gib": resources.get("commit_free_gib"),
                     "vram_gib": gpu.get("free_gib") if gpu else None}
        # Status must remain strict JSON even when sensors return NaN/Infinity.
        available = {resource: value if _number(value) else None for resource, value in available.items()}
        capacity = {"ram_gib": resources.get("ram_total_gib"),
                    "commit_gib": resources.get("commit_limit_gib"),
                    "vram_gib": gpu.get("total_gib") if gpu else None}
        # Total capacity is an upper bound, not a claim that all occupied memory
        # can be reclaimed. Missing/invalid optional capacities stay unknown.
        capacity = {resource: value if _number(value) else None for resource, value in capacity.items()}
        reasons = []
        capacity_conflicts, temporary_capacity_pressure = [], []
        for resource, minimum in required.items():
            current = available[resource]
            ceiling = capacity[resource]
            if ceiling is not None and _budget_exceeds(minimum_required[resource], ceiling):
                capacity_conflicts.append({"resource": resource,
                    "minimum_required_gib": minimum_required[resource], "capacity_gib": ceiling})
                reasons.append(f"{resource} persistent minimum required {minimum_required[resource]:.4f} > physical capacity {ceiling:.4f} GiB; policy infeasible, waiting alone cannot resolve it")
            elif ceiling is not None and _budget_exceeds(minimum, ceiling):
                temporary_capacity_pressure.append({"resource": resource,
                    "current_required_gib": minimum, "minimum_required_gib": minimum_required[resource],
                    "capacity_gib": ceiling})
                reasons.append(f"{resource} current required {minimum:.4f} > physical capacity {ceiling:.4f} GiB due to temporary load reserve; waiting for load recovery")
            if not _number(current):
                reasons.append(resource + " telemetry unavailable or invalid")
            elif _budget_exceeds(minimum, current):
                reasons.append(f"{resource} available {current:.2f} < required {minimum:.2f} GiB")
            breakdown[resource].update(required_gib=minimum,
                                       minimum_required_gib=minimum_required[resource],
                                       capacity_gib=ceiling)
        admission_state = "policy_infeasible" if capacity_conflicts else ("waiting" if reasons else "ready")
        # CPU telemetry is optional. Preserve at least two logical CPUs on the
        # measured host when possible; adapt scheduling without changing tokens,
        # sampling steps, resolution, quantization or model layers.
        cpus = resources.get("cpu_count", resources.get("logical_cpu_count", 1))
        if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus < 1:
            cpus = 1
        reserve_cpus = min(cpus - 1, (2, 4, 8)[tier])
        warm = self.config.get("idle_grace_seconds", 120)
        if not _number(warm):
            raise ValueError("idle_grace_seconds must be finite and nonnegative")
        warm = (warm, min(warm, 30), 0)[tier]
        if matching_learning and learned.get("disable_warm_cache") is True:
            warm = 0
            notes.append("warm cache disabled by matching workload feedback")
        return {
            "proceed": not reasons, "reasons": reasons,
            "admission_state": admission_state, "retryable": not bool(capacity_conflicts),
            "reserves": {key: round(value, 4) for key, value in reserves.items()},
            "required": {key: round(value, 4) for key, value in required.items()},
            "minimum_required": {key: round(value, 4) for key, value in minimum_required.items()},
            "capacity": capacity, "capacity_conflicts": capacity_conflicts,
            "temporary_capacity_pressure": temporary_capacity_pressure,
            "budget_breakdown": breakdown,
            "available": available,
            "runtime_profile": {"cpu_priority": "below_normal" if tier else "normal",
                                "reserve_logical_cpus": reserve_cpus},
            "warm_seconds": warm, "confidence": confidence,
            "profile_complete": profile_complete, "quality_preserved": True,
            "learned_reserve_bias": applied_bias,
            "task": normalized, "workload_key": key, "cold": bool(cold),
            "workload_kind": kind,
            "load": copy.deepcopy(self._state), "notes": notes,
        }
