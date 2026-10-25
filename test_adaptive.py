import copy
import json
import math
import unittest

from adaptive import AdaptivePolicy, normalize_task, resolve_kind, workload_key


CONFIG = {
    "sample_seconds": 5, "idle_grace_seconds": 120,
    "min_available_ram_gib": 12, "min_commit_free_gib": 12, "min_free_vram_gib": 3,
    "models": {
        "h3": {"startup_free_ram_gib": 20, "startup_commit_free_gib": 36, "startup_free_vram_gib": 24},
        "muse": {"startup_free_ram_gib": 16, "startup_commit_free_gib": 26, "startup_free_vram_gib": 22},
    },
}
H3 = {"width": 1024, "height": 576, "frames": 121, "batch_size": 1, "workflow_signature": "verified-workflow-a"}
MUSE = {"input_tokens": 1200, "max_output_tokens": 600, "context_tokens": 8192, "workflow_signature": "verified-chat-a"}


def telemetry(t=100, **updates):
    sample = {"time": t, "ram_available_gib": 40, "commit_free_gib": 50,
              "gpus": [{"index": 0, "free_gib": 28}], "cpu_percent": 10,
              "cpu_count": 32, "scheduler_delay_ms": 0}
    sample.update(updates)
    return sample


class TaskTests(unittest.TestCase):
    def test_quality_fields_untouched_and_input_not_mutated(self):
        source = dict(H3)
        self.assertEqual(normalize_task("h3", source), source)
        self.assertEqual(source, H3)

    def test_unknown_fields_rejected(self):
        for model, task in (("h3", {"steps": 12}), ("muse", {"temperature": 0.1})):
            with self.assertRaises(ValueError):
                normalize_task(model, task)

    def test_integer_and_estimate_validation(self):
        for bad in (True, -1, 0, 1.5, "1024", math.inf):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalize_task("h3", {"width": bad})
        for bad in (-1, math.inf, math.nan, True, "8"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalize_task("h3", {"estimated_peak_ram_gib": bad})

    def test_context_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            normalize_task("muse", {"input_tokens": 8000, "max_output_tokens": 1000, "context_tokens": 8192})

    def test_images_and_zero_input_valid(self):
        self.assertEqual(normalize_task("muse", {"input_tokens": 0, "images": 0}), {"images": 0, "input_tokens": 0})

    def test_workflow_signature_validation(self):
        for bad in (None, True, 1, "", " ", " leading", "trailing ", "newline\ninside", "x" * 129):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalize_task("h3", {"workflow_signature": bad})
        self.assertEqual(normalize_task("muse", {"workflow_signature": "x" * 128}), {"workflow_signature": "x" * 128})

    def test_exact_task_and_config_fingerprint(self):
        original = workload_key("h3", H3, CONFIG)
        self.assertEqual(original, workload_key("h3", dict(reversed(list(H3.items()))), CONFIG))
        self.assertNotEqual(original, workload_key("h3", {**H3, "frames": 122}, CONFIG))
        self.assertNotEqual(original, workload_key("h3", {**H3, "workflow_signature": "ref2va-different-checkpoint"}, CONFIG))
        changed = copy.deepcopy(CONFIG)
        changed["workload_fingerprint"] = "new-weight-sha256"
        self.assertNotEqual(original, workload_key("h3", H3, changed))

    def test_arbitrary_model_ids_use_registered_workload_kinds(self):
        config = copy.deepcopy(CONFIG)
        config["models"]["qwen-local"] = {**CONFIG["models"]["muse"], "kind": "llamacpp"}
        config["models"]["wan-video"] = {**CONFIG["models"]["h3"], "workload_kind": "video", "kind": "comfyui"}
        self.assertEqual(resolve_kind(config, "qwen-local"), "llm")
        self.assertEqual(resolve_kind(config, "wan-video"), "video")
        self.assertEqual(normalize_task("qwen-local", MUSE, "llm"), MUSE)
        self.assertEqual(normalize_task("wan-video", H3, "video"), H3)
        policy = AdaptivePolicy(config)
        for model, task, kind in (("qwen-local", MUSE, "llm"), ("wan-video", H3, "video")):
            plan = policy.plan(model, task, telemetry(), {}, cold=True)
            self.assertEqual(plan["workload_kind"], kind)
            self.assertTrue(plan["profile_complete"])
            self.assertTrue(plan["proceed"])
            self.assertEqual(plan["task"], task)
        self.assertNotEqual(workload_key("wan-video", H3, config), workload_key("h3", H3, config))

    def test_unregistered_workload_semantics_rejected(self):
        with self.assertRaises(ValueError):
            normalize_task("new-backend", {})
        with self.assertRaises(ValueError):
            resolve_kind({"models": {"new-backend": {"kind": "unknown-framework"}}}, "new-backend")
        with self.assertRaises(ValueError):
            normalize_task("qwen-local", H3, "llm")


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.p = AdaptivePolicy(CONFIG)

    def plan(self, resources=None, task=H3, learned=None, cold=False):
        return self.p.plan("h3", task, resources or telemetry(), {}, learned=learned, cold=cold)

    def record(self, **updates):
        result = {"workload_key": workload_key("h3", H3, CONFIG), "runs": 1, "samples": 5,
                  "peak_extra_ram_gib": 5, "peak_extra_commit_gib": 7, "peak_extra_vram_gib": 4}
        result.update(updates)
        return result

    def test_baselines_unchanged_without_pressure(self):
        self.p.observe(telemetry(), {})
        plan = self.plan(cold=True)
        self.assertEqual(plan["required"], {"ram_gib": 20, "commit_gib": 36, "vram_gib": 24})
        self.assertTrue(plan["quality_preserved"])
        self.assertEqual(plan["task"], H3)

    def test_high_external_load_protects_and_shortens_cache(self):
        self.p.observe(telemetry(cpu_percent=85), {"h3": {"cpu_percent": 10}})
        plan = self.plan()
        self.assertEqual(plan["runtime_profile"], {"cpu_priority": "below_normal", "reserve_logical_cpus": 8})
        self.assertEqual(plan["warm_seconds"], 0)
        self.assertGreater(plan["reserves"]["ram_gib"], 12)

    def test_model_cpu_is_subtracted(self):
        self.p.observe(telemetry(cpu_percent=85), {"h3": {"cpu_percent": 80}})
        self.assertEqual(self.plan()["load"]["pressure_level"], "normal")

    def test_scheduler_delay_protects_without_cpu_telemetry(self):
        self.p.observe(telemetry(cpu_percent=None, scheduler_delay_ms=1500), {})
        self.assertEqual(self.plan()["load"]["pressure_level"], "high")

    def test_falling_headroom_adds_trend_budget(self):
        self.p.observe(telemetry(), {})
        self.p.observe(telemetry(105, ram_available_gib=36, commit_free_gib=45), {})
        plan = self.plan()
        self.assertEqual(plan["load"]["pressure_tier"], 2)
        self.assertEqual(plan["reserves"]["ram_gib"], 24)
        self.assertEqual(plan["reserves"]["commit_gib"], 24)

    def test_verified_model_cold_load_is_not_desktop_pressure(self):
        self.p.observe(telemetry(ram_available_gib=43, commit_free_gib=55), {"muse": {"state": "stopped", "identity": None}})
        state = self.p.observe(telemetry(105, ram_available_gib=27, commit_free_gib=35),
            {"muse": {"state": "ready", "identity": {"pid": 12, "created": 10}, "rss_gib": 16, "private_commit_gib": 20}})
        self.assertEqual(state["pressure_tier"], 0)
        self.assertEqual(state["observed_model_ram_growth_gib"], 16)
        self.assertEqual(state["raw_ram_fall_gib_per_second"], 3.2)
        self.assertEqual(state["ram_fall_gib_per_second"], 0)
        self.assertEqual(self.plan()["reserves"]["ram_gib"], 12)

    def test_same_identity_growth_subtracted_but_other_app_drop_protects(self):
        original = {"h3": {"state": "ready", "identity": {"pid": 10}, "rss_gib": 2, "private_commit_gib": 3}}
        loaded = {"h3": {"state": "ready", "identity": {"pid": 10}, "rss_gib": 18, "private_commit_gib": 23}}
        self.p.observe(telemetry(ram_available_gib=43, commit_free_gib=55), original)
        state = self.p.observe(telemetry(105, ram_available_gib=22, commit_free_gib=35), loaded)
        self.assertEqual(state["observed_model_ram_growth_gib"], 16)
        self.assertEqual(state["ram_fall_gib_per_second"], 1)
        self.assertEqual(state["commit_fall_gib_per_second"], 0)
        self.assertEqual(state["pressure_tier"], 2)

    def test_model_growth_cannot_hide_absolute_low_memory(self):
        self.p.observe(telemetry(), {"h3": {"state": "stopped"}})
        state = self.p.observe(telemetry(105, ram_available_gib=10),
            {"h3": {"state": "ready", "identity": {"pid": 10}, "rss_gib": 30}})
        self.assertEqual(state["ram_fall_gib_per_second"], 0)
        self.assertEqual(state["pressure_tier"], 2)
        self.assertFalse(self.plan(telemetry(ram_available_gib=10))["proceed"])

    def test_unknown_model_memory_is_not_attributed_to_zero(self):
        self.p.observe(telemetry(), {"h3": {"state": "ready", "identity": {"pid": 10}, "rss_gib": None}})
        state = self.p.observe(telemetry(105, ram_available_gib=30, commit_free_gib=40),
            {"h3": {"state": "ready", "identity": {"pid": 10}, "rss_gib": 10, "private_commit_gib": 10}})
        self.assertEqual(state["observed_model_ram_growth_gib"], 0)
        self.assertEqual(state["observed_model_commit_growth_gib"], 0)
        self.assertEqual(state["pressure_tier"], 2)

    def test_replaced_identity_starts_its_allocation_from_zero(self):
        self.p.observe(telemetry(), {"h3": {"state": "ready", "identity": {"pid": 10}, "rss_gib": 3}})
        state = self.p.observe(telemetry(105, ram_available_gib=30),
            {"h3": {"state": "ready", "identity": {"pid": 11}, "rss_gib": 10}})
        self.assertEqual(state["observed_model_ram_growth_gib"], 10)
        self.assertEqual(state["pressure_tier"], 0)

    def test_unregistered_process_metrics_cannot_suppress_pressure(self):
        self.p.observe(telemetry(), {"not-registered": {"state": "stopped"}})
        state = self.p.observe(telemetry(105, ram_available_gib=30),
            {"not-registered": {"state": "ready", "identity": {"pid": 12}, "rss_gib": 10}})
        self.assertEqual(state["observed_model_ram_growth_gib"], 0)
        self.assertEqual(state["pressure_tier"], 2)

    def test_recovery_is_slow_and_repeated_polls_do_not_advance(self):
        self.p.observe(telemetry(cpu_percent=90), {})
        self.p.observe(telemetry(105), {})
        for _ in range(20):
            self.p.observe(telemetry(105), {})
            self.plan()
        self.assertEqual(self.plan()["load"]["pressure_tier"], 2)
        self.p.observe(telemetry(134), {})
        self.assertEqual(self.plan()["load"]["pressure_tier"], 2)
        self.p.observe(telemetry(135), {})
        self.assertEqual(self.plan()["load"]["pressure_tier"], 1)
        self.p.observe(telemetry(165), {})
        self.assertEqual(self.plan()["load"]["pressure_tier"], 0)

    def test_backward_or_missing_timestamp_cannot_recover(self):
        self.p.observe(telemetry(cpu_percent=90), {})
        self.p.observe(telemetry(99), {})
        self.p.observe(telemetry(None), {})
        self.assertEqual(self.plan()["load"]["pressure_tier"], 2)

    def test_required_sensor_missing_or_nan_fails_closed(self):
        for updates in ({"ram_available_gib": None}, {"ram_available_gib": math.nan},
                        {"commit_free_gib": math.inf}, {"gpus": []},
                        {"gpus": [{"index": 0, "free_gib": -1}]}):
            plan = self.plan(telemetry(**updates))
            self.assertFalse(plan["proceed"])
            json.dumps(plan, allow_nan=False)

    def test_optional_telemetry_missing_does_not_block(self):
        sample = telemetry(cpu_percent=None, cpu_count=None, scheduler_delay_ms=None)
        self.p.observe(sample, {})
        plan = self.plan(sample)
        self.assertTrue(plan["proceed"])
        self.assertEqual(plan["runtime_profile"]["reserve_logical_cpus"], 0)

    def test_exact_inference_peaks_with_margin(self):
        plan = self.plan(learned=self.record())
        self.assertEqual(plan["confidence"], "measured_exact")
        self.assertEqual(plan["required"]["ram_gib"], 17)
        self.assertEqual(plan["required"]["commit_gib"], 19)
        self.assertEqual(plan["required"]["vram_gib"], 7)
        self.assertEqual(plan["budget_breakdown"]["vram_gib"]["uncertainty_margin_gib"], 1)
        self.assertEqual(plan["budget_breakdown"]["vram_gib"]["shared_headroom_gib"], 3)

    def test_completed_h3_user_peak_fits_without_clipping_or_discarding_learning(self):
        peak = 24.443359375
        record = self.record(samples=177, peak_extra_vram_gib=peak, reserve_vram_bias_gib=0.5)
        sample = telemetry(gpus=[{"index": 0, "free_gib": 28.91796875, "total_gib": 31.84}])
        plan = self.plan(sample, learned=record)
        self.assertTrue(plan["proceed"])
        self.assertEqual(plan["admission_state"], "ready")
        self.assertEqual(plan["confidence"], "measured_exact")
        self.assertEqual(plan["required"]["vram_gib"], 28.1099)
        self.assertEqual(plan["minimum_required"]["vram_gib"], 28.1099)
        self.assertEqual(plan["reserves"]["vram_gib"], 3.5)
        budget = plan["budget_breakdown"]["vram_gib"]
        self.assertEqual(budget["measured_peak_extra_gib"], peak)
        self.assertAlmostEqual(budget["uncertainty_margin_gib"], peak * 0.15)
        self.assertEqual(budget["shared_headroom_gib"], max(3.5, peak * 0.15))
        self.assertGreaterEqual(budget["required_gib"] - peak, 3.5)
        self.assertEqual(plan["workload_key"], record["workload_key"])
        self.assertEqual(plan["task"], H3)

    def test_real_low_available_memory_still_waits_despite_known_success(self):
        record = self.record(peak_extra_vram_gib=24.443359375, reserve_vram_bias_gib=0.5)
        plan = self.plan(telemetry(gpus=[{"index": 0, "free_gib": 27.5, "total_gib": 31.84}]), learned=record)
        self.assertFalse(plan["proceed"])
        self.assertTrue(plan["retryable"])
        self.assertEqual(plan["admission_state"], "waiting")
        self.assertEqual(plan["capacity_conflicts"], [])
        self.assertEqual(plan["required"]["vram_gib"], 28.1099)

    def test_persistent_measured_budget_over_capacity_is_not_infinite_wait(self):
        plan = self.plan(telemetry(gpus=[{"index": 0, "free_gib": 31.84, "total_gib": 31.84}]),
                         learned=self.record(peak_extra_vram_gib=30))
        self.assertFalse(plan["proceed"])
        self.assertFalse(plan["retryable"])
        self.assertEqual(plan["admission_state"], "policy_infeasible")
        self.assertEqual(plan["required"]["vram_gib"], 34.5)
        self.assertEqual(plan["minimum_required"]["vram_gib"], 34.5)
        self.assertEqual(plan["capacity_conflicts"][0]["resource"], "vram_gib")
        self.assertTrue(any("waiting alone cannot resolve" in reason for reason in plan["reasons"]))

    def test_unknown_or_invalid_capacity_never_invents_infeasibility(self):
        for capacity in (None, math.nan, math.inf, -1):
            sample = telemetry(gpus=[{"index": 0, "free_gib": 28, "total_gib": capacity}],
                               ram_total_gib=capacity, commit_limit_gib=capacity)
            plan = self.plan(sample, learned=self.record(peak_extra_vram_gib=30))
            self.assertFalse(plan["proceed"])
            self.assertTrue(plan["retryable"])
            self.assertEqual(plan["admission_state"], "waiting")
            self.assertEqual(plan["capacity"], {"ram_gib": None, "commit_gib": None, "vram_gib": None})
            json.dumps(plan, allow_nan=False)

    def test_capacity_and_available_boundaries_use_full_precision(self):
        required = 24.443359375 * 1.15
        record = self.record(peak_extra_vram_gib=24.443359375, reserve_vram_bias_gib=0.5)
        equal = self.plan(telemetry(gpus=[{"index": 0, "free_gib": required, "total_gib": required}]), learned=record)
        self.assertEqual(equal["admission_state"], "ready")
        lower_free = self.plan(telemetry(gpus=[{"index": 0, "free_gib": required - 0.000001, "total_gib": required}]), learned=record)
        self.assertEqual(lower_free["admission_state"], "waiting")
        lower_capacity = self.plan(telemetry(gpus=[{"index": 0, "free_gib": required - 0.000001,
                                                  "total_gib": required - 0.000001}]), learned=record)
        self.assertEqual(lower_capacity["admission_state"], "policy_infeasible")

    def test_temporary_pressure_over_capacity_waits_for_load_recovery(self):
        self.p.observe(telemetry(cpu_percent=90), {})
        task = {**H3, "estimated_peak_vram_gib": 28}
        plan = self.plan(telemetry(gpus=[{"index": 0, "free_gib": 31.84, "total_gib": 31.84}]), task=task)
        self.assertFalse(plan["proceed"])
        self.assertTrue(plan["retryable"])
        self.assertEqual(plan["admission_state"], "waiting")
        self.assertEqual(plan["minimum_required"]["vram_gib"], 31)
        self.assertEqual(plan["required"]["vram_gib"], 32)
        self.assertEqual(plan["capacity_conflicts"], [])
        self.assertEqual(plan["temporary_capacity_pressure"][0]["resource"], "vram_gib")

    def test_explicit_estimate_remains_additive_and_can_be_infeasible(self):
        task = {**H3, "estimated_peak_vram_gib": 29}
        record = self.record(workload_key=workload_key("h3", task, CONFIG), peak_extra_vram_gib=1)
        plan = self.plan(telemetry(gpus=[{"index": 0, "free_gib": 31.84, "total_gib": 31.84}]), task=task, learned=record)
        self.assertEqual(plan["required"]["vram_gib"], 32)
        self.assertEqual(plan["minimum_required"]["vram_gib"], 32)
        self.assertEqual(plan["budget_breakdown"]["vram_gib"]["explicit_peak_extra_gib"], 29)
        self.assertEqual(plan["admission_state"], "policy_infeasible")
        self.assertFalse(plan["retryable"])

    def test_ram_and_commit_capacity_limits_are_checked(self):
        for task, sample, resource in (
            ({**H3, "estimated_peak_ram_gib": 60}, telemetry(ram_total_gib=64), "ram_gib"),
            ({**H3, "estimated_peak_commit_gib": 70}, telemetry(commit_limit_gib=80), "commit_gib"),
        ):
            plan = self.plan(sample, task=task)
            self.assertEqual(plan["admission_state"], "policy_infeasible")
            self.assertFalse(plan["retryable"])
            self.assertEqual(plan["capacity_conflicts"][0]["resource"], resource)

    def test_cold_startup_persistent_and_temporary_capacity_are_separate(self):
        self.p.observe(telemetry(cpu_percent=90), {})
        plan = self.plan(telemetry(commit_limit_gib=38), cold=True)
        self.assertEqual(plan["minimum_required"]["commit_gib"], 36)
        self.assertEqual(plan["required"]["commit_gib"], 40)
        self.assertEqual(plan["admission_state"], "waiting")
        self.assertTrue(plan["retryable"])
        plan = self.plan(telemetry(commit_limit_gib=35), cold=True)
        self.assertEqual(plan["admission_state"], "policy_infeasible")
        self.assertFalse(plan["retryable"])

    def test_learned_bias_remains_in_persistent_minimum(self):
        record = self.record(peak_extra_vram_gib=0, reserve_vram_bias_gib=3)
        plan = self.plan(telemetry(gpus=[{"index": 0, "free_gib": 5, "total_gib": 5}]), learned=record)
        self.assertEqual(plan["minimum_required"]["vram_gib"], 6)
        self.assertEqual(plan["admission_state"], "policy_infeasible")
        self.assertFalse(plan["retryable"])

    def test_startup_or_insufficient_samples_not_accepted(self):
        for updates in ({"completed": False}, {"samples": 2}, {"runs": 0}, {"measured_runs": 0},
                        {"peak_extra_ram_gib": math.nan}, {"workload_key": "other"}):
            plan = self.plan(learned=self.record(**updates))
            self.assertEqual(plan["confidence"], "configured_fallback")
        plan = self.plan(learned=self.record(), cold=True)
        self.assertEqual(plan["required"]["commit_gib"], 36)
        self.assertNotEqual(plan["confidence"], "measured_exact")

    def test_partial_shape_does_not_learn(self):
        partial = {"width": 1024}
        record = self.record(workload_key=workload_key("h3", partial, CONFIG))
        plan = self.plan(task=partial, learned=record)
        self.assertEqual(plan["confidence"], "configured_fallback")
        self.assertFalse(plan["profile_complete"])

    def test_dimensions_without_workflow_signature_do_not_calibrate(self):
        for model, source in (("h3", H3), ("muse", MUSE)):
            task = {key: value for key, value in source.items() if key != "workflow_signature"}
            record = self.record(workload_key=workload_key(model, task, CONFIG), reserve_ram_bias_gib=5)
            plan = self.p.plan(model, task, telemetry(), {}, learned=record)
            self.assertEqual(plan["confidence"], "configured_fallback")
            self.assertEqual(plan["reserves"]["ram_gib"], 12)
            self.assertFalse(plan["profile_complete"])

    def test_explicit_estimate_only_increases_required(self):
        plan = self.plan(task={**H3, "estimated_peak_ram_gib": 35})
        self.assertEqual(plan["required"]["ram_gib"], 47)
        self.assertFalse(plan["proceed"])
        self.assertEqual(plan["confidence"], "explicit_estimate")

    def test_lower_sample_peaks_cannot_reduce_fixed_floor(self):
        record = self.record(peak_extra_ram_gib=0, peak_extra_commit_gib=0, peak_extra_vram_gib=0)
        plan = self.plan(learned=record)
        self.assertGreaterEqual(plan["required"]["ram_gib"], 12)
        self.assertGreaterEqual(plan["required"]["vram_gib"], 3)

    def test_pressure_increments_cold_baseline(self):
        self.p.observe(telemetry(cpu_percent=90), {})
        plan = self.plan(cold=True)
        self.assertEqual(plan["required"], {"ram_gib": 24, "commit_gib": 40, "vram_gib": 25})

    def test_cpu_reserve_never_consumes_all_cores(self):
        self.p.observe(telemetry(cpu_percent=90, cpu_count=2), {})
        self.assertEqual(self.plan(telemetry(cpu_count=2))["runtime_profile"]["reserve_logical_cpus"], 1)

    def test_input_dictionaries_not_mutated(self):
        sample, models, learned = telemetry(), {"h3": {"cpu_percent": 5}}, self.record()
        original = copy.deepcopy((sample, models, learned, H3, CONFIG))
        self.p.observe(sample, models)
        self.p.plan("h3", H3, sample, models, learned)
        self.assertEqual(original, (sample, models, learned, H3, CONFIG))

    def test_failure_learning_raises_reserves_without_claiming_calibration(self):
        record = {"workload_key": workload_key("h3", H3, CONFIG),
                  "reserve_ram_bias_gib": 4, "reserve_commit_bias_gib": 6,
                  "reserve_vram_bias_gib": 2, "disable_warm_cache": True}
        plan = self.plan(learned=record)
        self.assertEqual(plan["reserves"], {"ram_gib": 16, "commit_gib": 18, "vram_gib": 5})
        self.assertEqual(plan["confidence"], "configured_fallback")
        self.assertEqual(plan["warm_seconds"], 0)
        self.assertEqual(plan["task"], H3)

    def test_invalid_learning_biases_or_wrong_identity_cannot_change_budget(self):
        record = {"workload_key": workload_key("h3", H3, CONFIG),
                  "reserve_ram_bias_gib": math.inf, "reserve_commit_bias_gib": -1,
                  "reserve_vram_bias_gib": 999, "disable_warm_cache": "true"}
        plan = self.plan(learned=record)
        self.assertEqual(plan["reserves"], {"ram_gib": 12, "commit_gib": 12, "vram_gib": 3})
        self.assertEqual(plan["warm_seconds"], 120)
        record.update(workload_key="other", reserve_ram_bias_gib=4, disable_warm_cache=True)
        plan = self.plan(learned=record)
        self.assertEqual(plan["reserves"]["ram_gib"], 12)
        self.assertEqual(plan["warm_seconds"], 120)


if __name__ == "__main__":
    unittest.main()
