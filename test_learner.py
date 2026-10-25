import copy
import json
import unittest

import learner


class LearnerTests(unittest.TestCase):
    def setUp(self):
        self.state = learner.initial_state()
        self.key = learner.profile_key("h3", {"width": 720, "frames": 121, "steps": 40}, "cfg-1", "cold")

    def measurement(self, **changes):
        result = {"before": {"ram_available_gib": 40, "commit_free_gib": 55, "free_vram_gib": 29},
                  "peak_extra_ram_gib": 12, "peak_extra_commit_gib": 16,
                  "peak_extra_vram_gib": 22, "busy_samples": 10,
                  "elapsed_seconds": 50, "pressure_observed": False}
        result.update(changes)
        return result

    def record(self, job_id, **feedback):
        return learner.record_run(self.state, self.key, job_id, self.measurement(), feedback)

    def policy(self):
        return learner.get_policy(self.state, self.key)

    def clean(self, job_id, **changes):
        return learner.record_run(self.state, self.key, job_id, self.measurement(**changes),
                                  {"outcome": "completed", "quality": "passed", "desktop_responsive": True})

    def test_exact_task_cache_configuration_scope(self):
        self.assertEqual(self.key, learner.profile_key("h3", {"steps": 40, "frames": 121, "width": 720}, "cfg-1", "cold"))
        for key in [learner.profile_key("h3", {"width": 720, "frames": 122, "steps": 40}, "cfg-1", "cold"),
                    learner.profile_key("h3", {"width": 720, "frames": 121, "steps": 40}, "cfg-2", "cold"),
                    learner.profile_key("h3", {"width": 720, "frames": 121, "steps": 40}, "cfg-1", "warm")]:
            self.assertNotEqual(self.key, key)

    def test_unknown_outcome_does_not_claim_completed_peaks_or_quality(self):
        self.record("a")
        p = self.policy()
        self.assertEqual(p["samples"], 0)
        self.assertIsNone(p["peak_extra_ram_gib"])
        self.assertFalse(p["quarantined"])

    def test_completed_measured_run_learns_peak_but_not_qa(self):
        self.record("a", outcome="completed")
        p = self.policy()
        self.assertEqual(p["samples"], 10)
        self.assertEqual(p["measured_runs"], 1)
        self.assertEqual(p["peak_extra_vram_gib"], 22)
        self.assertEqual(self.state["profiles"][self.key]["clean_streak"], 0)

    def test_short_missing_and_uncompleted_runs_do_not_calibrate(self):
        values = [(self.measurement(busy_samples=2), "completed"),
                  (self.measurement(peak_extra_commit_gib=None), "completed"),
                  (self.measurement(before={}), "completed"),
                  (self.measurement(before={"ram_available_gib": 20, "commit_free_gib": None, "free_vram_gib": None}), "completed"),
                  (self.measurement(before=None), "completed"),
                  (self.measurement(elapsed_seconds=0), "completed"),
                  (self.measurement(), "failed")]
        for index, (measurement, outcome) in enumerate(values):
            learner.record_run(self.state, self.key, str(index), measurement, {"outcome": outcome})
        self.assertIsNone(self.policy()["peak_extra_ram_gib"])
        self.assertEqual(self.policy()["samples"], 0)

    def test_late_completion_and_qa_calibrate_once(self):
        self.record("a")
        learner.report_feedback(self.state, self.key, "a", "exit", {"outcome": "completed"})
        self.assertEqual(self.policy()["samples"], 10)
        learner.report_feedback(self.state, self.key, "a", "qa", {"quality": "passed", "desktop_responsive": True})
        self.assertEqual(self.state["profiles"][self.key]["clean_streak"], 1)
        duplicate = learner.report_feedback(self.state, self.key, "a", "qa", {"quality": "passed", "desktop_responsive": True})
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(self.state["profiles"][self.key]["clean_streak"], 1)

    def test_pressure_automatically_raises_bounded_safeguards(self):
        for i in range(10):
            learner.record_run(self.state, self.key, str(i), self.measurement(pressure_observed=True), {})
        p = self.policy()
        self.assertEqual(p["reserve_ram_bias_gib"], 8)
        self.assertEqual(p["reserve_commit_bias_gib"], 8)
        self.assertEqual(p["reserve_vram_bias_gib"], 3)
        self.assertTrue(p["disable_warm_cache"])
        self.assertFalse(p["quarantined"])

    def test_cpu_only_pressure_does_not_raise_memory_bias(self):
        learner.record_run(self.state, self.key, "cpu", self.measurement(pressure_observed=True, pressure_resources=[]), {})
        p = self.policy()
        for resource in learner.RESOURCES:
            self.assertEqual(p["reserve_" + resource + "_bias_gib"], 0)
        self.assertTrue(p["disable_warm_cache"])
        self.assertFalse(p["quarantined"])

    def test_pressure_attribution_only_increases_observed_resource(self):
        learner.record_run(self.state, self.key, "ram", self.measurement(pressure_observed=True, pressure_resources=["ram", "ram"]), {})
        p = self.policy()
        self.assertEqual(p["reserve_ram_bias_gib"], 2)
        self.assertEqual(p["reserve_commit_bias_gib"], 0)
        self.assertEqual(p["reserve_vram_bias_gib"], 0)
        self.assertEqual(self.state["profiles"][self.key]["jobs"]["ram"]["measurement"]["pressure_resources"], ["ram"])

    def test_late_unknown_cause_feedback_adds_remaining_resources_once(self):
        learner.record_run(self.state, self.key, "ram", self.measurement(pressure_observed=True, pressure_resources=["ram"]), {})
        learner.report_feedback(self.state, self.key, "ram", "desktop", {"desktop_responsive": False})
        p = self.policy()
        self.assertEqual(p["reserve_ram_bias_gib"], 2)
        self.assertEqual(p["reserve_commit_bias_gib"], 2)
        self.assertEqual(p["reserve_vram_bias_gib"], .5)
        learner.report_feedback(self.state, self.key, "ram", "oom", {"outcome": "oom"})
        self.assertEqual(self.policy()["reserve_ram_bias_gib"], 2)
        self.assertEqual(self.policy()["reserve_vram_bias_gib"], .5)

    def test_cpu_only_pressure_followed_oom_retains_unknown_cause_safeguards(self):
        learner.record_run(self.state, self.key, "cpu", self.measurement(pressure_observed=True, pressure_resources=[]), {})
        learner.report_feedback(self.state, self.key, "cpu", "oom", {"outcome": "oom"})
        self.assertEqual(self.policy()["reserve_ram_bias_gib"], 2)
        self.assertEqual(self.policy()["reserve_commit_bias_gib"], 2)
        self.assertEqual(self.policy()["reserve_vram_bias_gib"], .5)

    def test_cpu_only_pressure_does_not_undo_memory_relaxation(self):
        self.record("bad", outcome="oom")
        for i in range(5):
            self.clean(str(i))
        prior = self.policy()
        learner.record_run(self.state, self.key, "cpu", self.measurement(pressure_observed=True, pressure_resources=[]), {})
        after = self.policy()
        for resource in learner.RESOURCES:
            field = "reserve_" + resource + "_bias_gib"
            self.assertEqual(prior[field], after[field])
        self.assertTrue(after["disable_warm_cache"])

    def test_resource_rollback_does_not_invent_other_resource_pressure(self):
        self.record("bad", outcome="oom")
        for i in range(5):
            self.clean(str(i))
        learner.record_run(self.state, self.key, "ram", self.measurement(pressure_observed=True, pressure_resources=["ram"]), {})
        self.assertEqual(self.policy()["reserve_ram_bias_gib"], 4)
        self.assertEqual(self.policy()["reserve_commit_bias_gib"], 1.5)
        self.assertEqual(self.policy()["reserve_vram_bias_gib"], .375)
        # A later quality failure can still roll back the un-restored resources.
        self.record("qa", quality="failed")
        self.assertEqual(self.policy()["reserve_commit_bias_gib"], 2)
        self.assertEqual(self.policy()["reserve_vram_bias_gib"], .5)

    def test_legacy_observation_is_conservative_without_history_rewrite(self):
        self.record("legacy", desktop_responsive=False)
        job = self.state["profiles"][self.key]["jobs"]["legacy"]
        del job["bias_applied_resources"]  # Snapshot from a pre-attribution version.
        before_measurement = copy.deepcopy(job["measurement"])
        learner.report_feedback(self.state, self.key, "legacy", "oom", {"outcome": "oom"})
        self.assertNotIn("pressure_resources", job["measurement"])
        self.assertEqual(job["measurement"], before_measurement)
        self.assertEqual(self.policy()["reserve_vram_bias_gib"], .5)

    def test_invalid_pressure_attribution_rejected_before_mutation(self):
        for affected in (None, "ram", ["cpu"], [True], ["ram"] * 4):
            before = copy.deepcopy(self.state)
            with self.assertRaises(ValueError):
                learner.record_run(self.state, self.key, "invalid", self.measurement(pressure_observed=True, pressure_resources=affected), {})
            self.assertEqual(before, self.state)
        with self.assertRaises(ValueError):
            learner.record_run(self.state, self.key, "contradiction", self.measurement(pressure_observed=False, pressure_resources=["ram"]), {})

    def test_oom_and_headroom_same_job_only_raise_once(self):
        self.record("a", desktop_responsive=False)
        learner.report_feedback(self.state, self.key, "a", "oom", {"outcome": "oom"})
        p = self.policy()
        self.assertEqual(p["reserve_ram_bias_gib"], 2)
        self.assertEqual(p["reserve_vram_bias_gib"], .5)

    def test_quality_failure_quarantines_without_inventing_memory_cause(self):
        self.record("a", outcome="completed", quality="failed")
        p = self.policy()
        self.assertTrue(p["quarantined"])
        self.assertTrue(p["disable_warm_cache"])
        self.assertEqual(p["reserve_ram_bias_gib"], 0)
        self.assertNotIn("steps", p)
        self.assertNotIn("quantization", p)
        learner.report_feedback(self.state, self.key, "a", "retract", {"quality": "passed", "desktop_responsive": True})
        self.assertTrue(self.policy()["quarantined"])

    def test_three_later_clean_runs_restore_cache_after_quarantine(self):
        self.record("bad", quality="failed")
        for i in range(2):
            self.clean(str(i))
            self.assertTrue(self.policy()["quarantined"])
        self.clean("third")
        self.assertFalse(self.policy()["quarantined"])
        self.assertFalse(self.policy()["disable_warm_cache"])

    def test_old_late_success_cannot_clear_new_adverse_evidence(self):
        for i in range(3):
            self.record(str(i))
        self.record("bad", quality="failed")
        for i in range(3):
            learner.report_feedback(self.state, self.key, str(i), "late", {"outcome": "completed", "quality": "passed", "desktop_responsive": True})
        self.assertTrue(self.policy()["quarantined"])

    def test_five_comparable_clean_runs_ease_only_extra_margin(self):
        self.record("bad", outcome="oom")
        for i in range(5):
            self.clean(str(i))
        p = self.policy()
        self.assertEqual(p["reserve_ram_bias_gib"], 1.5)
        self.assertEqual(p["reserve_vram_bias_gib"], .375)
        self.assertEqual(p["peak_extra_vram_gib"], 22)
        self.assertTrue(any("cautiously ease" in h["reason"] for h in learner.history(self.state, self.key)))

    def test_incomparable_durations_do_not_relax(self):
        self.record("bad", outcome="oom")
        for i in range(5):
            self.clean(str(i), elapsed_seconds=50 + i * 50)
        self.assertEqual(self.policy()["reserve_ram_bias_gib"], 2)

    def test_unknown_quality_never_relaxes(self):
        self.record("bad", outcome="oom")
        for i in range(10):
            self.record(str(i), outcome="completed", quality="unknown", desktop_responsive=True)
        self.assertEqual(self.policy()["reserve_ram_bias_gib"], 2)

    def test_quiet_telemetry_plus_qa_relaxes_without_subjective_feedback(self):
        self.record("bad", outcome="oom")
        for i in range(5):
            self.record(str(i), outcome="completed", quality="passed", desktop_responsive=None)
        self.assertEqual(self.policy()["reserve_ram_bias_gib"], 1.5)
        self.assertIsNone(self.state["profiles"][self.key]["jobs"]["4"]["feedback"]["desktop_responsive"])

    def test_explicit_unresponsive_feedback_blocks_relaxation(self):
        self.record("bad", outcome="oom")
        for i in range(5):
            self.record(str(i), outcome="completed", quality="passed", desktop_responsive=False)
        self.assertEqual(self.policy()["reserve_ram_bias_gib"], 8)
        self.assertTrue(self.policy()["disable_warm_cache"])

    def test_new_adversity_rolls_back_last_relaxation_then_adds_safeguard(self):
        self.record("bad", outcome="oom")
        for i in range(5):
            self.clean(str(i))
        self.assertEqual(self.policy()["reserve_ram_bias_gib"], 1.5)
        result = self.record("again", desktop_responsive=False)
        self.assertEqual(self.policy()["reserve_ram_bias_gib"], 4)
        self.assertIn("rollback previous efficiency", result["reason"])

    def test_duplicate_run_conflicting_data_rejected_without_mutation(self):
        self.record("a", outcome="oom")
        before = copy.deepcopy(self.state)
        duplicate = self.record("a", outcome="oom")
        self.assertTrue(duplicate["duplicate"])
        with self.assertRaises(ValueError):
            self.record("a", outcome="completed")
        self.assertEqual(before, self.state)

    def test_conflicting_feedback_id_rejected_without_mutation(self):
        self.record("a")
        learner.report_feedback(self.state, self.key, "a", "report", {"quality": "failed"})
        before = copy.deepcopy(self.state)
        with self.assertRaises(ValueError):
            learner.report_feedback(self.state, self.key, "a", "report", {"quality": "passed"})
        self.assertEqual(before, self.state)

    def test_evicted_run_cannot_reapply_learning(self):
        self.record("old", outcome="oom")
        for i in range(learner.MAX_JOBS + 2):
            self.record(str(i))
        before = copy.deepcopy(self.state)
        with self.assertRaises(ValueError):
            self.record("old", outcome="oom")
        with self.assertRaises(ValueError):
            learner.report_feedback(self.state, self.key, "old", "late", {"outcome": "oom"})
        self.assertEqual(before, self.state)

    def test_bad_measurements_rejected_before_state_change(self):
        for changes in [{"busy_samples": True}, {"peak_extra_ram_gib": float("nan")},
                        {"elapsed_seconds": -1}, {"pressure_observed": 1}, {"untrusted": "setting"}]:
            before = copy.deepcopy(self.state)
            with self.assertRaises(ValueError):
                learner.record_run(self.state, self.key, "bad", self.measurement(**changes), {})
            self.assertEqual(before, self.state)

    def test_unknown_feedback_fields_rejected(self):
        for feedback in [{"quality": "good"}, {"outcome": "success"}, {"desktop_responsive": 1}, {"steps": 20}]:
            with self.assertRaises(ValueError):
                self.record("a", **feedback)
        self.assertEqual(self.state["profiles"], {})

    def test_rollback_preserves_new_safety_evidence(self):
        self.record("a", outcome="completed")
        target = self.policy()["revision"]
        self.record("b", quality="failed", desktop_responsive=False)
        learner.rollback(self.state, self.key, target, "review prior scheduler")
        p = self.policy()
        self.assertTrue(p["quarantined"])
        self.assertTrue(p["disable_warm_cache"])
        self.assertEqual(p["reserve_ram_bias_gib"], 2)

    def test_serialization_roundtrip_and_return_isolation(self):
        self.record("a", outcome="completed")
        reloaded = json.loads(json.dumps(self.state))
        learner.report_feedback(reloaded, self.key, "a", "late", {"quality": "passed", "desktop_responsive": True})
        got = learner.get_policy(reloaded, self.key)
        got["reserve_ram_bias_gib"] = -5
        self.assertGreaterEqual(learner.get_policy(reloaded, self.key)["reserve_ram_bias_gib"], 0)

    def test_state_bound_with_lru_archival(self):
        for n in range(learner.MAX_PROFILES + learner.MAX_ARCHIVES + 1):
            key = learner.profile_key("muse", "task-" + str(n), "config", "warm")
            for i in range(learner.MAX_JOBS + 2):
                learner.record_run(self.state, key, "job-" + str(i), self.measurement(), {"outcome": "completed"})
            p = self.state["profiles"][key]
            self.assertLessEqual(len(p["jobs"]), learner.MAX_JOBS)
            self.assertLessEqual(len(p["revisions"]), learner.MAX_REVISIONS)
            self.assertLessEqual(len(p["run_ids"]), learner.MAX_RUNS)
        encoded = json.dumps(self.state, separators=(",", ":"))
        self.assertLess(len(encoded.encode()), 1024 * 1024)
        self.assertEqual(len(self.state["profiles"]), learner.MAX_PROFILES)
        self.assertEqual(len(self.state["archives"]), learner.MAX_ARCHIVES)
        learner.record_run(self.state, self.key, "overflow", self.measurement(), {})
        self.assertIn(self.key, self.state["profiles"])
        self.assertEqual(len(self.state["profiles"]), learner.MAX_PROFILES)
        for archived in self.state["archives"].values():
            self.assertNotIn("jobs", archived)
            self.assertNotIn("revisions", archived)

    def test_archive_restore_retains_adverse_safeguards(self):
        self.record("bad", quality="failed", desktop_responsive=False)
        for i in range(learner.MAX_PROFILES):
            key = learner.profile_key("muse", "task-" + str(i), "cfg", "warm")
            learner.record_run(self.state, key, "run", self.measurement(), {})
        self.assertIn(self.key, self.state["archives"])
        self.assertTrue(self.policy()["quarantined"])
        self.record("return")
        self.assertIn(self.key, self.state["profiles"])
        self.assertTrue(self.policy()["quarantined"])
        self.assertEqual(self.policy()["reserve_ram_bias_gib"], 2)
        with self.assertRaises(ValueError):
            self.record("bad", quality="failed", desktop_responsive=False)

    def test_unknown_cache_bucket_is_distinct_and_new_policy_peaks_none(self):
        key = learner.profile_key("h3", "task", "cfg", "unknown")
        self.assertNotEqual(key, learner.profile_key("h3", "task", "cfg", "warm"))
        self.assertIsNone(learner.get_policy(self.state, key)["peak_extra_ram_gib"])


if __name__ == "__main__":
    unittest.main()
