import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import os
from broker import Broker
from resources import pressure_reasons

CONFIG = json.loads((Path(__file__).parent / 'tests/fixtures/config.json').read_text())
HEALTHY = {"ram_available_gib": 50, "commit_free_gib": 60,
           "gpus": [{"index": 0, "free_gib": 28, "used_gib": 4}], "errors": []}


class FakeAdapter:
    def __init__(self, model):
        self.model, self.starts, self.stops = model, 0, 0
        self.current = {"state": "stopped", "busy": False, "identity": None}
        self.fail_start = False
        self.fail_stop = False

    def inspect(self):
        return copy.deepcopy(self.current)

    def external(self, busy=False):
        self.current = {"state": "ready", "busy": busy,
                        "identity": {"pid": 12 if self.model == "h3" else 13, "created": 123.0, "exe": self.model}}

    def start(self):
        self.starts += 1
        if self.fail_start:
            raise RuntimeError("injected startup failure")
        self.external()
        return self.current["identity"]

    def stop(self, identity):
        if self.fail_stop or self.current["busy"] is not False or self.current["state"] != "ready" or self.current["identity"] != identity:
            raise RuntimeError("stop refused")
        self.stops += 1
        self.current = {"state": "stopped", "busy": False, "identity": None}


class BrokerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.adapters = {k: FakeAdapter(k) for k in CONFIG["models"]}
        self.sample = copy.deepcopy(HEALTHY)
        self.b = Broker(self.temp.name, CONFIG, self.adapters, lambda: copy.deepcopy(self.sample))

    def tearDown(self):
        self.b.db.close()
        self.temp.cleanup()

    def request(self, model="h3", key="a", **kwargs):
        return self.b.request_model(model, "test-chat", key, **kwargs)

    def lease(self, x, renew=False):
        return self.b.lease_status(x["id"], x["token"], renew)

    def release(self, x):
        return self.b.release(x["id"], x["token"])

    def test_start_release_and_measured_history(self):
        a = self.request()
        self.b.tick()
        self.assertTrue(self.lease(a)["proceed"])
        self.assertEqual(self.adapters["h3"].starts, 1)
        self.release(a)
        self.assertFalse(self.lease(a)["proceed"])
        self.b.tick()
        self.assertEqual(self.lease(a)["state"], "released")
        self.assertEqual(self.adapters["h3"].stops, 1)
        self.assertGreater(self.b.history()["leases"][0]["measurements"]["samples"], 0)

    def test_global_fifo_between_chats(self):
        a = self.request()
        b = self.b.request_model("muse", "another-chat", "b")
        self.b.tick()
        self.assertEqual(self.lease(b)["state"], "queued")
        self.release(a)
        self.b.tick()
        self.b.tick()
        self.assertTrue(self.lease(b)["proceed"])

    def test_external_service_not_stopped_on_conflict(self):
        self.adapters["muse"].external()
        a = self.request()
        self.b.tick()
        self.assertEqual(self.lease(a)["state"], "queued")
        self.assertEqual(self.adapters["muse"].stops, 0)
        with self.assertRaisesRegex(ValueError, "External service"):
            self.b.release_idle("muse")

    def test_explicit_external_release_then_queue(self):
        self.adapters["muse"].external()
        a = self.request()
        self.b.release_idle("muse", True)
        self.b.tick()
        self.b.tick()
        self.assertEqual(self.adapters["muse"].stops, 1)
        self.assertTrue(self.lease(a)["proceed"])

    def test_external_reuse_preserved(self):
        self.adapters["muse"].external()
        a = self.request("muse")
        self.b.tick()
        self.assertTrue(self.lease(a)["proceed"])
        self.release(a)
        self.b.tick()
        self.assertEqual(self.lease(a)["state"], "released")
        self.assertEqual(self.adapters["muse"].stops, 0)

    def test_busy_release_deferred(self):
        a = self.request()
        self.b.tick()
        self.adapters["h3"].current["busy"] = True
        self.release(a)
        self.b.tick()
        self.assertEqual(self.adapters["h3"].stops, 0)
        self.adapters["h3"].current["busy"] = False
        self.b.tick()
        self.assertEqual(self.adapters["h3"].stops, 1)

    def test_expired_busy_lease_holds_gpu(self):
        a = self.request()
        b = self.request("muse", "b")
        self.b.tick()
        self.b.state["leases"][0]["expires"] = 0
        self.adapters["h3"].current["busy"] = True
        self.b.tick()
        self.assertEqual(self.lease(a)["state"], "orphaned")
        self.assertEqual(self.lease(b)["state"], "queued")
        self.assertEqual(self.adapters["h3"].stops, 0)

    def test_unknown_telemetry_blocks_admission(self):
        self.sample["commit_free_gib"] = None
        a = self.request()
        self.b.tick()
        self.assertFalse(self.lease(a)["proceed"])
        self.assertEqual(self.adapters["h3"].starts, 0)

    def test_gpu_memory_pressure_blocks_admission(self):
        self.sample["gpus"][0]["free_gib"] = 5
        a = self.request()
        self.b.tick()
        self.assertEqual(self.lease(a)["state"], "queued")

    def test_pressure_pauses_submissions_without_killing(self):
        a = self.request()
        self.b.tick()
        self.sample["ram_available_gib"] = 3
        self.b.tick()
        self.assertFalse(self.lease(a)["proceed"])
        self.assertEqual(self.adapters["h3"].stops, 0)

    def test_stale_sample_blocks_proceed(self):
        a = self.request()
        self.b.tick()
        self.b.last_sample = 0
        self.assertFalse(self.lease(a)["proceed"])

    def test_wrong_token_cannot_release(self):
        a = self.request()
        with self.assertRaises(ValueError):
            self.b.release(a["id"], "wrong")
        self.assertNotIn("token", self.b.status()["leases"][0])
        self.assertNotIn("token", self.b.history()["leases"][0])

    def test_idempotent_concurrent_requests(self):
        results = []
        threads = [threading.Thread(target=lambda: results.append(self.request())) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len({x["id"] for x in results}), 1)
        self.assertEqual(len(self.b.state["leases"]), 1)

    def test_request_key_collision_rejected(self):
        self.request()
        with self.assertRaises(ValueError):
            self.request("muse")

    def test_queued_cancel_does_not_start(self):
        a = self.request()
        self.release(a)
        self.b.tick()
        self.assertEqual(self.lease(a)["state"], "cancelled")
        self.assertEqual(self.adapters["h3"].starts, 0)

    def test_warm_handoff_stops_only_owned_service(self):
        a = self.request(keep_loaded=True)
        self.b.tick()
        self.release(a)
        self.b.tick()
        self.assertEqual(self.adapters["h3"].stops, 0)
        b = self.request("muse", "b")
        self.b.tick()
        self.b.tick()
        self.assertTrue(self.lease(b)["proceed"])
        self.assertEqual(self.adapters["h3"].stops, 1)

    def test_startup_failure_not_retried(self):
        self.adapters["h3"].fail_start = True
        a = self.request()
        self.b.tick()
        self.b.tick()
        self.assertEqual(self.lease(a)["state"], "failed")
        self.assertEqual(self.adapters["h3"].starts, 1)

    def test_explicit_release_rechecks_busy(self):
        self.adapters["h3"].external()
        self.b.release_idle("h3", True)
        self.adapters["h3"].current["busy"] = True
        self.b.tick()
        self.assertEqual(self.adapters["h3"].stops, 0)
        self.assertEqual(self.b.status()["cleanup"][0]["state"], "failed")

    def test_active_lease_prevents_cleanup(self):
        a = self.request()
        self.b.tick()
        with self.assertRaises(ValueError):
            self.b.release_idle("h3")

    def test_restart_preserves_active_and_credential(self):
        a = self.request()
        self.b.tick()
        self.b.db.close()
        self.b = Broker(self.temp.name, CONFIG, self.adapters, lambda: self.sample)
        self.b.tick()
        self.assertTrue(self.lease(a)["proceed"])
        self.assertEqual(self.adapters["h3"].starts, 1)

    def test_reused_pid_identity_change_not_killed(self):
        a = self.request()
        self.b.tick()
        self.adapters["h3"].current["identity"]["created"] = 999
        self.b.tick()
        self.assertFalse(self.lease(a)["proceed"])
        self.assertEqual(self.lease(a)["state"], "failed")
        self.assertEqual(self.adapters["h3"].stops, 0)

    def test_expired_queued_lease_not_started(self):
        a = self.request()
        self.b.state["leases"][0]["expires"] = 0
        self.b.tick()
        self.assertEqual(self.lease(a)["state"], "expired")
        self.assertEqual(self.adapters["h3"].starts, 0)

    def test_registered_live_worker_prevents_expiry_cleanup(self):
        a = self.request()
        self.b.tick()
        self.b.attach_worker(a["id"], a["token"], os.getpid())
        self.b.state["leases"][0]["expires"] = 0
        self.b.tick()
        self.assertEqual(self.lease(a)["state"], "orphaned")
        self.assertEqual(self.adapters["h3"].stops, 0)

    def test_stop_failure_is_not_retried_automatically(self):
        a = self.request()
        self.b.tick()
        self.adapters["h3"].fail_stop = True
        self.release(a)
        self.b.tick()
        self.adapters["h3"].fail_stop = False
        self.b.tick()
        self.assertEqual(self.adapters["h3"].stops, 0)
        self.release(a)
        self.b.tick()
        self.assertEqual(self.adapters["h3"].stops, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
