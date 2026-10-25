"""Temporary-only registry integration: real broker/storage, fake model adapters."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from broker import Broker


class InertAdapter:
    def __init__(self, model, config):
        self.model, self.config = model, copy.deepcopy(config)
        self.starts = 0
        self.current = {"state": "stopped", "busy": False, "identity": None}

    def inspect(self):
        return copy.deepcopy(self.current)

    def start(self):
        self.starts += 1
        raise AssertionError("Registration must never start a model")

    def stop(self, identity):
        raise AssertionError("Registration must never stop a model")


class RegistryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.state_root = self.root / "state"
        self.model_root = self.root / "model"
        self.model_root.mkdir()
        self.exe = self.model_root / "llama-server.exe"
        self.exe.touch()
        self.model_file = self.model_root / "model.gguf"
        self.model_file.touch()
        self.manifest = {
            "label": "Integration model", "kind": "llamacpp", "workload_kind": "llm",
            "root": str(self.model_root), "exe": str(self.exe), "url": "http://127.0.0.1:8090",
            "context_capacity": 32768,
            "start_argv": [str(self.exe), "-m", str(self.model_file), "-c", "32768", "-np", "1",
                           "--host", "127.0.0.1", "--port", "8090", "--slots"],
            "startup_free_ram_gib": 16, "startup_commit_free_gib": 26, "startup_free_vram_gib": 22}
        self.config = {
            "schema_version": 1, "sample_seconds": 5, "idle_grace_seconds": 120,
            "min_available_ram_gib": 12, "min_commit_free_gib": 12, "min_free_vram_gib": 3,
            "models": {"h3": {
                "label": "Fake H3", "kind": "comfyui", "workload_kind": "video",
                "root": str(self.model_root), "exe": str(self.exe), "url": "http://127.0.0.1:8188",
                "startup_free_ram_gib": 20, "startup_commit_free_gib": 36, "startup_free_vram_gib": 24}}}
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        self.created = []
        self.next_state = "stopped"
        self.patch_adapter = patch("adapters.Adapter", side_effect=self.make_adapter)
        self.patch_adapter.start()
        self.patch_spawn = patch("subprocess.Popen", side_effect=AssertionError("No process launch in registration test"))
        self.patch_spawn.start()
        self.brokers = []
        self.b = self.new_broker(self.config)

    def make_adapter(self, model, config):
        adapter = InertAdapter(model, config)
        if self.next_state != "stopped":
            adapter.current = {"state": self.next_state, "busy": False, "identity": {"pid": 12345}}
        self.created.append(adapter)
        return adapter

    def new_broker(self, config):
        adapters = {k: self.make_adapter(k, v) for k, v in config["models"].items()}
        healthy = {"ram_total_gib": 64, "ram_available_gib": 50, "commit_free_gib": 80,
                   "gpus": [{"index": 0, "free_gib": 30, "used_gib": 2, "total_gib": 32}], "errors": []}
        broker = Broker(self.state_root, copy.deepcopy(config), adapters, lambda: copy.deepcopy(healthy),
                        config_path=self.config_path)
        self.brokers.append(broker)
        return broker

    def tearDown(self):
        for broker in self.brokers:
            broker.db.close()
        self.patch_spawn.stop()
        self.patch_adapter.stop()
        self.temp.cleanup()

    def register(self, **kwargs):
        return self.b.register_model("extra-llm", self.manifest, **kwargs)

    def assert_no_model_start(self):
        self.assertTrue(all(adapter.starts == 0 for adapter in self.created))

    def test_registration_updates_persistent_config_and_adapter_without_start(self):
        result = self.register()
        self.assertEqual(result["registered"], "extra-llm")
        self.assertFalse(result["started"])
        persisted = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["models"]["extra-llm"]["context_capacity"], 32768)
        self.assertEqual(persisted["models"]["h3"], self.config["models"]["h3"])
        self.assertIn("extra-llm", self.b.adapters)
        self.assertEqual(self.b.models["extra-llm"]["state"], "stopped")
        self.assertEqual(self.b.config["models"]["extra-llm"], persisted["models"]["extra-llm"])
        self.assert_no_model_start()

    def test_queued_or_active_work_blocks_registration_without_config_change(self):
        for state in ("queued", "active"):
            with self.subTest(state=state):
                self.b.state["leases"] = []
                self.b.request_model("h3", "test-chat", state)
                self.b.state["leases"][0]["state"] = state
                before = self.config_path.read_bytes()
                with self.assertRaisesRegex(ValueError, "quiescent"):
                    self.register()
                self.assertEqual(self.config_path.read_bytes(), before)
                self.assertNotIn("extra-llm", self.b.adapters)
        self.assert_no_model_start()

    def test_queued_cleanup_and_inflight_lifecycle_block_registration(self):
        before = self.config_path.read_bytes()
        self.b.state["cleanup"] = [{"state": "queued"}]
        with self.assertRaisesRegex(ValueError, "quiescent"):
            self.register()
        self.b.state["cleanup"] = []
        self.b.lifecycle_action = {"action": "start"}
        with self.assertRaisesRegex(ValueError, "quiescent"):
            self.register()
        self.assertEqual(self.config_path.read_bytes(), before)
        self.assert_no_model_start()

    def test_duplicate_service_port_rejected_without_mutation(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["url"] = "http://127.0.0.1:8188"
        manifest["start_argv"][manifest["start_argv"].index("--port") + 1] = "8188"
        before = self.config_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "URL already"):
            self.b.register_model("extra-llm", manifest)
        self.assertEqual(self.config_path.read_bytes(), before)
        self.assertNotIn("extra-llm", self.b.adapters)
        self.assert_no_model_start()

    def test_running_existing_target_blocks_explicit_replace(self):
        self.register()
        self.b.adapters["extra-llm"].current = {"state": "ready", "busy": False, "identity": {"pid": 12345}}
        before = self.config_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "must be stopped"):
            self.register(replace=True)
        self.assertEqual(self.config_path.read_bytes(), before)
        self.assert_no_model_start()

    def test_unregistered_active_service_not_implicitly_adopted(self):
        self.next_state = "ready"
        before = self.config_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "do not adopt"):
            self.register()
        self.assertEqual(self.config_path.read_bytes(), before)
        self.assertNotIn("extra-llm", self.b.adapters)
        self.assert_no_model_start()

    def test_original_config_backup_created_before_update(self):
        original = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.register()
        backups = list(self.state_root.glob("config-backup-*.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(json.loads(backups[0].read_text(encoding="utf-8")), original)
        self.assertFalse(self.config_path.with_suffix(".json.new").exists())
        self.assert_no_model_start()

    def test_restart_loads_registered_model_from_saved_config(self):
        self.register()
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        restarted = self.new_broker(config)
        self.assertIn("extra-llm", restarted.adapters)
        self.assertEqual(restarted.adapters["extra-llm"].config["start_argv"], self.manifest["start_argv"])
        event = restarted.db.execute("SELECT data FROM events WHERE kind='model_registered'").fetchone()
        self.assertEqual(json.loads(event[0])["model"], "extra-llm")
        self.assert_no_model_start()

    def test_atomic_replace_failure_preserves_live_config_and_adapters(self):
        before = self.config_path.read_bytes()
        with patch("broker_adaptive.os.replace", side_effect=OSError("injected filesystem fault")):
            with self.assertRaises(OSError):
                self.register()
        self.assertEqual(self.config_path.read_bytes(), before)
        self.assertNotIn("extra-llm", self.b.adapters)
        self.assertNotIn("extra-llm", self.b.config["models"])
        self.assert_no_model_start()


if __name__ == "__main__":
    unittest.main()
