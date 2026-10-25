import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from registry import register_config, validate_manifest


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.exe = self.root / "llama-server.exe"
        self.exe.touch()
        self.script = self.root / "main.py"
        self.script.write_text("# existing installed model server")
        self.key = self.root / "api-key.txt"
        self.key.write_text("secret-not-to-be-read")
        self.manifest = {"label": "Test LLM", "kind": "llamacpp", "workload_kind": "llm",
                         "root": str(self.root), "exe": str(self.exe), "url": "http://127.0.0.1:8090",
                         "context_capacity": 32768,
                         "start_argv": [str(self.exe), "-m", str(self.root / "model.gguf"), "-c", "32768",
                                        "-np", "1", "--host", "127.0.0.1", "--port", "8090", "--slots"],
                         "startup_free_ram_gib": 16, "startup_commit_free_gib": 26,
                         "startup_free_vram_gib": 22}

    def tearDown(self):
        self.temp.cleanup()

    def valid(self, **changes):
        m = copy.deepcopy(self.manifest)
        m.update(changes)
        return validate_manifest("another-llm", m)

    def test_llama_manifest_defaults_cwd_and_preserves_parameters(self):
        original = copy.deepcopy(self.manifest)
        result = self.valid()
        self.assertEqual(self.manifest, original)
        self.assertEqual(result["start_cwd"], str(self.root))
        self.assertEqual(result["start_argv"], original["start_argv"])
        self.assertEqual(result["context_capacity"], 32768)

    def test_comfy_manifest_uses_video_backend(self):
        exe = self.root / "python.exe"
        exe.touch()
        m = copy.deepcopy(self.manifest)
        m.update(kind="comfyui", workload_kind="video", exe=str(exe),
                 start_argv=[str(exe), "-s", str(self.script), "--listen", "127.0.0.1", "--port", "8090"])
        del m["context_capacity"]
        result = validate_manifest("another-video", m)
        self.assertEqual(result["kind"], "comfyui")
        self.assertEqual(result["workload_kind"], "video")

    def test_fixed_loopback_only(self):
        for url in ["http://localhost:8090", "http://0.0.0.0:8090", "https://127.0.0.1:8090",
                    "http://127.0.0.1", "http://127.0.0.1:0", "http://127.0.0.1:65536",
                    "http://127.0.0.1:8090/path", "http://user:pass@127.0.0.1:8090",
                    "http://127.0.0.1:8090?token=secret", "http://127.0.0.1:08090"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.valid(url=url)
        self.assertEqual(self.valid(url="http://127.0.0.1:8090/")["url"], "http://127.0.0.1:8090")

    def test_id_and_reserved_builtin_protection(self):
        for value in ["h3", "muse", "UPPER", "../model", "1model", "a/b", "a" * 65, "", 42]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_manifest(value, self.manifest)
        self.assertEqual(validate_manifest("model_v2-3", self.manifest)["kind"], "llamacpp")

    def test_required_fields_and_plugin_code_rejected(self):
        for key in ["kind", "root", "exe", "start_argv", "startup_free_ram_gib"]:
            m = copy.deepcopy(self.manifest)
            del m[key]
            with self.assertRaises(ValueError):
                validate_manifest("x", m)
        for change in [{"plugin": "anything.py"}, {"eval": "anything"}, {"kind": "custom"},
                       {"workload_kind": "video"}, {"kind": ["llamacpp"]}]:
            with self.assertRaises(ValueError):
                self.valid(**change)

    def test_existing_absolute_paths_and_cwd(self):
        for change in [{"root": "relative"}, {"exe": "relative.exe"},
                       {"root": str(self.exe)}, {"exe": str(self.root)},
                       {"exe": str(self.root / "missing.exe")}, {"start_cwd": str(self.root.parent)}]:
            with self.assertRaises(ValueError):
                self.valid(**change)

    def test_shells_and_shell_scripts_rejected(self):
        for filename in ["cmd.exe", "powershell.exe", "pwsh.exe", "bash.exe", "sh.exe", "start.cmd", "start.ps1"]:
            exe = self.root / filename
            exe.touch()
            argv = [str(exe), *self.manifest["start_argv"][1:]]
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                self.valid(exe=str(exe), start_argv=argv)

    def test_argv_must_be_direct_exact_executable_list(self):
        other = self.root / "other.exe"
        other.touch()
        for argv in ["llama-server.exe --port 8090", [], [str(self.exe)],
                     [str(other), *self.manifest["start_argv"][1:]],
                     [str(self.exe), "--port", "8090", "--host", "127.0.0.1", "x\x00y"],
                     [str(self.exe), "--port", "8090", "--host", "127.0.0.1", 5]]:
            with self.assertRaises(ValueError):
                self.valid(start_argv=argv)

    def test_argv_port_binding_must_match_and_not_be_ambiguous(self):
        for argv in [[str(self.exe), "--port", "8091", "--host", "127.0.0.1"],
                     [str(self.exe), "--port", "8090", "--host", "0.0.0.0"],
                     [str(self.exe), "--port=8090", "--host", "127.0.0.1"],
                     [str(self.exe), "--port", "8090", "--host", "127.0.0.1", "--port", "9000"],
                     [str(self.exe), "--port", "8090", "--host", "127.0.0.1", "--listen", "0.0.0.0"]]:
            with self.assertRaises(ValueError):
                self.valid(start_argv=argv)

    def test_python_inline_code_and_module_are_rejected(self):
        exe = self.root / "python.exe"
        exe.touch()
        for flag in ("-c", "-m"):
            with self.assertRaises(ValueError):
                self.valid(exe=str(exe), start_argv=[str(exe), flag, "arbitrary", "--host", "127.0.0.1", "--port", "8090"])

    def test_resource_and_context_validation(self):
        for value in [-1, float("inf"), float("nan"), True, "12"]:
            with self.assertRaises(ValueError):
                self.valid(startup_free_ram_gib=value)
        for value in [0, -1, 1048577, True, 2.5, "32768"]:
            with self.assertRaises(ValueError):
                self.valid(context_capacity=value)
        self.assertEqual(self.valid(startup_free_ram_gib=0)["startup_free_ram_gib"], 0)

    def test_key_file_inside_root_never_read(self):
        keyed_argv = [*self.manifest["start_argv"], "--api-key-file", str(self.key)]
        with patch.object(Path, "read_text", side_effect=AssertionError("secret read forbidden")):
            result = self.valid(api_key_file=str(self.key), start_argv=keyed_argv,
                                agent_url="http://127.0.0.1:4096")
        self.assertEqual(result["api_key_file"], str(self.key))
        with tempfile.NamedTemporaryFile() as outside:
            with self.assertRaises(ValueError):
                self.valid(api_key_file=outside.name)
        for url in ["http://127.0.0.1:8090", "http://localhost:4096"]:
            with self.assertRaises(ValueError):
                self.valid(agent_url=url)

    def test_declared_context_exactly_matches_launch(self):
        with self.assertRaises(ValueError):
            self.valid(context_capacity=65536)
        for argv in [[x for x in self.manifest["start_argv"] if x not in ("-c", "32768")],
                     [*self.manifest["start_argv"], "--ctx-size", "32768"],
                     [*self.manifest["start_argv"], "-c", "32768"],
                     ["--ctx-size=32768" if x == "-c" else x for x in self.manifest["start_argv"]]]:
            with self.assertRaises(ValueError):
                self.valid(start_argv=argv)
        argv = ["--ctx-size" if x == "-c" else x for x in self.manifest["start_argv"]]
        self.assertEqual(self.valid(start_argv=argv)["context_capacity"], 32768)

    def test_plaintext_key_rejected_and_key_file_must_match_both_ways(self):
        other_key = self.root / "different-key.txt"
        other_key.touch()
        for extra in [["--api-key", "example-secret"], ["--api-key=example-secret"]]:
            with self.assertRaises(ValueError):
                self.valid(start_argv=[*self.manifest["start_argv"], *extra])
        for changes in [{"api_key_file": str(self.key)},
                        {"start_argv": [*self.manifest["start_argv"], "--api-key-file", str(self.key)]},
                        {"api_key_file": str(self.key), "start_argv": [*self.manifest["start_argv"], "--api-key-file", str(other_key)]},
                        {"api_key_file": str(self.key), "start_argv": [*self.manifest["start_argv"], "--api-key-file", str(self.key), "--api-key-file", str(self.key)]}]:
            with self.assertRaises(ValueError):
                self.valid(**changes)

    def test_explicit_single_slot_required_to_preserve_context_capacity(self):
        base = self.manifest["start_argv"]
        no_slots = [x for x in base if x not in ("-np", "1")]
        for argv in [no_slots, [*no_slots, "--parallel", "2"], [*no_slots, "-np", "0"],
                     [*base, "--parallel", "1"], [*no_slots, "--parallel=1"]]:
            with self.assertRaises(ValueError):
                self.valid(start_argv=argv)
        self.assertEqual(self.valid(start_argv=[*no_slots, "--parallel", "1"])["context_capacity"], 32768)

    def test_register_is_pure_and_replace_explicit(self):
        config = {"models": {"h3": {"kind": "comfyui", "url": "http://127.0.0.1:8188"}}, "other": {"keep": True}}
        original = copy.deepcopy(config)
        result = register_config(config, "another-llm", self.manifest)
        self.assertEqual(config, original)
        self.assertIn("another-llm", result["models"])
        self.assertEqual(result["models"]["h3"], config["models"]["h3"])
        result["other"]["keep"] = False
        self.assertTrue(config["other"]["keep"])
        with self.assertRaises(ValueError):
            register_config(result, "another-llm", self.manifest)
        updated = register_config(result, "another-llm", {**self.manifest, "label": "Updated"}, replace=True)
        self.assertEqual(updated["models"]["another-llm"]["label"], "Updated")
        with self.assertRaises(ValueError):
            register_config(result, "h3", self.manifest, replace=True)

    def test_duplicate_service_url_and_invalid_config_rejected(self):
        config = {"models": {"existing": {"url": "http://127.0.0.1:8090/"}}}
        with self.assertRaises(ValueError):
            register_config(config, "another-llm", self.manifest)
        for config in [{}, {"models": []}, []]:
            with self.assertRaises(ValueError):
                register_config(config, "another-llm", self.manifest)


if __name__ == "__main__":
    unittest.main()
