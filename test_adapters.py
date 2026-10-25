import copy
import json
from pathlib import Path
import tempfile
import time
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from adapters import Adapter

CONFIG = json.loads((Path(__file__).parent / 'tests/fixtures/config.json').read_text())
IDENTITY = {'pid': 123, 'created': 123.0, 'exe': 'test'}
IDLE = {'state': 'ready', 'busy': False, 'identity': IDENTITY}


class AdapterGuards(unittest.TestCase):
    def test_busy_service_never_terminated(self):
        a = Adapter('h3', CONFIG['models']['h3'])
        with patch.object(a, 'inspect', return_value={**IDLE, 'busy': True}), patch('adapters.psutil.Process') as proc:
            with self.assertRaises(RuntimeError):
                a.stop(IDENTITY)
            proc.assert_not_called()

    def test_unknown_service_never_terminated(self):
        a = Adapter('h3', CONFIG['models']['h3'])
        with patch.object(a, 'inspect', return_value={**IDLE, 'state': 'unknown', 'busy': None}), patch('adapters.psutil.Process') as proc:
            with self.assertRaises(RuntimeError):
                a.stop(IDENTITY)
            proc.assert_not_called()

    def test_second_idle_check_catches_new_work(self):
        a = Adapter('h3', CONFIG['models']['h3'])
        with patch.object(a, 'inspect', side_effect=[IDLE, {**IDLE, 'busy': True}]), patch('adapters.psutil.Process') as proc:
            with self.assertRaises(RuntimeError):
                a.stop(IDENTITY)
            proc.assert_not_called()

    def test_process_reuse_never_terminated(self):
        a = Adapter('h3', CONFIG['models']['h3'])
        with patch.object(a, 'inspect', return_value=IDLE), patch('adapters.same_process', return_value=False), patch('adapters.psutil.Process') as proc:
            with self.assertRaises(RuntimeError):
                a.stop(IDENTITY)
            proc.assert_not_called()

    def test_other_identity_never_terminated(self):
        a = Adapter('h3', CONFIG['models']['h3'])
        with patch.object(a, 'inspect', return_value=IDLE), patch('adapters.psutil.Process') as proc:
            with self.assertRaises(RuntimeError):
                a.stop({**IDENTITY, 'created': 999})
            proc.assert_not_called()

    def test_malformed_queue_fails_closed(self):
        a = Adapter('h3', CONFIG['models']['h3'])
        p = MagicMock()
        p.pid = 123
        p.create_time.return_value = 123
        p.exe.return_value = CONFIG['models']['h3']['exe']
        c = MagicMock()
        c.status = 'LISTEN'
        c.laddr.port = 8188
        p.net_connections.return_value = [c]
        with patch.object(a, 'process', return_value=p), patch('adapters.http_json', return_value={'queue_running': []}):
            s = a.inspect()
            self.assertEqual(s['state'], 'unknown')
            self.assertIsNone(s['busy'])

    def test_start_refuses_existing_service(self):
        a = Adapter('muse', CONFIG['models']['muse'])
        with patch.object(a, 'inspect', return_value=IDLE), patch('adapters.subprocess.Popen') as spawn:
            with self.assertRaises(RuntimeError):
                a.start()
            spawn.assert_not_called()


class RuntimeScheduling(unittest.TestCase):
    def setUp(self):
        self.adapter = Adapter('h3', CONFIG['models']['h3'])
        self.values = {'priority': 0x20, 'affinity': [0, 1, 2, 3, 4, 5, 6, 7]}
        self.p = MagicMock()

        def nice(value=None):
            if value is not None:
                self.values['priority'] = value
            return self.values['priority']

        def affinity(value=None):
            if value is not None:
                self.values['affinity'] = list(value)
            return list(self.values['affinity'])

        self.p.nice.side_effect = nice
        self.p.cpu_affinity.side_effect = affinity
        self.stack = __import__('contextlib').ExitStack()
        self.stack.enter_context(patch.object(self.adapter, 'inspect', return_value=IDLE))
        self.stack.enter_context(patch('adapters.same_process', return_value=True))
        self.stack.enter_context(patch('adapters.psutil.Process', return_value=self.p))
        self.addCleanup(self.stack.close)

    def test_apply_and_restore_exact_original_settings(self):
        result = self.adapter.apply_runtime(IDENTITY, {'cpu_priority': 'below_normal', 'reserve_logical_cpus': 2})
        self.assertTrue(result['applied'])
        self.assertEqual(result['priority'], 0x4000)
        self.assertEqual(result['affinity'], [2, 3, 4, 5, 6, 7])
        self.assertEqual(result['baseline']['affinity'], list(range(8)))
        restored = self.adapter.apply_runtime(IDENTITY, {'cpu_priority': 'restore'})
        self.assertEqual(restored['priority'], 0x20)
        self.assertEqual(restored['affinity'], list(range(8)))

    def test_repeated_adjustment_does_not_shrink_again(self):
        first = self.adapter.apply_runtime(IDENTITY, {'reserve_logical_cpus': 2})
        second = self.adapter.apply_runtime(IDENTITY, {'reserve_logical_cpus': 2})
        self.assertEqual(first['affinity'], second['affinity'])
        self.assertEqual(second['reserved_logical_cpus'], 2)

    def test_persisted_baseline_restores_after_adapter_restart(self):
        first = self.adapter.apply_runtime(IDENTITY, {'reserve_logical_cpus': 2})
        fresh = Adapter('h3', CONFIG['models']['h3'])
        with patch.object(fresh, 'inspect', return_value=IDLE):
            result = fresh.apply_runtime(IDENTITY, {'cpu_priority': 'restore'}, first['baseline'])
        self.assertEqual(result['affinity'], list(range(8)))

    def test_never_widens_original_affinity_or_raises_original_low_priority(self):
        self.values.update(priority=0x40, affinity=[4, 6])
        result = self.adapter.apply_runtime(IDENTITY, {'cpu_priority': 'normal', 'reserve_logical_cpus': 99})
        self.assertEqual(result['affinity'], [6])
        self.assertEqual(result['priority'], 0x40)
        result = self.adapter.apply_runtime(IDENTITY, {'reserve_logical_cpus': 0})
        self.assertEqual(result['affinity'], [4, 6])

    def test_new_process_discards_only_cached_old_baseline(self):
        self.adapter.apply_runtime(IDENTITY, {'reserve_logical_cpus': 2})
        identity = {**IDENTITY, 'created': 999}
        self.values['affinity'] = [3, 4]
        with patch.object(self.adapter, 'inspect', return_value={**IDLE, 'identity': identity}):
            result = self.adapter.apply_runtime(identity, {'reserve_logical_cpus': 1})
        self.assertEqual(result['baseline']['affinity'], [3, 4])

    def test_wrong_persisted_baseline_refused(self):
        baseline = {'identity': {**IDENTITY, 'created': 999}, 'priority': 0x20, 'affinity': [0, 1]}
        with self.assertRaises(ValueError):
            self.adapter.apply_runtime(IDENTITY, {'reserve_logical_cpus': 1}, baseline)
        self.assertEqual(self.values['affinity'], list(range(8)))

    def test_quality_fields_and_invalid_reservation_refused(self):
        for profile in ({'context_tokens': 4096}, {'cpu_priority': 'high'},
                        {'reserve_logical_cpus': -1}, {'reserve_logical_cpus': True}):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                self.adapter.apply_runtime(IDENTITY, profile)

    def test_runtime_changes_allowed_during_inference(self):
        with patch.object(self.adapter, 'inspect', return_value={**IDLE, 'busy': True}):
            result = self.adapter.apply_runtime(IDENTITY, {'cpu_priority': 'below_normal'})
        self.assertTrue(result['applied'])

    def test_identity_change_refused(self):
        with patch('adapters.same_process', return_value=False), self.assertRaises(RuntimeError):
            self.adapter.apply_runtime(IDENTITY, {'cpu_priority': 'below_normal'})
        self.p.nice.assert_not_called()

    def test_unknown_health_refused(self):
        with patch.object(self.adapter, 'inspect', return_value={**IDLE, 'state': 'unknown'}), self.assertRaises(RuntimeError):
            self.adapter.apply_runtime(IDENTITY, {'reserve_logical_cpus': 2})
        self.p.nice.assert_not_called()

    def test_partial_failure_rolls_back_priority(self):
        original_affinity = self.p.cpu_affinity.side_effect

        def fail_change(value=None):
            if value is not None and value == [2, 3, 4, 5, 6, 7]:
                raise PermissionError('test affinity denial')
            return original_affinity(value)

        self.p.cpu_affinity.side_effect = fail_change
        with self.assertRaises(PermissionError):
            self.adapter.apply_runtime(IDENTITY, {'cpu_priority': 'below_normal', 'reserve_logical_cpus': 2})
        self.assertEqual(self.values['priority'], 0x20)
        self.assertEqual(self.values['affinity'], list(range(8)))

    def test_readback_mismatch_is_not_success(self):
        self.p.nice.side_effect = lambda value=None: 0x20
        with self.assertRaises(RuntimeError):
            self.adapter.apply_runtime(IDENTITY, {'cpu_priority': 'below_normal'})


class CacheRelease(unittest.TestCase):
    def test_idle_cache_request_has_no_stop_or_quality_change(self):
        a = Adapter('h3', CONFIG['models']['h3'])
        with patch.object(a, 'inspect', return_value=IDLE) as probe, \
                patch('adapters.same_process', return_value=True), \
                patch('adapters.time.sleep'), patch('adapters.http_json') as http, \
                patch('adapters.psutil.Process') as process:
            result = a.free_idle_cache(IDENTITY)
        self.assertEqual(probe.call_count, 2)
        self.assertTrue(result['requested'])
        self.assertNotIn('freed_bytes', result)
        http.assert_called_once_with(CONFIG['models']['h3']['url'] + '/free',
                                     {'unload_models': True, 'free_memory': True}, headers={})
        process.assert_not_called()

    def test_busy_or_unknown_or_changed_process_never_unloads(self):
        for changed in ({**IDLE, 'busy': True}, {**IDLE, 'state': 'unknown'},
                        {**IDLE, 'identity': {**IDENTITY, 'created': 555}}):
            a = Adapter('h3', CONFIG['models']['h3'])
            with self.subTest(state=changed), patch.object(a, 'inspect', side_effect=[IDLE, changed]), \
                    patch('adapters.time.sleep'), patch('adapters.http_json') as http:
                with self.assertRaises(RuntimeError):
                    a.free_idle_cache(IDENTITY)
                http.assert_not_called()

    def test_pid_reuse_never_unloads(self):
        a = Adapter('h3', CONFIG['models']['h3'])
        with patch.object(a, 'inspect', return_value=IDLE), patch('adapters.time.sleep'), \
                patch('adapters.same_process', return_value=False), patch('adapters.http_json') as http:
            with self.assertRaises(RuntimeError):
                a.free_idle_cache(IDENTITY)
            http.assert_not_called()

    def test_llama_cache_release_unsupported(self):
        a = Adapter('muse', CONFIG['models']['muse'])
        with patch('adapters.http_json') as http, self.assertRaises(ValueError):
            a.free_idle_cache(IDENTITY)
        http.assert_not_called()


class ProcessTelemetry(unittest.TestCase):
    def test_delta_cpu_and_io_normalized_and_identity_reset(self):
        a = Adapter('h3', CONFIG['models']['h3'])
        p = MagicMock()
        p.cpu_times.side_effect = [SimpleNamespace(user=10, system=2), SimpleNamespace(user=14, system=2),
                                  SimpleNamespace(user=100, system=2)]
        p.io_counters.side_effect = [SimpleNamespace(read_bytes=100, write_bytes=200),
                                    SimpleNamespace(read_bytes=900, write_bytes=400),
                                    SimpleNamespace(read_bytes=1000, write_bytes=500)]
        with patch('adapters.time.monotonic', side_effect=[1, 3, 5]), patch('adapters.psutil.cpu_count', return_value=8):
            first = a._process_telemetry(p, IDENTITY)
            second = a._process_telemetry(p, IDENTITY)
            changed = a._process_telemetry(p, {**IDENTITY, 'created': 999})
        self.assertIsNone(first['cpu_percent'])
        self.assertEqual(second['cpu_percent'], 25.0)
        self.assertEqual(second['cpu_process_percent'], 200.0)
        self.assertEqual(second['read_bytes_per_sec'], 400)
        self.assertIsNone(changed['cpu_percent'])

    def test_unknown_health_preserves_verified_process_ram_and_private_commit(self):
        a = Adapter('h3', CONFIG['models']['h3'])
        p = MagicMock(pid=123)
        p.create_time.return_value = 123.
        p.exe.return_value = CONFIG['models']['h3']['exe']
        p.memory_info.return_value = SimpleNamespace(rss=2 * 1024**3, private=3 * 1024**3, vms=99 * 1024**3)
        listener = MagicMock(status='LISTEN')
        listener.laddr.port = 8188
        p.net_connections.return_value = [listener]
        with patch.object(a, 'process', return_value=p), patch('adapters.http_json', side_effect=TimeoutError):
            result = a.inspect()
        self.assertEqual(result['state'], 'unknown')
        self.assertIsNone(result['busy'])
        self.assertEqual(result['rss_gib'], 2.)
        self.assertEqual(result['private_commit_gib'], 3.)

    def test_posix_virtual_size_is_never_reported_as_private_commit(self):
        a = Adapter('h3', CONFIG['models']['h3'])
        p = MagicMock(pid=123)
        p.create_time.return_value = 123.
        p.exe.return_value = CONFIG['models']['h3']['exe']
        p.memory_info.return_value = SimpleNamespace(rss=2 * 1024**3, vms=99 * 1024**3)
        listener = MagicMock(status='LISTEN')
        listener.laddr.port = 8188
        p.net_connections.return_value = [listener]
        with patch.object(a, 'process', return_value=p), patch('adapters.os', SimpleNamespace(name='posix', path=os.path)), \
                patch('adapters.http_json', return_value={'queue_running': [], 'queue_pending': []}):
            result = a.inspect()
        self.assertEqual(result['state'], 'ready')
        self.assertIsNone(result['private_commit_gib'])


class RegisteredAdapters(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Windows CI may expose TEMP through an 8.3 alias (RUNNER~1).
        # Real psutil executable paths are resolved; make the mock equally canonical.
        self.config = {'root': self.tmp.name, 'exe': str((Path(self.tmp.name) / 'model.exe').resolve()),
                       'kind': 'llamacpp', 'url': 'http://127.0.0.1:8787'}

    def fake_process(self, args):
        p = MagicMock()
        p.pid = 789
        p.info = {'exe': self.config['exe'], 'cmdline': args}
        p.exe.return_value = self.config['exe']
        p.create_time.return_value = 321.
        listener = MagicMock(status='LISTEN')
        listener.laddr.port = 8787
        p.net_connections.return_value = [listener]
        return p

    def test_generic_comfyui_does_not_require_h3_bootstrap(self):
        self.config['kind'] = 'comfyui'
        a = Adapter('my-video', self.config)
        p = self.fake_process([self.config['exe'], 'main.py', '--port', '8787'])
        with patch('adapters.psutil.process_iter', return_value=[p]):
            self.assertIs(a.process(), p)

    def test_registered_process_marker_must_match(self):
        marker = str(Path(self.tmp.name) / 'main.py')
        self.config.update(kind='comfyui', process_marker=marker)
        a = Adapter('my-video', self.config)
        wrong = self.fake_process([self.config['exe'], 'wrong.py', '--port', '8787'])
        right = self.fake_process([self.config['exe'], marker, '--port', '8787'])
        with patch('adapters.psutil.process_iter', return_value=[wrong]):
            self.assertIsNone(a.process())
        with patch('adapters.psutil.process_iter', return_value=[right]):
            self.assertIs(a.process(), right)

    def test_generic_llama_without_agent_or_key_checks_health_and_slots(self):
        a = Adapter('my-text', self.config)
        p = self.fake_process([self.config['exe'], '--port', '8787'])
        with patch.object(a, 'process', return_value=p), patch('adapters.socket.socket') as sock, \
                patch('adapters.http_json', side_effect=[{'status': 'ok'}, [{'is_processing': False}]]) as http:
            result = a.inspect()
        self.assertEqual(result['state'], 'ready')
        self.assertFalse(result['busy'])
        sock.assert_not_called()
        self.assertEqual(http.call_count, 2)
        self.assertEqual(a.headers(), {})

    def test_registered_key_file_is_used_only_as_runtime_header(self):
        key = Path(self.tmp.name) / 'key.txt'
        key.write_text('synthetic-unit-test-key', encoding='utf-8')
        self.config['api_key_file'] = str(key)
        a = Adapter('my-text', self.config)
        self.assertEqual(a.headers(), {'Authorization': 'Bearer synthetic-unit-test-key'})
        self.assertNotIn('synthetic-unit-test-key', json.dumps(a.c))

    def test_generic_start_preserves_exact_argv_and_does_not_use_shell(self):
        argv = [self.config['exe'], '--port', '8787', '--ctx-size', '32768', '-m', 'original.gguf']
        self.config['start_argv'] = list(argv)
        a = Adapter('my-text', self.config)
        identity = {'pid': 789, 'created': 321., 'exe': self.config['exe']}
        states = [{'state': 'stopped'}, {'state': 'ready', 'identity': identity}]
        with patch.object(a, 'inspect', side_effect=states), patch('adapters.subprocess.Popen', return_value=MagicMock(pid=789)) as spawn, \
                patch('adapters.time.time', return_value=320.):
            self.assertEqual(a.start(), identity)
        self.assertEqual(spawn.call_args.args[0], argv)
        self.assertFalse(spawn.call_args.kwargs.get('shell', False))
        self.assertEqual(a.c['start_argv'], argv)

    def test_generic_model_without_start_argv_is_not_guessed(self):
        a = Adapter('my-text', self.config)
        with patch.object(a, 'inspect', return_value={'state': 'stopped'}), patch('adapters.subprocess.Popen') as spawn:
            with self.assertRaises(ValueError):
                a.start()
        spawn.assert_not_called()

    def test_ready_different_pid_is_retained_without_ownership(self):
        self.config['start_argv'] = [self.config['exe'], '--port', '8787']
        a = Adapter('my-text', self.config)
        identity = {'pid': 999, 'created': time.time() + 1, 'exe': self.config['exe']}
        states = [{'state': 'stopped'}, {'state': 'ready', 'identity': identity}]
        spawned = MagicMock(pid=789)
        with patch.object(a, 'inspect', side_effect=states), patch('adapters.subprocess.Popen', return_value=spawned):
            with self.assertRaisesRegex(RuntimeError, 'does not match this launch'):
                a.start()
        spawned.terminate.assert_not_called()

    def test_ready_old_creation_time_is_not_claimed(self):
        self.config['start_argv'] = [self.config['exe'], '--port', '8787']
        a = Adapter('my-text', self.config)
        identity = {'pid': 789, 'created': 1., 'exe': self.config['exe']}
        states = [{'state': 'stopped'}, {'state': 'ready', 'identity': identity}]
        spawned = MagicMock(pid=789)
        with patch.object(a, 'inspect', side_effect=states), patch('adapters.subprocess.Popen', return_value=spawned):
            with self.assertRaisesRegex(RuntimeError, 'does not match this launch'):
                a.start()
        spawned.terminate.assert_not_called()

    def test_h3_fresh_launcher_receipt_matches_ready_identity(self):
        self.config['kind'] = 'comfyui'
        a = Adapter('h3', self.config)
        identity = {'pid': 789, 'created': 1000.01, 'exe': self.config['exe']}
        logs = Path(self.tmp.name) / 'work/logs'
        logs.mkdir(parents=True)

        def launched(*args, **kwargs):
            (logs / 'comfyui.pid').write_text('789', encoding='ascii')
            (logs / 'comfyui-launch.json').write_text(json.dumps({'pid': 789}), encoding='utf-8')
            # Pin receipt times: real filesystem timestamps can lag wall time
            # by a clock tick; this unit test verifies ownership, not the clock.
            for name in ('comfyui.pid', 'comfyui-launch.json'):
                os.utime(logs/name, (1000.01,1000.01))
            return SimpleNamespace(returncode=0)

        states = [{'state': 'stopped'}, {'state': 'ready', 'identity': identity}]
        with patch.object(a, 'inspect', side_effect=states), patch('adapters.subprocess.run', side_effect=launched), \
                patch('adapters.time.time', return_value=1000.):
            self.assertEqual(a.start(), identity)

    def test_h3_stale_launcher_receipt_is_not_owned(self):
        self.config['kind'] = 'comfyui'
        a = Adapter('h3', self.config)
        logs = Path(self.tmp.name) / 'work/logs'
        logs.mkdir(parents=True)
        for name, text in [('comfyui.pid', '789'), ('comfyui-launch.json', '{"pid":789}')]:
            path = logs / name
            path.write_text(text, encoding='ascii')
            os.utime(path, (1., 1.))
        with patch.object(a, 'inspect', return_value={'state': 'stopped'}), \
                patch('adapters.subprocess.run', return_value=SimpleNamespace(returncode=0)):
            with self.assertRaisesRegex(RuntimeError, 'ownership could not be verified'):
                a.start()


if __name__ == '__main__':
    unittest.main()
