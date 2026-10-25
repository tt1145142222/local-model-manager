import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import manager
from service import dispatch


class MetadataAndQA(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'metadata.json'

    def write(self, value):
        self.path.write_text(json.dumps(value), encoding='utf-8')
        return self.path.stat().st_mtime_ns

    def test_task_roundtrip_and_nonobject_refused(self):
        task = {'width': 1280, 'height': 720, 'frames': 121, 'precision': 'original'}
        self.write(task)
        self.assertEqual(manager.json_object_file(self.path), task)
        self.write([task])
        with self.assertRaises(ValueError):
            manager.json_object_file(self.path)

    def test_large_metadata_refused_with_bounded_read(self):
        self.path.write_bytes(b' ' * 65537)
        with self.assertRaises(ValueError):
            manager.json_object_file(self.path)

    def test_successful_exit_does_not_claim_quality(self):
        result = manager.qa_result(None, 0, 0)
        self.assertEqual(result['outcome'], 'completed')
        self.assertEqual(result['quality'], 'unknown')
        self.assertIsNone(result['desktop_responsive'])

    def test_fresh_qa_accepted(self):
        stamp = self.write({'quality': 'passed', 'desktop_responsive': True})
        result = manager.qa_result(self.path, stamp, 0)
        self.assertEqual(result['quality'], 'passed')
        self.assertTrue(result['desktop_responsive'])

    def test_stale_qa_cannot_claim_success(self):
        stamp = self.write({'quality': 'passed', 'desktop_responsive': True})
        with contextlib.redirect_stderr(io.StringIO()):
            result = manager.qa_result(self.path, stamp + 1, 0)
        self.assertEqual(result['quality'], 'unknown')
        self.assertIsNone(result['desktop_responsive'])

    def test_malformed_qa_leaves_failed_exit_unchanged(self):
        self.path.write_text('{invalid', encoding='utf-8')
        with contextlib.redirect_stderr(io.StringIO()):
            result = manager.qa_result(self.path, 0, 7)
        self.assertEqual(result['outcome'], 'failed')
        self.assertEqual(result['quality'], 'unknown')

    def test_boolean_feedback_not_inferred_from_strings(self):
        self.write({'quality': 'passed', 'desktop_responsive': 'yes'})
        with contextlib.redirect_stderr(io.StringIO()):
            result = manager.qa_result(self.path, 0, 0)
        self.assertEqual(result['quality'], 'unknown')
        self.assertIsNone(result['desktop_responsive'])

    def test_qa_cannot_override_nonzero_exit_as_completed(self):
        self.write({'outcome': 'completed', 'quality': 'passed'})
        result = manager.qa_result(self.path, 0, 2)
        self.assertEqual(result['outcome'], 'failed')
        self.assertEqual(result['quality'], 'unknown')

    def test_request_cli_transmits_task_and_backward_default(self):
        self.write({'width': 1280, 'height': 720})
        with patch('manager.call', return_value={}) as call, contextlib.redirect_stdout(io.StringIO()):
            manager.main(['request', 'h3', '--owner', 'unit', '--key', 'a', '--task-file', str(self.path)])
            self.assertEqual(call.call_args.args[1]['task']['width'], 1280)
            manager.main(['request', 'muse', '--owner', 'unit', '--key', 'b'])
            self.assertIsNone(call.call_args.args[1]['task'])

    def test_feedback_cli_keeps_credentials_and_boolean_types(self):
        with patch('manager.call', return_value={}) as call, contextlib.redirect_stdout(io.StringIO()):
            manager.main(['feedback', 'lease', 'private', 'work', '--feedback-id', 'qa1',
                          '--quality', 'passed', '--desktop-responsive', 'false'])
        method, params = call.call_args.args
        self.assertEqual(method, 'feedback')
        self.assertEqual(params['token'], 'private')
        self.assertIs(params['desktop_responsive'], False)

    def test_generic_register_cli_serializes_explicit_manifest_without_starting(self):
        manifest = {'kind': 'llamacpp', 'start_argv': ['model.exe', '--port', '8787'],
                    'url': 'http://127.0.0.1:8787', 'context_capacity': 32768}
        self.write(manifest)
        with patch('manager.call', return_value={}) as call, contextlib.redirect_stdout(io.StringIO()):
            manager.main(['register', '--model', 'my-text', '--manifest', str(self.path)])
        call.assert_called_once_with('register_model', {'model': 'my-text', 'manifest': manifest, 'replace': False})

    def test_cli_accepts_generic_registered_model_id(self):
        with patch('manager.call', return_value={}) as call, contextlib.redirect_stdout(io.StringIO()):
            manager.main(['request', 'my-text', '--owner', 'unit', '--key', 'generic'])
        self.assertEqual(call.call_args.args[1]['model'], 'my-text')


class WrapperLifecycle(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.child = MagicMock(pid=456, returncode=0)
        self.child.poll.return_value = 0
        self.args = SimpleNamespace(command=['--', 'python', 'example.py'], model='h3', owner='unit',
                                    keep_loaded=False, wait_seconds=20, task_file=None, qa_file=None)

    def call(self, method, params=None):
        self.calls.append((method, params))
        if method == 'request_model':
            return {'id': 'lease', 'token': 'private'}
        if method == 'lease_status':
            return {'state': 'active', 'proceed': True}
        if method == 'prepare_work':
            return {'proceed': True}
        return {}

    def test_prepare_precedes_child_and_completion_precedes_release(self):
        def spawn(command):
            self.assertEqual(self.calls[-1][0], 'prepare_work')
            self.assertEqual(command, ['python', 'example.py'])
            return self.child
        with patch('manager.call', side_effect=self.call), patch('manager.subprocess.Popen', side_effect=spawn):
            self.assertEqual(manager.run_managed(self.args), 0)
        methods = [method for method, _ in self.calls]
        self.assertEqual(methods[-2:], ['complete_work', 'release'])
        record = self.calls[-2][1]
        self.assertEqual(record['quality'], 'unknown')
        self.assertEqual(record['work_id'], self.calls[2][1]['work_id'])

    def test_prepare_denial_delays_child(self):
        attempts = 0
        def call(method, params=None):
            nonlocal attempts
            result = self.call(method, params)
            if method == 'prepare_work':
                attempts += 1
                return {'proceed': attempts > 1, 'reason': 'pressure'}
            return result
        def spawn(command):
            self.assertEqual(attempts, 2)
            return self.child
        with patch('manager.call', side_effect=call), patch('manager.subprocess.Popen', side_effect=spawn), \
                patch('manager.time.sleep') as sleep, contextlib.redirect_stderr(io.StringIO()):
            manager.run_managed(self.args)
        sleep.assert_called_once_with(5)

    def test_spawn_failure_completes_cancelled_then_releases(self):
        with patch('manager.call', side_effect=self.call), patch('manager.subprocess.Popen', side_effect=OSError('test')):
            with self.assertRaises(OSError):
                manager.run_managed(self.args)
        self.assertEqual(self.calls[-2][1]['outcome'], 'cancelled')
        self.assertEqual(self.calls[-1][0], 'release')

    def test_infeasible_prepare_exits_and_releases_without_launch_or_poll(self):
        def call(method, params=None):
            result = self.call(method, params)
            if method == 'prepare_work':
                return {'proceed': False, 'retryable': False, 'admission_state': 'policy_infeasible',
                        'reason': 'Persistent VRAM requirement exceeds physical capacity'}
            return result
        with patch('manager.call', side_effect=call), patch('manager.subprocess.Popen') as spawn, \
                patch('manager.time.sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'exceeds physical capacity'):
                manager.run_managed(self.args)
        spawn.assert_not_called()
        sleep.assert_not_called()
        self.assertEqual(self.calls[-1][0], 'release')
        self.assertNotIn('complete_work', [method for method, _ in self.calls])

    def test_live_child_keeps_lease_when_registration_errors(self):
        self.child.poll.return_value = None
        def call(method, params=None):
            if method == 'attach_worker':
                raise RuntimeError('test registration failure')
            return self.call(method, params)
        with patch('manager.call', side_effect=call), patch('manager.subprocess.Popen', return_value=self.child):
            with self.assertRaises(RuntimeError):
                manager.run_managed(self.args)
        methods = [method for method, _ in self.calls]
        self.assertNotIn('release', methods)
        self.assertNotIn('complete_work', methods)

    def test_missing_qa_preserves_exit_and_releases(self):
        self.child.returncode = 7
        self.child.poll.return_value = 7
        self.args.qa_file = 'Z:/nonexistent-test-qa.json'
        with patch('manager.call', side_effect=self.call), patch('manager.subprocess.Popen', return_value=self.child), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(manager.run_managed(self.args), 7)
        self.assertEqual(self.calls[-2][1]['outcome'], 'failed')
        self.assertEqual(self.calls[-2][1]['quality'], 'unknown')
        self.assertEqual(self.calls[-1][0], 'release')


class ServiceDispatch(unittest.TestCase):
    def setUp(self):
        self.broker = SimpleNamespace(lock=threading.RLock(), refresh_lock=threading.RLock(), stop_event=threading.Event(), lifecycle_action=None,
                                      control_status=lambda: {'disabled': False},
                                      state={'leases': [], 'cleanup': []}, request_model=MagicMock(return_value={}),
                                      status=MagicMock(return_value={}), prepare_work=MagicMock(return_value={'proceed': True}))

    def test_shutdown_blocks_subsequent_mutation_but_keeps_status_readable(self):
        self.assertEqual(dispatch(self.broker, 'shutdown', {})['state'], 'stopping')
        with self.assertRaises(ValueError):
            dispatch(self.broker, 'request_model', {'model': 'h3'})
        self.broker.request_model.assert_not_called()
        self.assertEqual(dispatch(self.broker, 'status', {}), {})

    def test_active_lease_prevents_shutdown(self):
        self.broker.state['leases'] = [{'state': 'active'}]
        with self.assertRaises(ValueError):
            dispatch(self.broker, 'shutdown', {})
        self.assertFalse(self.broker.stop_event.is_set())

    def test_new_prepare_dispatches_without_leaking_credential_elsewhere(self):
        params = {'lease_id': 'lease', 'token': 'private', 'work_id': 'one', 'task': {'width': 1280}}
        self.assertTrue(dispatch(self.broker, 'prepare_work', params)['proceed'])
        self.broker.prepare_work.assert_called_once_with(**params)


class MCPSerialization(unittest.TestCase):
    def test_tools_transmit_task_and_unknown_qa_defaults(self):
        import mcp_server
        with patch('mcp_server.call', return_value={}) as call:
            asyncio.run(mcp_server.local_models_request('h3', 'unit', 'one', task={'width': 1280}))
            self.assertEqual(call.call_args.args[1]['task'], {'width': 1280})
            asyncio.run(mcp_server.local_models_complete('lease', 'private', 'work'))
            self.assertEqual(call.call_args.args[1]['quality'], 'unknown')
            self.assertIsNone(call.call_args.args[1]['desktop_responsive'])
            asyncio.run(mcp_server.local_models_learning('h3'))
            self.assertEqual(call.call_args.args, ('learning_status', {'model': 'h3'}))

    def test_mcp_generic_registration_does_not_request_a_lease(self):
        import mcp_server
        manifest = {'kind': 'comfyui', 'url': 'http://127.0.0.1:8787'}
        with patch('mcp_server.call', return_value={}) as call:
            asyncio.run(mcp_server.local_models_register('my-video', manifest))
        call.assert_called_once_with('register_model', {'model': 'my-video', 'manifest': manifest, 'replace': False})


if __name__ == '__main__':
    unittest.main()
