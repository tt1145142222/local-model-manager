import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib import error

import client
from broker import Broker
from manager_control import disabled_state, verify_continuation
from service import dispatch, serve
from test_broker_adaptive import FakeAdapter, CONFIG, HEALTHY, TASK


class DisableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root/'work/state'
        self.adapters = {k: FakeAdapter(k) for k in CONFIG['models']}
        self.sample = copy.deepcopy(HEALTHY)
        self.config = copy.deepcopy(CONFIG)
        self.config['app_memory_reclaim'] = {'enabled': True}
        self.b = Broker(self.data, self.config, self.adapters, lambda: copy.deepcopy(self.sample))
        self.addCleanup(lambda: self.b.db.close())
        self.b.refresh()

    def acquire(self):
        lease = self.b.request_model('h3', 'existing-task', 'batch', task=TASK)
        self.b.tick()
        return lease

    def disable(self, lease=None):
        marker = {'disabled': True, 'preserve_lease_ids': [lease['id']] if lease else [],
                  'reason': 'user_explicit_disable_until_explicit_enable'}
        (self.data/'disabled.json').write_text(json.dumps(marker), encoding='utf-8')

    def test_new_request_refused_but_existing_retry_and_renewal_work(self):
        lease = self.acquire()
        self.disable(lease)
        with self.assertRaisesRegex(ValueError, 'explicit user'):
            self.b.request_model('h3', 'new-task', 'new')
        same = self.b.request_model('h3', 'existing-task', 'batch', task=TASK)
        self.assertEqual(same['id'], lease['id'])
        self.assertEqual(same['token'], lease['token'])
        reply = dispatch(self.b, 'lease_status', {'lease_id': lease['id'], 'token': lease['token']})
        self.assertEqual(reply['state'], 'active')
        self.assertGreater(reply['expires'], time.time())

    def test_existing_prepare_and_completion_keep_quality_and_credentials(self):
        lease = self.acquire()
        self.disable(lease)
        cred = {'lease_id': lease['id'], 'token': lease['token']}
        result = dispatch(self.b, 'prepare_work', {**cred, 'work_id': 'one', 'task': TASK})
        self.assertTrue(result['proceed'])
        self.assertFalse(result['quality_parameters_changed'])
        dispatch(self.b, 'complete_work', {**cred, 'work_id': 'one', 'quality': 'unknown'})
        self.assertEqual(self.b.state['works'][-1]['feedback']['quality'], 'unknown')

    def test_unrelated_mutations_and_unknown_lease_refused(self):
        lease = self.acquire()
        self.disable(lease)
        for method, params in [('release_idle', {'model': 'h3'}),
                               ('register_model', {'model': 'new', 'manifest': {}}),
                               ('lease_status', {'lease_id': 'not-preserved', 'token': 'x'})]:
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, 'explicit user'):
                dispatch(self.b, method, params)
        self.assertEqual(self.adapters['h3'].stops, 0)

    def test_disabled_app_context_and_restart_does_not_reenable_reclaimer(self):
        lease = self.acquire()
        self.disable(lease)
        self.assertFalse(self.b.app_memory_context()['allowed'])
        self.b.db.close()
        self.b = Broker(self.data, self.config, self.adapters, lambda: copy.deepcopy(self.sample))
        self.assertFalse(self.b.app_reclaimer.enabled)
        self.assertFalse(self.b.status()['control']['automatic_start_allowed'])

    def test_task_maintenance_is_retained_but_no_idle_model_auto_cleanup(self):
        lease = self.acquire()
        self.disable(lease)
        self.b.tick()  # Original per-task runtime maintenance remains possible.
        self.assertTrue(self.adapters['h3'].runtime_calls)
        dispatch(self.b, 'release', {'lease_id': lease['id'], 'token': lease['token']})
        self.b.tick()
        self.assertTrue(self.b.disabled_tasks_finished())
        self.assertEqual(self.adapters['h3'].stops, 0)
        self.b.tick()
        self.assertEqual(self.adapters['h3'].stops, 0)

    def test_busy_model_and_live_worker_are_not_abandoned_on_release(self):
        lease = self.acquire()
        self.disable(lease)
        self.b.release(lease['id'], lease['token'])
        self.adapters['h3'].current['busy'] = True
        self.b.tick()
        self.assertFalse(self.b.disabled_tasks_finished())
        self.adapters['h3'].current['busy'] = False
        self.b.state['leases'][0]['worker'] = {'pid': 999, 'created': 1, 'exe': 'fake'}
        with patch('broker.same_process', return_value=True):
            self.b.tick()
        self.assertFalse(self.b.disabled_tasks_finished())
        self.assertEqual(self.adapters['h3'].stops, 0)

    def test_monitor_exits_after_existing_task_finishes(self):
        lease = self.acquire()
        self.disable(lease)
        self.b.release(lease['id'], lease['token'])
        self.b.run()
        self.assertTrue(self.b.stop_event.is_set())
        self.assertEqual(self.adapters['h3'].stops, 0)

    def test_disabled_monitor_with_no_tasks_does_not_start_a_model(self):
        self.disable()
        self.b.run()
        self.assertTrue(self.b.stop_event.is_set())
        self.assertTrue(all(a.starts == 0 and a.stops == 0 for a in self.adapters.values()))

    def test_disabled_client_does_not_spawn_but_preserves_running_interface(self):
        self.disable()
        with patch.object(client, 'ROOT', self.root), patch.object(client, 'raw_call', side_effect=error.URLError('offline')), \
                patch.object(client.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'disabled by explicit user'):
                client.ensure_service()
            spawn.assert_not_called()
        with patch.object(client, 'ROOT', self.root), patch.object(client, 'raw_call', return_value={'service': 'local-model-manager'}), \
                patch.object(client.subprocess, 'Popen') as spawn:
            client.ensure_service()
            spawn.assert_not_called()

    def test_service_startup_refused_before_any_model_or_server_initialization(self):
        self.disable()
        with patch('service.Broker') as broker, patch('service.ThreadingHTTPServer') as server:
            with self.assertRaisesRegex(RuntimeError, 'disabled by explicit user'):
                serve(root=self.root)
            broker.assert_not_called()
            server.assert_not_called()

    def test_continuation_is_only_for_preserved_live_leases(self):
        lease = self.acquire()
        self.disable(lease)
        with self.assertRaises(RuntimeError):
            verify_continuation(self.data)
        verify_continuation(self.data, explicit_continuation=True)
        self.b.release(lease['id'], lease['token'])
        self.b.tick()
        with self.assertRaises(RuntimeError):
            verify_continuation(self.data, explicit_continuation=True)

    def test_malformed_disable_marker_never_allows_automatic_start(self):
        for text in ('{broken', '{}', 'false', '{"disabled": false}'):
            with self.subTest(text=text):
                (self.data/'disabled.json').write_text(text, encoding='utf-8')
                self.assertTrue(disabled_state(self.data)['disabled'])
                with self.assertRaises(RuntimeError):
                    verify_continuation(self.data)


if __name__ == '__main__':
    unittest.main()
