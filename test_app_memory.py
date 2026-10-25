import copy
import math
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from app_memory import AppMemoryReclaimer, IdleTracker, MIB, WindowsApps
from broker import Broker
from service import dispatch
from test_broker import CONFIG, FakeAdapter, HEALTHY
import manager


def app(pid=77, safe=True):
    return {'exe': 'C:\\Apps\\ChatClient\\app.exe', 'name': 'app.exe', 'safe': safe,
            'reason': 'observing_minimized_app' if safe else 'window_restored',
            'rss': 512*MIB, 'cpu_seconds': 1.0, 'io_bytes': 1000,
            'processes': [{'pid': pid, 'created': 1000.0, 'exe': 'C:\\Apps\\ChatClient\\app.exe', 'rss': 512*MIB}]}


class IdleTests(unittest.TestCase):
    def warm(self, tracker, value=None, end=600):
        value = value or app()
        for t in range(0, end+1, 5):
            rows = tracker.observe([value], t)
        return rows[0]

    def test_minimized_and_quiet_for_ten_minutes_only(self):
        tracker = IdleTracker()
        self.assertFalse(self.warm(tracker, end=595)['eligible'])
        self.assertTrue(tracker.observe([app()], 600)[0]['eligible'])

    def test_active_window_cpu_io_and_pid_reuse_reset_idle(self):
        for change in ('safe', 'cpu', 'io', 'pid', 'created'):
            with self.subTest(change=change):
                tracker = IdleTracker()
                self.warm(tracker)
                row = app()
                if change == 'safe': row['safe'] = False
                if change == 'cpu': row['cpu_seconds'] += .5
                if change == 'io': row['io_bytes'] += 100000
                if change == 'pid': row['processes'][0]['pid'] = 88
                if change == 'created': row['processes'][0]['created'] = 1001
                result = tracker.observe([row], 605)[0]
                self.assertFalse(result['eligible'])
                self.assertEqual(result['idle_seconds'], 0)

    def test_gap_duplicate_and_backward_clock_cannot_count_as_idle(self):
        for next_time in (600, 590, 630):
            tracker = IdleTracker()
            self.warm(tracker)
            self.assertFalse(tracker.observe([app()], next_time)[0]['eligible'])

    def test_removed_apps_are_forgotten_and_small_apps_skipped(self):
        tracker = IdleTracker()
        self.warm(tracker)
        tracker.observe([], 605)
        self.assertEqual(tracker.rows, {})
        row = app()
        row['rss'] = 64*MIB
        self.assertFalse(self.warm(IdleTracker(), row)['eligible'])


class FakeDesktop:
    def __init__(self):
        self.apps = [app()]
        self.ram = 10.0
        self.trimmed = []
        self.before_trim = lambda: None

    def snapshot(self, context):
        return copy.deepcopy(self.apps)

    def available_ram(self):
        return self.ram

    def trim_one(self, identity, permitted):
        self.before_trim()
        if not permitted():
            return None
        self.trimmed.append(identity['pid'])
        return {'pid': identity['pid'], 'before_rss_mib': 512, 'after_rss_mib': 32,
                'working_set_reduction_mib': 480}


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.desktop = FakeDesktop()
        self.context = {'allowed': True, 'required_ram_gib': 20, 'model': 'h3'}
        self.events = []
        self.controller = AppMemoryReclaimer({'app_memory_reclaim': {'enabled': True}},
            lambda: self.context, lambda kind, data: self.events.append((kind, data)),
            self.desktop, clock=lambda: self.now)

    def ticks(self, start=0, end=600):
        for self.now in range(start, end+1, 5):
            self.controller.tick()

    def test_automatic_trim_and_honest_accounting(self):
        self.ticks()
        self.assertEqual(self.desktop.trimmed, [77])
        self.assertEqual(self.events[0][0], 'app_working_set_trimmed')
        event = self.events[0][1]
        self.assertEqual(event['before_available_ram_gib'], event['after_available_ram_gib'])
        self.assertFalse(self.controller.status()['releases_commit_or_vram'])

    def test_no_task_or_no_ram_pressure_never_trims(self):
        for ctx, ram in [({'allowed': False}, 1), (self.context, 40), (self.context, math.nan)]:
            with self.subTest(context=ctx, ram=ram):
                self.context, self.desktop.ram = ctx, ram
                self.ticks()
                self.assertEqual(self.desktop.trimmed, [])

    def test_transient_ram_shortage_is_not_enough(self):
        self.desktop.ram = 30
        self.ticks()
        self.desktop.ram = 10
        self.ticks(605, 615)
        self.assertEqual(self.desktop.trimmed, [])
        self.ticks(620, 620)
        self.assertEqual(self.desktop.trimmed, [77])

    def test_foreground_or_identity_change_at_execution_skips(self):
        for kind in ('foreground', 'pid'):
            self.setUp()
            def change():
                if kind == 'foreground': self.desktop.apps[0]['safe'] = False
                else: self.desktop.apps[0]['processes'][0]['created'] += 1
            self.desktop.before_trim = change
            self.ticks()
            self.assertEqual(self.desktop.trimmed, [])

    def test_cancelled_model_request_is_rechecked_before_trim(self):
        self.desktop.before_trim = lambda: self.context.update(allowed=False)
        self.ticks()
        self.assertEqual(self.desktop.trimmed, [])

    def test_cooldown_blocks_repeated_paging(self):
        self.ticks(end=1800)
        self.assertEqual(self.desktop.trimmed, [77])

    def test_only_four_processes_and_stop_when_ram_recovers(self):
        row = self.desktop.apps[0]
        row['processes'] = [dict(row['processes'][0], pid=p) for p in range(70, 80)]
        self.ticks()
        self.assertEqual(len(self.desktop.trimmed), 4)
        self.setUp()
        original = self.desktop.trim_one
        def once(identity, permitted):
            value = original(identity, permitted)
            self.desktop.ram = 30
            return value
        self.desktop.trim_one = once
        self.desktop.apps[0]['processes'] += [dict(self.desktop.apps[0]['processes'][0], pid=88)]
        self.ticks()
        self.assertEqual(self.desktop.trimmed, [77])

    def test_three_action_cap_and_one_minute_spacing(self):
        self.desktop.apps = []
        for i in range(4):
            row = app(pid=100+i)
            row['exe'] = row['processes'][0]['exe'] = f'C:\\Apps\\App{i}\\app.exe'
            self.desktop.apps.append(row)
        self.ticks(end=900)
        self.assertEqual(len(self.desktop.trimmed), 3)
        self.assertEqual(self.controller.status()['state'], 'episode_limit_reached')


class DesktopSelectionTests(unittest.TestCase):
    def setUp(self):
        self.backend = object.__new__(WindowsApps)
        self.backend.username, self.backend.session, self.backend.windows_root = 'user', 1, 'C:\\Windows'
        self.rows = {1: {'pid': 1, 'ppid': 0, 'name': 'app.exe'},
                     2: {'pid': 2, 'ppid': 1, 'name': 'app.exe'},
                     10: {'pid': 10, 'ppid': 0, 'name': 'ChatGPT.exe'}}
        self.paths = {1: 'C:\\Apps\\Test\\app.exe', 2: 'C:\\Apps\\Test\\app.exe',
                      10: 'C:\\Codex\\ChatGPT.exe'}
        self.win = {1: [True]}
        self.front = 10
        self.backend.windows = lambda: self.win
        self.backend.foreground = lambda: self.front
        self.backend.processes = lambda: copy.deepcopy(self.rows)
        self.backend.session_id = lambda pid: 1
        def process(pid):
            return SimpleNamespace(exe=lambda: self.paths[pid], username=lambda: 'User',
                create_time=lambda: 1000+pid, cpu_times=lambda: SimpleNamespace(user=1, system=0),
                io_counters=lambda: SimpleNamespace(read_bytes=0, write_bytes=0, other_bytes=0),
                memory_info=lambda: SimpleNamespace(rss=256*MIB))
        self.patch = patch('app_memory.psutil.Process', side_effect=process)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_minimized_same_user_family_is_observed(self):
        values = self.backend.snapshot({})
        self.assertEqual(len(values), 1)
        self.assertTrue(values[0]['safe'])
        self.assertEqual(len(values[0]['processes']), 2)

    def test_any_restored_window_or_foreground_helper_protects_whole_family(self):
        self.win[2] = [False]
        self.assertFalse(self.backend.snapshot({})[0]['safe'])
        self.win[2] = [True]
        self.front = 2
        self.assertEqual(self.backend.snapshot({}), [])

    def test_office_parent_protects_its_separate_webview_window(self):
        self.rows[1]['name'], self.paths[1] = 'WINWORD.EXE', 'C:\\Office\\WINWORD.EXE'
        self.rows[2]['name'], self.paths[2] = 'msedgewebview2.exe', 'C:\\WebView\\msedgewebview2.exe'
        self.win = {2: [True]}
        self.assertEqual(self.backend.snapshot({}), [])

    def test_background_runtime_child_blocks_parent_app(self):
        self.rows[2]['name'], self.paths[2] = 'python.exe', 'C:\\Runtime\\python.exe'
        self.assertEqual(self.backend.snapshot({}), [])

    def test_registered_model_or_worker_family_is_protected(self):
        self.assertEqual(self.backend.snapshot({'protected_pids': [1]}), [])
        self.assertEqual(self.backend.snapshot({'protected_roots': ['C:\\Apps\\Test']}), [])

    def test_different_session_or_unknown_identity_skips(self):
        self.backend.session_id = lambda pid: 2
        self.assertEqual(self.backend.snapshot({}), [])
        self.backend.session_id = lambda pid: 1
        self.paths[2] = None
        self.assertEqual(self.backend.snapshot({}), [])


class BrokerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.adapters = {k: FakeAdapter(k) for k in CONFIG['models']}
        self.broker = Broker(self.tmp.name, copy.deepcopy(CONFIG), self.adapters, lambda: copy.deepcopy(HEALTHY))
        self.addCleanup(self.broker.db.close)
        self.broker.refresh()

    def request(self):
        return self.broker.request_model('h3', 'test', 'test')

    def test_no_work_stale_and_cancelled_requests_do_not_reclaim(self):
        self.assertFalse(self.broker.app_memory_context()['allowed'])
        lease = self.request()
        self.assertTrue(self.broker.app_memory_context()['allowed'])
        self.broker.last_sample = time.time()-100
        self.assertFalse(self.broker.app_memory_context()['allowed'])
        self.broker.refresh()
        self.broker.release(lease['id'], lease['token'])
        self.assertFalse(self.broker.app_memory_context()['allowed'])

    def test_ram_budget_is_existing_admission_plan_and_never_mutated(self):
        self.request()
        plan = self.broker.plan_for(self.broker.state['leases'][0], cold=True, admission=True)
        self.assertEqual(self.broker.app_memory_context()['required_ram_gib'], plan['required']['ram_gib'])
        self.assertEqual(self.broker.plan_for(self.broker.state['leases'][0], cold=True, admission=True), plan)

    def test_waiting_on_external_model_is_not_a_reason_to_touch_apps(self):
        self.adapters['muse'].external()
        self.broker.refresh()
        self.request()
        self.assertFalse(self.broker.app_memory_context()['allowed'])

    def test_infeasible_policy_does_not_loop_reclaim(self):
        self.request()
        with patch.object(self.broker, 'plan_for', return_value={'retryable': False}):
            self.assertFalse(self.broker.app_memory_context()['allowed'])

    def test_unknown_model_state_skips_and_stop_gate_applies(self):
        self.request()
        self.broker.models['h3']['state'] = 'unknown'
        self.assertFalse(self.broker.app_memory_context()['allowed'])
        self.broker.refresh()
        self.broker.stop_event.set()
        self.assertFalse(self.broker.app_memory_context()['allowed'])

    def test_status_and_cli_expose_readonly_reclaimer(self):
        self.assertIn('app_memory_reclaim', self.broker.status())
        self.assertIn('enabled', dispatch(self.broker, 'app_memory_status', {}))
        with patch('manager.call', return_value={}) as call, patch('builtins.print'):
            manager.main(['app-memory'])
            call.assert_called_once_with('app_memory_status')


if __name__ == '__main__':
    unittest.main()
