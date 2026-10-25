"""Regression tests for admission guards, short windows and honest reporting."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from adaptive import AdaptivePolicy
from broker import Broker
from protection import observe_guard
from test_broker_adaptive import FakeAdapter, HEALTHY, TASK


FLOORS = {'ram': 12., 'commit': 12., 'vram': 3.}


def sample(t, ram=40., commit=50., vram=27.):
    return {'time': t, 'ram_available_gib': ram, 'commit_free_gib': commit, 'free_vram_gib': vram}


class GuardTests(unittest.TestCase):
    def test_breach_latches_and_requires_fresh_stable_recovery(self):
        s = observe_guard({}, sample(100, commit=.5), FLOORS)
        self.assertTrue(s['blocked'])
        self.assertEqual(s['breach_episodes']['commit'], 1)
        s = observe_guard(s, sample(101, commit=40), FLOORS)
        self.assertTrue(s['blocked'])
        s = observe_guard(s, sample(110, commit=40), FLOORS)
        self.assertTrue(s['blocked'])
        s = observe_guard(s, sample(111, commit=40), FLOORS)
        self.assertFalse(s['blocked'])

    def test_duplicate_timestamps_do_not_accelerate_recovery(self):
        s = observe_guard({}, sample(100, ram=1), FLOORS)
        s = observe_guard(s, sample(101), FLOORS)
        for _ in range(20):
            s = observe_guard(s, sample(101), FLOORS)
        self.assertTrue(s['blocked'])

    def test_monitoring_gap_does_not_count_as_quiet_time(self):
        s = observe_guard({}, sample(100, commit=1), FLOORS)
        s = observe_guard(s, sample(101), FLOORS)
        s = observe_guard(s, sample(200), FLOORS)
        self.assertTrue(s['blocked'])
        self.assertEqual(s['quiet_since'], 200)

    def test_raw_decline_forecasts_risk_before_floor_crossing(self):
        s = observe_guard({}, sample(100, commit=25), FLOORS)
        s = observe_guard(s, sample(101, commit=23), FLOORS)
        self.assertTrue(s['blocked'])
        self.assertEqual(s['forecast_resources'], ['commit'])
        self.assertEqual(s['breach_episodes']['commit'], 0)

    def test_missing_sensor_blocks_and_does_not_manufacture_breach(self):
        s = observe_guard({}, sample(100, commit=None), FLOORS)
        self.assertTrue(s['blocked'])
        self.assertEqual(s['missing_resources'], ['commit'])
        self.assertEqual(s['breach_episodes']['commit'], 0)

    def test_nonfinite_sensor_cannot_leak_into_json_health_or_admit(self):
        s = observe_guard({}, sample(100, commit=float('nan')), FLOORS)
        self.assertTrue(s['blocked'])
        json.dumps(s, allow_nan=False)

    def test_restarted_guard_restarts_stable_recovery_window(self):
        s = observe_guard({}, sample(100, commit=1), FLOORS)
        s = observe_guard(s, sample(101), FLOORS)
        s.pop('previous')
        s['quiet_since'] = None
        s = observe_guard(s, sample(500), FLOORS)
        self.assertTrue(s['blocked'])
        self.assertEqual(s['quiet_since'],500)

    def test_float_tolerance_matches_admission(self):
        s = observe_guard({}, sample(100, vram=3-1e-13), FLOORS)
        self.assertFalse(s['blocked'])
        self.assertEqual(s['breach_episodes']['vram'], 0)

    def test_breach_episodes_not_sampling_frequency(self):
        s = {}
        for t in range(100, 110):
            s = observe_guard(s, sample(t, commit=1), FLOORS)
        self.assertEqual(s['breach_episodes']['commit'], 1)


class BrokerProtectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.c = json.loads((Path(__file__).parent / 'tests/fixtures/config.json').read_text())
        for name, conf in self.c['models'].items():
            conf['root'] = str(Path(self.tmp.name)/name)
            conf['exe'] = str(Path(self.tmp.name)/name/'fake.exe')
        self.a = {m: FakeAdapter(m) for m in self.c['models']}
        self.s = copy.deepcopy(HEALTHY)
        self.b = Broker(Path(self.tmp.name)/'state', self.c, self.a, lambda:copy.deepcopy(self.s))

    def tearDown(self):
        self.b.db.close()
        self.tmp.cleanup()

    def refresh(self, dt=.5):
        self.s['time'] += dt
        self.b.refresh()

    def acquire(self):
        x = self.b.request_model('h3','unit','one',task=TASK)
        self.b.tick()
        self.assertEqual(self.b.lease_status(x['id'],x['token'])['state'],'active')
        return x

    def prepare(self, x, work='w'):
        return self.b.prepare_work(x['id'],x['token'],work)

    def test_unprofiled_budget_prevents_floor_only_admission(self):
        x = self.acquire()
        self.s['commit_free_gib'] = 25
        self.refresh(30)  # No instantaneous trend needed to reproduce this gap.
        r = self.prepare(x)
        self.assertFalse(r['proceed'])
        self.assertGreaterEqual(r['plan']['required']['commit_gib'],38)
        self.assertEqual(r['plan']['task'],TASK)
        self.assertEqual(len(self.b.state['works']),0)

    def test_running_memory_is_not_charged_fallback_twice(self):
        x = self.acquire()
        self.assertTrue(self.prepare(x)['proceed'])
        self.assertEqual(self.b.plan_for(x)['budget_breakdown']['commit_gib'].get('configured_unprofiled_extra_gib'),None)

    def test_idle_cache_can_be_reclaimed_for_pending_inference_budget(self):
        x = self.acquire()
        self.s['commit_free_gib'] = 25
        self.refresh(30)
        self.assertFalse(self.prepare(x)['proceed'])
        # Apply the scheduling profile, then inspect the next idle action.
        self.b.tick()
        self.b.tick()
        self.assertGreater(len(self.a['h3'].cache_calls),0)
        self.assertEqual(self.a['h3'].stops,0)

    def test_prepare_wakes_idle_monitor_and_uses_half_second_sampling(self):
        self.assertEqual(self.b.sampling_interval(),10)
        x = self.acquire()
        self.assertEqual(self.b.sampling_interval(),1)
        self.b.monitor_wake.clear()
        self.assertTrue(self.prepare(x)['proceed'])
        self.assertTrue(self.b.monitor_wake.is_set())
        self.assertEqual(self.b.sampling_interval(),.5)

    def test_three_second_native_work_has_real_busy_samples_and_peak(self):
        x = self.acquire()
        self.assertTrue(self.prepare(x)['proceed'])
        self.a['h3'].current['busy'] = True
        self.s['ram_available_gib'] -= 2
        for _ in range(6): self.refresh()
        w = self.b.active_work(x)
        self.assertEqual(w['measurement']['busy_samples'],6)
        self.assertEqual(w['measurement']['peak_extra_ram_gib'],2)
        self.a['h3'].current['busy'] = False
        self.b.complete_work(x['id'],x['token'],'w')
        self.assertEqual(w['feedback']['quality'],'unknown')

    def test_idle_backend_window_is_visible_but_never_fake_calibration(self):
        x = self.acquire()
        self.prepare(x)
        self.s['ram_available_gib'] -= 1
        for _ in range(6): self.refresh()
        self.b.complete_work(x['id'],x['token'],'w',quality='passed')
        w = self.b.state['works'][-1]
        self.assertEqual(w['measurement']['busy_samples'],0)
        self.assertGreaterEqual(w['sampling']['window_samples'],6)
        self.assertEqual(w['sampling']['window_peak_extra_ram_gib'],1)
        p = self.b.learning_status()['profiles'][0]['policy']
        self.assertEqual(p['measured_runs'],0)
        self.assertIsNone(p['peak_extra_ram_gib'])

    def test_sampling_gap_invalidates_peak_instead_of_learning_underestimate(self):
        x = self.acquire()
        self.prepare(x)
        self.a['h3'].current['busy'] = True
        for _ in range(3): self.refresh()
        self.refresh(20)
        w = self.b.active_work(x)
        self.assertTrue(w['sampling']['discontinuous'])
        self.assertIsNone(w['measurement']['peak_extra_commit_gib'])

    def test_duplicate_samples_never_increment_busy_count(self):
        x = self.acquire()
        self.prepare(x)
        self.a['h3'].current['busy'] = True
        self.refresh()
        for _ in range(5): self.b.refresh()
        self.assertEqual(self.b.active_work(x)['measurement']['busy_samples'],1)

    def test_guard_blocks_next_work_but_never_stops_executing_model(self):
        x = self.acquire()
        self.prepare(x)
        self.a['h3'].current['busy'] = True
        self.s['commit_free_gib'] = 1
        self.refresh()
        self.assertFalse(self.b.lease_status(x['id'],x['token'])['proceed'])
        self.b.tick()
        self.assertEqual(self.a['h3'].stops,0)
        self.assertTrue(self.a['h3'].current['busy'])

    def test_health_report_is_bounded_and_credential_free(self):
        x = self.acquire()
        self.prepare(x)
        self.refresh()
        self.b.complete_work(x['id'],x['token'],'w')
        h = self.b.health_report()
        self.assertEqual(h['work_window']['runs'],1)
        self.assertEqual(h['work_window']['unobserved_runs'],1)
        self.assertEqual(h['work_window']['quality'],{'unknown':1})
        self.assertNotIn(x['token'],json.dumps(h))
        self.assertNotIn('token',json.dumps(h))

    def test_learning_lists_profiles_outside_recent_twenty(self):
        x = self.acquire()
        self.prepare(x)
        self.b.complete_work(x['id'],x['token'],'w')
        first_key = self.b.state['works'][0]['profile_key']
        # Existing learner state need not occur in the recent-work slice.
        self.b.state['works'] = []
        result = self.b.learning_status('h3')
        self.assertIn(first_key,[p['profile_key'] for p in result['profiles']])
        self.assertEqual(result['recent_work'],[])

    def test_fallback_capacity_conflict_is_nonretryable(self):
        self.s['gpus'][0]['total_gib'] = 20
        x = self.b.request_model('h3','unit','small-device',task=TASK)
        self.b.tick()
        r = self.b.lease_status(x['id'],x['token'])
        self.assertEqual(r['state'],'failed')
        self.assertEqual(self.a['h3'].starts,0)


if __name__ == '__main__': unittest.main()
