"""Broker integration tests with fake model processes; no live model operations."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest

from broker import Broker
import learner


CONFIG = json.loads((Path(__file__).parent / 'tests/fixtures/config.json').read_text())
TASK = {'width': 1280, 'height': 720, 'frames': 121, 'batch_size': 1, 'workflow_signature': 'test-fixed-quality-v1'}
HEALTHY = {'time': 1000., 'ram_available_gib': 50., 'commit_free_gib': 60.,
           'cpu_percent': 5., 'cpu_count': 32, 'scheduler_delay_ms': 0.,
           'gpus': [{'index': 0, 'free_gib': 28., 'used_gib': 4.}], 'errors': []}


class FakeAdapter:
    def __init__(self, model):
        self.model = model
        self.current = {'state': 'stopped', 'busy': False, 'identity': None}
        self.runtime_calls, self.cache_calls = [], []
        self.starts, self.stops = 0, 0

    def inspect(self):
        return copy.deepcopy(self.current)

    def external(self, busy=False):
        self.current = {'state': 'ready', 'busy': busy, 'cpu_percent': 1., 'rss_gib': 5.,
                        'identity': {'pid': 123 if self.model == 'h3' else 124, 'created': 123., 'exe': self.model}}

    def start(self):
        self.starts += 1
        self.external()
        return copy.deepcopy(self.current['identity'])

    def stop(self, identity):
        if self.current['identity'] != identity or self.current['busy'] is not False:
            raise RuntimeError('stop identity or busy guard')
        self.stops += 1
        self.current = {'state': 'stopped', 'busy': False, 'identity': None}

    def apply_runtime(self, identity, profile, baseline=None):
        if self.current['identity'] != identity:
            raise RuntimeError('runtime identity guard')
        self.runtime_calls.append((copy.deepcopy(identity), copy.deepcopy(profile), copy.deepcopy(baseline)))
        original = baseline or {'identity': copy.deepcopy(identity), 'priority': 32, 'affinity': list(range(32))}
        return {'applied': True, 'baseline': original, 'reserved_logical_cpus': profile['reserve_logical_cpus']}

    def free_idle_cache(self, identity):
        if self.current['identity'] != identity or self.current['busy'] is not False:
            raise RuntimeError('cache identity or busy guard')
        self.cache_calls.append(copy.deepcopy(identity))
        return {'requested': True}


class AdaptiveBrokerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.adapters = {model: FakeAdapter(model) for model in CONFIG['models']}
        self.config = copy.deepcopy(CONFIG)
        # Keep test signature discovery entirely inside the temporary fixture.
        for model, config in self.config['models'].items():
            config['root'] = str(Path(self.tmp.name) / model)
            config['exe'] = str(Path(self.tmp.name) / model / 'fake.exe')
        self.sample = copy.deepcopy(HEALTHY)
        self.b = Broker(Path(self.tmp.name) / 'state', self.config, self.adapters, lambda: copy.deepcopy(self.sample))

    def tearDown(self):
        self.b.db.close()
        self.tmp.cleanup()

    def tick(self):
        self.sample['time'] += 5
        self.b.tick()

    def refresh(self):
        self.sample['time'] += 5
        self.b.refresh()

    def acquire(self, task=TASK, external=False, key='one', keep_loaded=False):
        if external:
            self.adapters['h3'].external()
        lease = self.b.request_model('h3', 'unit', key, task=task, keep_loaded=keep_loaded)
        self.tick()
        state = self.b.lease_status(lease['id'], lease['token'])
        self.assertEqual(state['state'], 'active')
        self.assertTrue(state['proceed'])
        return lease

    def prepare(self, lease, work='work', task=None):
        return self.b.prepare_work(lease['id'], lease['token'], work, task)

    def complete(self, lease, work='work', **kwargs):
        return self.b.complete_work(lease['id'], lease['token'], work, **kwargs)

    def work(self, lease, work='work'):
        return next(w for w in self.b.state['works'] if w['id'] == work and w['lease_id'] == lease['id'])

    def busy_samples(self, count=3):
        self.adapters['h3'].current['busy'] = True
        for _ in range(count):
            self.refresh()

    def test_capacity_conflict_queue_fails_without_start_or_indefinite_wait(self):
        self.sample['gpus'][0]['total_gib'] = 32.
        task = {**TASK, 'estimated_peak_vram_gib': 40.}
        lease = self.b.request_model('h3', 'unit', 'impossible', task=task)
        self.tick()
        state = self.b.lease_status(lease['id'], lease['token'])
        self.assertEqual(state['state'], 'failed')
        self.assertFalse(state['retryable'])
        self.assertIn('capacity', state['reason'])
        self.assertEqual(self.adapters['h3'].starts, 0)

    def test_capacity_conflict_prepare_returns_nonretryable_without_work(self):
        self.sample['gpus'][0]['total_gib'] = 32.
        lease = self.acquire()
        result = self.prepare(lease, task={**TASK, 'estimated_peak_vram_gib': 40.})
        self.assertFalse(result['proceed'])
        self.assertFalse(result['retryable'])
        self.assertEqual(result['admission_state'], 'policy_infeasible')
        self.assertEqual(self.b.state['works'], [])
        self.assertEqual(self.adapters['h3'].stops, 0)

    def test_cpu_pressure_does_not_invent_memory_shortages(self):
        lease = self.acquire()
        self.assertTrue(self.prepare(lease)['proceed'])
        self.sample['cpu_percent'] = 90.
        self.busy_samples()
        self.adapters['h3'].current['busy'] = False
        self.complete(lease, quality='passed')
        work = self.work(lease)
        self.assertTrue(work['measurement']['pressure_observed'])
        self.assertEqual(work['measurement']['pressure_resources'], [])
        learned = learner.get_policy(self.b.state['learning'], work['profile_key'])
        self.assertTrue(learned['disable_warm_cache'])
        for resource in ('ram', 'commit', 'vram'):
            self.assertEqual(learned['reserve_'+resource+'_bias_gib'], 0.)

    def test_ram_pressure_does_not_raise_commit_or_vram_bias(self):
        lease = self.acquire()
        self.assertTrue(self.prepare(lease)['proceed'])
        self.sample['ram_available_gib'] = 10.
        self.busy_samples()
        self.adapters['h3'].current['busy'] = False
        self.complete(lease, quality='passed')
        work = self.work(lease)
        self.assertEqual(work['measurement']['pressure_resources'], ['ram'])
        learned = learner.get_policy(self.b.state['learning'], work['profile_key'])
        self.assertEqual(learned['reserve_ram_bias_gib'], 2.)
        self.assertEqual(learned['reserve_commit_bias_gib'], 0.)
        self.assertEqual(learned['reserve_vram_bias_gib'], 0.)

    def vram_pressure_with_existing_bias(self, free_vram):
        lease = self.acquire()
        self.assertTrue(self.prepare(lease)['proceed'])
        work = self.work(lease)
        prior = copy.deepcopy(work['measurement'])
        prior.update(pressure_observed=True, pressure_resources=['vram'])
        learner.record_run(self.b.state['learning'], work['profile_key'], 'prior-vram-pressure',
                           prior, {'outcome': 'unknown'})
        self.sample['cpu_percent'] = 90.
        self.sample['gpus'][0]['free_gib'] = free_vram
        self.busy_samples()
        plan = self.b.plan_for(lease, task=work['task'])
        self.assertEqual(self.b.load_state['pressure_tier'], 2)
        self.assertEqual(plan['reserves']['vram_gib'], 4.5)
        self.assertEqual(plan['budget_breakdown']['vram_gib']['base_floor_gib'], 3.)
        self.assertTrue(plan['reasons'])
        self.assertFalse(self.b.lease_status(lease['id'], lease['token'])['proceed'])
        self.assertTrue(work['measurement']['pressure_observed'])
        self.adapters['h3'].current['busy'] = False
        self.complete(lease, quality='passed')
        learned = learner.get_policy(self.b.state['learning'], work['profile_key'])
        self.assertTrue(learned['disable_warm_cache'])
        self.assertEqual(learned['reserve_ram_bias_gib'], 0.)
        self.assertEqual(learned['reserve_commit_bias_gib'], 0.)
        return work, learned

    def test_dynamic_reserve_and_prior_bias_do_not_reinforce_vram_learning(self):
        work, learned = self.vram_pressure_with_existing_bias(4.47)
        self.assertEqual(work['measurement']['pressure_resources'], [])
        self.assertEqual(learned['reserve_vram_bias_gib'], .5)

    def test_actual_vram_floor_breach_still_adds_protection(self):
        work, learned = self.vram_pressure_with_existing_bias(2.9)
        self.assertEqual(work['measurement']['pressure_resources'], ['vram'])
        self.assertEqual(learned['reserve_vram_bias_gib'], 1.)

    def test_learning_floor_comparison_shares_admission_float_tolerance(self):
        lease = self.acquire()
        self.assertTrue(self.prepare(lease)['proceed'])
        self.sample['gpus'][0]['free_gib'] = 3. - 1e-13
        self.busy_samples()
        work = self.work(lease)
        self.assertEqual(work['measurement']['pressure_resources'], [])
        self.assertFalse(work['measurement']['pressure_observed'])
        self.assertTrue(self.b.lease_status(lease['id'], lease['token'])['proceed'])
        self.adapters['h3'].current['busy'] = False
        self.complete(lease, quality='passed')
        learned = learner.get_policy(self.b.state['learning'], work['profile_key'])
        self.assertEqual(learned['reserve_vram_bias_gib'], 0.)

    def test_learning_uses_unrounded_configured_floor(self):
        self.b.policy.config['min_free_vram_gib'] = 3.00004
        lease = self.acquire()
        self.assertTrue(self.prepare(lease)['proceed'])
        self.sample['gpus'][0]['free_gib'] = 3.00002
        self.busy_samples()
        work = self.work(lease)
        plan = self.b.plan_for(lease, task=work['task'])
        self.assertEqual(plan['reserves']['vram_gib'], 3.)
        self.assertEqual(plan['budget_breakdown']['vram_gib']['base_floor_gib'], 3.00004)
        self.assertEqual(work['measurement']['pressure_resources'], ['vram'])
        self.assertTrue(work['measurement']['pressure_observed'])
        self.adapters['h3'].current['busy'] = False
        self.complete(lease, quality='passed')
        learned = learner.get_policy(self.b.state['learning'], work['profile_key'])
        self.assertEqual(learned['reserve_vram_bias_gib'], .5)

    def test_prepare_uses_specific_task_budget_and_creates_no_record_when_blocked(self):
        lease = self.acquire()
        task = {**TASK, 'estimated_peak_ram_gib': 45.}
        result = self.prepare(lease, task=task)
        self.assertFalse(result['proceed'])
        self.assertIn('ram', result['reason'])
        self.assertEqual(self.b.state['works'], [])
        self.assertEqual(self.adapters['h3'].stops, 0)

    def test_automatic_busy_window_records_unknown_on_idle(self):
        lease = self.acquire()
        self.busy_samples()
        self.adapters['h3'].current['busy'] = False
        self.refresh()
        work = self.b.state['works'][-1]
        self.assertEqual(work['lease_id'], lease['id'])
        self.assertTrue(work['automatic'])
        self.assertEqual(work['state'], 'recorded')
        self.assertEqual(work['feedback']['outcome'], 'unknown')
        self.assertEqual(work['feedback']['quality'], 'unknown')
        self.assertEqual(work['measurement']['busy_samples'], 3)
        self.assertEqual(learner.get_policy(self.b.state['learning'], work['profile_key'])['measured_runs'], 0)

    def test_manual_completion_and_late_qa_learn_one_measured_work(self):
        lease = self.acquire()
        self.assertTrue(self.prepare(lease)['proceed'])
        self.sample['ram_available_gib'] -= 2
        self.sample['commit_free_gib'] -= 2
        self.sample['gpus'][0].update(free_gib=27., used_gib=5.)
        self.busy_samples()
        self.adapters['h3'].current['busy'] = False
        self.refresh()
        self.assertEqual(self.work(lease)['state'], 'observing')
        self.complete(lease)
        work = self.work(lease)
        policy = learner.get_policy(self.b.state['learning'], work['profile_key'])
        self.assertEqual(policy['runs'], 1)
        self.assertEqual(policy['measured_runs'], 1)
        self.assertEqual(policy['peak_extra_ram_gib'], 2)
        self.assertEqual(policy['peak_extra_vram_gib'], 1)
        result = self.b.feedback(lease['id'], lease['token'], 'work', 'qa-later',
                                 quality='passed', desktop_responsive=True)
        self.assertEqual(result['policy']['runs'], 1)
        self.assertEqual(result['policy']['measured_runs'], 1)
        repeated = self.b.feedback(lease['id'], lease['token'], 'work', 'qa-later',
                                   quality='passed', desktop_responsive=True)
        self.assertTrue(repeated['duplicate'])

    def test_inflight_peak_is_not_charged_twice_as_admission_budget(self):
        task = {**TASK, 'estimated_peak_ram_gib': 20., 'estimated_peak_commit_gib': 20.,
                'estimated_peak_vram_gib': 15.}
        lease = self.acquire(task=task)
        self.assertTrue(self.prepare(lease)['proceed'])
        self.adapters['h3'].current['busy'] = True
        self.sample['ram_available_gib'] = 30.
        self.sample['commit_free_gib'] = 40.
        self.sample['gpus'][0].update(free_gib=13., used_gib=19.)
        # Keep the policy's pressure trend unchanged to isolate double charging.
        self.b.refresh()
        self.assertFalse(self.work(lease)['measurement']['pressure_observed'])
        state = self.b.lease_status(lease['id'], lease['token'])
        self.assertTrue(state['proceed'])

    def test_external_service_never_receives_runtime_or_cache_mutations(self):
        lease = self.acquire(external=True)
        self.sample.update(cpu_percent=90., ram_available_gib=10.)
        for _ in range(3):
            self.tick()
        self.assertEqual(self.adapters['h3'].runtime_calls, [])
        self.assertEqual(self.adapters['h3'].cache_calls, [])
        self.assertEqual(self.adapters['h3'].stops, 0)
        self.assertFalse(self.b.lease_status(lease['id'], lease['token'])['proceed'])

    def test_runtime_baseline_persists_and_identity_replacement_never_adjusted(self):
        lease = self.acquire()
        self.tick()
        original = copy.deepcopy(self.b.state['managed']['h3']['runtime_baseline'])
        self.sample['cpu_percent'] = 90.
        self.tick()
        self.assertEqual(self.adapters['h3'].runtime_calls[-1][2], original)
        calls = len(self.adapters['h3'].runtime_calls)
        self.adapters['h3'].current['identity']['created'] = 999.
        self.tick()
        self.assertEqual(len(self.adapters['h3'].runtime_calls), calls)
        self.assertEqual(self.b.lease_status(lease['id'], lease['token'])['state'], 'failed')

    def test_prepared_work_prevents_idle_cache_release_until_completed(self):
        lease = self.acquire()
        self.assertTrue(self.prepare(lease)['proceed'])
        self.sample['ram_available_gib'] = 10.
        self.tick()
        self.tick()
        self.assertEqual(self.adapters['h3'].cache_calls, [])
        self.complete(lease, outcome='cancelled')
        self.tick()
        self.assertEqual(len(self.adapters['h3'].cache_calls), 1)
        self.assertEqual(self.adapters['h3'].stops, 0)

    def test_completion_rechecks_live_busy_state_instead_of_stale_idle_sample(self):
        lease = self.acquire()
        self.assertTrue(self.prepare(lease)['proceed'])
        self.adapters['h3'].current['busy'] = True
        with self.assertRaises(ValueError):
            self.complete(lease)
        self.assertEqual(self.work(lease)['state'], 'observing')
        self.adapters['h3'].current.update(busy=False, state='unknown')
        with self.assertRaises(ValueError):
            self.complete(lease)
        self.assertEqual(self.work(lease)['state'], 'observing')

    def test_invalid_qa_leaves_work_and_learner_state_unchanged(self):
        lease = self.acquire()
        self.prepare(lease)
        before_work = copy.deepcopy(self.work(lease))
        before_learning = copy.deepcopy(self.b.state['learning'])
        with self.assertRaises(ValueError):
            self.complete(lease, quality='invented')
        self.assertEqual(self.work(lease), before_work)
        self.assertEqual(self.b.state['learning'], before_learning)

    def test_late_failed_qa_is_visible_in_learning_status(self):
        lease = self.acquire()
        self.prepare(lease)
        self.busy_samples(3)
        self.adapters['h3'].current['busy'] = False
        self.refresh()
        self.complete(lease, quality='passed', desktop_responsive=True)
        self.b.feedback(lease['id'], lease['token'], 'work', 'review-failed', quality='failed')
        status = self.b.learning_status('h3')
        row = next(w for w in status['recent_work'] if w['id'] == 'work')
        self.assertEqual(row['feedback']['quality'], 'failed')

    def test_history_bounds_keep_latest_terminal_records_and_live_lease(self):
        lease = self.acquire()
        self.b.state['works'] = [{'id': f'old-{i}', 'lease_id': 'old', 'state': 'recorded'} for i in range(300)]
        self.b.state['leases'] += [{'id': f'old-{i}', 'state': 'released'} for i in range(300)]
        self.b.state['cleanup'] = [{'id': f'old-{i}', 'state': 'succeeded'} for i in range(60)]
        self.refresh()
        self.assertEqual(len(self.b.state['works']), 128)
        self.assertEqual(self.b.state['works'][0]['id'], 'old-172')
        self.assertEqual(len(self.b.state['leases']), 257)
        self.assertTrue(any(x['id'] == lease['id'] for x in self.b.state['leases']))
        self.assertEqual(len(self.b.state['cleanup']), 32)

    def test_failed_model_reconciles_observing_record_without_quality_claim(self):
        lease = self.acquire()
        self.prepare(lease)
        self.busy_samples(1)
        self.adapters['h3'].current = {'state': 'stopped', 'busy': False, 'identity': None}
        self.tick()
        self.refresh()
        self.assertEqual(self.b.lease_status(lease['id'], lease['token'])['state'], 'failed')
        self.assertEqual(self.work(lease)['state'], 'recorded')
        self.assertNotEqual(self.work(lease)['feedback']['outcome'], 'completed')
        self.assertEqual(self.work(lease)['feedback']['quality'], 'unknown')

    def test_same_work_id_in_different_leases_does_not_alias_learning(self):
        first = self.acquire(external=True, keep_loaded=True)
        self.prepare(first, 'same-id')
        self.busy_samples()
        self.adapters['h3'].current['busy'] = False
        self.refresh()
        self.complete(first, 'same-id')
        self.b.release(first['id'], first['token'])
        self.tick()
        second = self.acquire(external=True, key='second', keep_loaded=True)
        self.prepare(second, 'same-id')
        self.busy_samples()
        self.adapters['h3'].current['busy'] = False
        self.refresh()
        self.complete(second, 'same-id')
        key = self.work(second, 'same-id')['profile_key']
        self.assertEqual(self.work(first, 'same-id')['profile_key'], key)
        self.assertEqual(learner.get_policy(self.b.state['learning'], key)['runs'], 2)
        self.b.feedback(first['id'], first['token'], 'same-id', 'first-qa', quality='failed')
        self.b.feedback(second['id'], second['token'], 'same-id', 'second-qa', quality='passed', desktop_responsive=True)
        self.assertEqual(learner.get_policy(self.b.state['learning'], key)['runs'], 2)

    def test_missing_or_nan_busy_measurements_never_teach_zero(self):
        lease = self.acquire()
        self.prepare(lease)
        self.adapters['h3'].current['busy'] = True
        self.sample['ram_available_gib'] = None
        self.sample['commit_free_gib'] = float('nan')
        self.sample['gpus'][0]['free_gib'] = float('nan')
        self.refresh()
        self.sample.update(ram_available_gib=50., commit_free_gib=60.)
        self.sample['gpus'][0]['free_gib'] = 28.
        self.busy_samples(3)
        self.adapters['h3'].current['busy'] = False
        self.refresh()
        self.complete(lease)
        work = self.work(lease)
        self.assertTrue(all(work['measurement']['peak_extra_' + resource + '_gib'] is None
                            for resource in ('ram', 'commit', 'vram')))
        policy = learner.get_policy(self.b.state['learning'], work['profile_key'])
        self.assertEqual(policy['measured_runs'], 0)
        self.assertIsNone(policy['peak_extra_ram_gib'])

    def test_restart_gap_keeps_work_but_disables_peak_calibration(self):
        lease = self.acquire()
        self.prepare(lease)
        self.busy_samples(3)
        self.b.db.close()
        self.b = Broker(Path(self.tmp.name) / 'state', self.config, self.adapters, lambda: copy.deepcopy(self.sample))
        self.busy_samples(3)
        self.adapters['h3'].current['busy'] = False
        self.refresh()
        self.complete(lease, quality='passed', desktop_responsive=True)
        work = self.work(lease)
        self.assertEqual(work['state'], 'recorded')
        self.assertGreaterEqual(work['measurement']['busy_samples'], 6)
        self.assertEqual(learner.get_policy(self.b.state['learning'], work['profile_key'])['measured_runs'], 0)

    def test_concurrent_refresh_never_runs_two_collectors_or_overwrites_newer_sample(self):
        started, release_first, second_started = threading.Event(), threading.Event(), threading.Event()
        sample_calls, errors = [], []

        def sample():
            sample_calls.append(len(sample_calls) + 1)
            number = sample_calls[-1]
            if number == 1:
                started.set()
                if not release_first.wait(2):
                    raise RuntimeError('test timed out')
            return {**copy.deepcopy(HEALTHY), 'time': 1000. + number}

        def refresh(second=False):
            try:
                if second:
                    second_started.set()
                self.b.refresh()
            except Exception as exc:
                errors.append(exc)

        self.b.sample_fn = sample
        first = threading.Thread(target=refresh)
        second = threading.Thread(target=refresh, args=(True,))
        first.start()
        self.assertTrue(started.wait(1))
        second.start()
        self.assertTrue(second_started.wait(1))
        self.assertEqual(sample_calls, [1])
        release_first.set()
        first.join(2)
        second.join(2)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(sample_calls, [1, 2])
        self.assertEqual(self.b.resources['time'], 1002.)

    def test_automatic_observation_with_bad_baseline_records_unknown_safely(self):
        lease = self.acquire()
        self.sample['ram_available_gib'] = float('nan')
        self.refresh()
        self.sample['ram_available_gib'] = 50.
        self.busy_samples(3)
        self.adapters['h3'].current['busy'] = False
        self.refresh()
        work = self.b.state['works'][-1]
        self.assertEqual(work['lease_id'], lease['id'])
        self.assertEqual(work['state'], 'recorded')
        self.assertEqual(work['feedback']['quality'], 'unknown')
        self.assertEqual(learner.get_policy(self.b.state['learning'], work['profile_key'])['measured_runs'], 0)


if __name__ == '__main__':
    unittest.main()
