import copy
import unittest
from adaptive import AdaptivePolicy
from test_adaptive import CONFIG, H3, telemetry

class StartupGateTests(unittest.TestCase):
    def plan(self, free=24, tier=0, gate=24, startup=True, cold=True, task=None):
        c=copy.deepcopy(CONFIG)
        c['models']['h3']['startup_vram_admission_gib']=gate
        p=AdaptivePolicy(c);p._tier=tier
        return p.plan('h3',task or H3,telemetry(gpus=[{'index':0,'free_gib':free}]),{},cold=cold,
                      fallback={'ram':8,'commit':26,'vram':23.5},startup_admission=startup)
    def test_exact_gate_at_all_pressure_tiers(self):
        for tier in range(3):
            p=self.plan(tier=tier)
            self.assertEqual(p['required']['vram_gib'],24)
            self.assertFalse(any('vram_gib available' in x for x in p['reasons']))
            self.assertTrue(p['quality_preserved'])
    def test_below_gate_waits(self):
        self.assertTrue(any('vram_gib available' in x for x in self.plan(free=23.99)['reasons']))
    def test_first_prepare_uses_same_gate(self):
        self.assertEqual(self.plan(cold=False)['required']['vram_gib'],24)
    def test_other_admission_not_overridden(self):
        self.assertEqual(self.plan(cold=False,startup=False)['required']['vram_gib'],26.5)
    def test_explicit_larger_task_estimate_preserved(self):
        p=self.plan(task={**H3,'estimated_peak_vram_gib':26})
        self.assertEqual(p['required']['vram_gib'],29)
    def test_other_resources_preserved(self):
        p=self.plan()
        self.assertEqual(p['required']['commit_gib'],38)
        self.assertEqual(p['required']['ram_gib'],20)
    def test_invalid_gate_rejected(self):
        for value in [True,-1,2,float('nan')]:
            with self.assertRaises(ValueError):self.plan(gate=value)
if __name__=='__main__':unittest.main()
