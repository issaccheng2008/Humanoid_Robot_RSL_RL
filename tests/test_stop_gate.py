import importlib.util
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'tasks/manager_based/humanoid_robot_policy_rsl_rl/mdp/stop_gate.py'

class StopGateTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.is_file(), 'stop gate not implemented')
        spec = importlib.util.spec_from_file_location('stop_gate', MODULE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.gate = self.module.StopGate(3, 'cpu', dt=0.02)
        self.update = torch.ones(3, dtype=torch.bool)
        self.contact = torch.ones(3, 2, dtype=torch.bool)
        self.velocity = torch.zeros(3, 3)
        self.yaw = torch.zeros(3)

    def tick(self, trigger=None):
        return self.gate.update(torch.zeros(3, dtype=torch.bool) if trigger is None else trigger,
                                self.update, self.contact, self.velocity, self.yaw)

    def test_requires_full_half_second_after_trigger(self):
        self.tick(torch.tensor([True, False, False]))
        for _ in range(24):
            self.assertFalse(self.tick().any())
        self.assertEqual(self.tick().tolist(), [True, False, False])
        self.assertFalse(self.tick().any())
        self.assertTrue(self.gate.completed[0])

    def test_unstable_sample_resets_continuous_timer(self):
        self.tick(torch.tensor([True, False, False]))
        for _ in range(20): self.tick()
        self.velocity[0, 0] = 0.06
        self.tick()
        self.assertEqual(self.gate.stable_steps[0], 0)
        self.velocity.zero_()
        for _ in range(24): self.assertFalse(self.tick().any())
        self.assertTrue(self.tick()[0])

    def test_requires_both_feet_and_low_yaw(self):
        self.tick(torch.ones(3, dtype=torch.bool))
        self.contact[0, 0] = False
        self.yaw[1] = 0.11
        for _ in range(25): ready = self.tick()
        self.assertEqual(ready.tolist(), [False, False, True])

    def test_timeout_never_grants_crossing(self):
        self.tick(torch.tensor([True, False, False]))
        self.velocity[0, 0] = 0.2
        for _ in range(150): self.assertFalse(self.tick().any())
        self.assertTrue(self.gate.failed[0])
        self.velocity.zero_()
        for _ in range(30): self.assertFalse(self.tick().any())

    def test_partial_reset_does_not_change_other_environments(self):
        self.tick(torch.ones(3, dtype=torch.bool))
        for _ in range(10): self.tick()
        self.gate.reset(torch.tensor([1]))
        self.assertEqual(self.gate.active.tolist(), [True, False, True])
        self.assertEqual(self.gate.stable_steps.tolist(), [10, 0, 10])

    def test_non_updated_environment_does_not_accumulate_time(self):
        self.tick(torch.ones(3, dtype=torch.bool))
        self.update[1] = False
        for _ in range(25): self.tick()
        self.assertEqual(self.gate.completed.tolist(), [True, False, True])
        self.assertEqual(self.gate.stable_steps[1], 0)

    def test_completed_gate_cannot_be_triggered_again(self):
        for _ in range(30): self.tick(torch.ones(3, dtype=torch.bool))
        self.assertTrue(self.gate.completed.all())
        self.assertFalse(self.gate.active.any())

if __name__ == '__main__': unittest.main()
