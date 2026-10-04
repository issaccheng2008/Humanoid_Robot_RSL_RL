"""First completed swing must issue the cross command without a stride gate."""
import importlib.util
from pathlib import Path
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('first_touchdown_state',ROOT/'tasks/manager_based/humanoid_robot_policy_rsl_rl/mdp/fixed_stick_state.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FirstTouchdownTests(unittest.TestCase):
    def make(self):
        state=module.FixedStickState(1,'cpu',.02,collisionless_mode=True)
        front=torch.zeros(1,2)
        state.reset(torch.tensor([0]),front)
        contact=torch.ones(1,2,dtype=torch.bool)
        state.update(0,front,front-.148,contact,torch.tensor([False]))
        return state,front,contact

    def test_first_swing_landing_switches_even_when_stride_misses_10cm(self):
        for foot in (0,1):
            for actual in (.03,.07,.15):
                with self.subTest(foot=foot,actual=actual):
                    state,front,contact=self.make()
                    near=state.bar_near.clone()
                    contact[0,foot]=False
                    for step in (1,2):
                        state.update(step,front,front-.148,contact,torch.tensor([False]))
                        self.assertFalse(state.crossing_command[0])
                    front[0,foot]=actual
                    contact[0,foot]=True
                    state.update(3,front,front-.148,contact,torch.tensor([False]))
                    self.assertTrue(state.crossing_command[0],'must switch on first landing, regardless of stride accuracy')
                    self.assertEqual(state.stage[0].item(),state.LEAD)
                    self.assertEqual(state.expected_foot[0].item(),1-foot)
                    self.assertAlmostEqual(state.step_distance[0].item(),.23,places=6)
                    torch.testing.assert_close(state.bar_near,near)
                    self.assertAlmostEqual(state.touchdown_error[0,foot].item(),actual-.10,places=6)
                    state.update(3,front,front-.148,contact,torch.tensor([False]))
                    self.assertTrue(state.crossing_command[0])

    def test_initial_gap_is_8cm_and_walk_command_stays_10cm(self):
        state,front,contact=self.make()
        self.assertAlmostEqual(state.bar_near[0].item(),.08,places=6)
        self.assertAlmostEqual(state.step_distance[0].item(),.10,places=6)

if __name__=='__main__':
    unittest.main()
