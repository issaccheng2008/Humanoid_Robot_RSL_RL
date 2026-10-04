import importlib.util
from pathlib import Path
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'tasks/manager_based/humanoid_robot_policy_rsl_rl/mdp/fixed_stick_state.py'
spec=importlib.util.spec_from_file_location('direct_test_state',PATH)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class DirectCurriculumTests(unittest.TestCase):
    def test_user_approved_geometry_pairs(self):
        self.assertEqual(module.DirectCrossingCurriculum.LEVELS,
            ((.10,.08),(.08,.07),(.06,.06),(.04,.05),(.02,.04),(0.,.03)))

    def test_strict_threshold_and_old_episodes_do_not_skip_levels(self):
        c=module.DirectCrossingCurriculum()
        c.record([0]*200,[True]*140+[False]*60)
        self.assertEqual(c.level,0)
        c.record([0],[True])
        self.assertEqual(c.level,1)
        self.assertEqual(len(c.outcomes),0)
        c.record([0]*1000,[True]*1000)
        self.assertEqual(c.level,1)
        for level in range(1,6):
            c.record([level]*200,[True]*200)
        self.assertEqual(c.level,5)
        self.assertEqual(c.parameters,(0.,.03))

    def test_minimum_window_resume_and_frozen_replay(self):
        c=module.DirectCrossingCurriculum()
        c.record([0]*199,[True]*199)
        self.assertEqual(c.level,0)
        restored=module.DirectCrossingCurriculum(saved=c.summary())
        restored.record([0],[True])
        self.assertEqual(restored.level,1)
        frozen=module.DirectCrossingCurriculum(auto_advance=False,saved=restored.summary())
        frozen.record([1]*300,[True]*300)
        self.assertEqual(frozen.level,1)

    def test_geometry_changes_only_on_reset_and_zero_step_starts_crossing(self):
        state=module.FixedStickState(2,'cpu',.02)
        state.curriculum=module.DirectCrossingCurriculum()
        front=torch.zeros(2,2)
        state.reset(torch.arange(2),front)
        torch.testing.assert_close(state.bar_near,torch.tensor([.08,.08]))
        torch.testing.assert_close(state.step_distance,torch.tensor([.10,.10]))
        state.curriculum.level=5
        state.reset(torch.tensor([0]),front)
        torch.testing.assert_close(state.bar_near,torch.tensor([.03,.08]))
        self.assertEqual(state.stage.tolist(),[state.LEAD,state.WALK])
        self.assertEqual(state.command_stage.tolist(),[state.LEAD,state.WALK])
        self.assertEqual(state.episode_level.tolist(),[5,0])
        self.assertTrue(state.crossing_command[0])
        self.assertFalse(state.walk_completed_event[0])
        self.assertEqual(state.expected_foot[0].item(),1)
        self.assertAlmostEqual(state.step_distance[0].item(),.23,places=6)
        contact=torch.ones(2,2,dtype=torch.bool)
        hit=torch.zeros(2,dtype=torch.bool)
        for tick in (1,2):
            contact[0,1]=False
            state.update(tick,front,front-.148,contact,hit)
        contact[0,1]=True
        front[0,1]=.25
        state.update(3,front,front-.148,contact,hit)
        self.assertEqual(state.stage[0].item(),state.FOLLOW)
        contact[0,0]=False
        for tick in (4,5): state.update(tick,front,front-.148,contact,hit)
        contact[0,0]=True
        front[0,0]=.25
        for tick in (6,7): state.update(tick,front,front-.148,contact,hit)
        self.assertTrue(state.clean_success[0])
        self.assertFalse(state.walk_completed[0])

if __name__=='__main__':unittest.main()
