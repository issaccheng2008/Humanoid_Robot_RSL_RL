from pathlib import Path
import importlib.util
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'tasks/manager_based/humanoid_robot_policy_rsl_rl/mdp/fixed_stick_state.py'
spec = importlib.util.spec_from_file_location('stage_state', STATE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class StageStateTests(unittest.TestCase):
    def make(self, collisionless=True):
        state = module.FixedStickState(1, 'cpu', .02, collisionless_mode=collisionless)
        front = torch.zeros(1, 2)
        state.reset(torch.tensor([0]), front)
        return state, front

    def test_hit_is_recorded_without_failure_in_stage1(self):
        state, front = self.make()
        state.update(1, front, front-.148, torch.ones(1,2,dtype=torch.bool), torch.tensor([True]))
        self.assertTrue(state.hit[0])
        self.assertTrue(state.current_hit[0])
        self.assertFalse(state.failed[0])
        state.update(2, front, front-.148, torch.ones(1,2,dtype=torch.bool), torch.tensor([False]))
        self.assertTrue(state.hit[0])
        self.assertFalse(state.current_hit[0])

    def test_dirty_episode_can_complete_but_cannot_be_clean(self):
        state, front = self.make()
        contact = torch.ones(1,2,dtype=torch.bool)
        step = 0
        for foot, target in [(0,.10),(1,.43),(0,.43)]:
            contact[0,foot] = False
            for _ in range(2):
                step += 1
                state.update(step, front, front-.148, contact, torch.tensor([step==1]))
            front[0,foot] = target
            contact[0,foot] = True
            step += 1
            state.update(step, front, front-.148, contact, torch.tensor([False]))
        state.update(step+1, front, front-.148, contact, torch.tensor([False]))
        self.assertTrue(state.success[0])
        self.assertFalse(state.clean_success[0])
        self.assertFalse(state.failed[0])

    def test_stage2_hit_and_stage1_fall_still_fail(self):
        state, front = self.make(False)
        contact = torch.ones(1,2,dtype=torch.bool)
        state.update(1, front, front-.148, contact, torch.tensor([True]))
        self.assertTrue(state.failed[0])
        state, front = self.make(True)
        state.update(1, front, front-.148, contact, torch.tensor([False]), failed=torch.tensor([True]))
        self.assertTrue(state.failed[0])

    def test_entry_is_once_per_foot_and_hovering_has_no_progress_reward(self):
        state, front = self.make()
        state.stage[:] = state.LEAD
        state.expected_foot[:] = 1
        state.crossing_command[:] = True
        front[0,1] = .20
        contact = torch.tensor([[True,False]])
        kwargs = dict(foot_clearance=torch.tensor([[0.,.03]]), foot_velocity=torch.zeros(1,2))
        state.update(1, front, front-.148, contact, torch.tensor([False]), **kwargs)
        self.assertEqual(state.entry_score.item(), 1.)
        self.assertEqual(state.crossing_progress_score.item(), 0.)
        state.update(2, front, front-.148, contact, torch.tensor([False]), **kwargs)
        self.assertEqual(state.entry_score.item(), 0.)
        kwargs['foot_velocity'][0,1] = .5
        state.update(3, front, front-.148, contact, torch.tensor([False]), **kwargs)
        self.assertGreater(state.crossing_progress_score.item(), 0.)
        kwargs['foot_clearance'].zero_()
        state.update(4, front, front-.148, contact, torch.tensor([True]), **kwargs)
        self.assertEqual(state.crossing_progress_score.item(), 0.)

    def test_partial_reset_clears_hit_and_entry_history(self):
        state, front = self.make()
        state.hit[:] = True
        state.entry_seen[:] = True
        state.reset(torch.tensor([0]), front)
        self.assertFalse(state.hit.any())
        self.assertFalse(state.entry_seen.any())



class CleanWindowTests(unittest.TestCase):
    def test_minimum_sample_and_rolling_failures_are_respected(self):
        stats = module.RecentCleanCrossings(.6, 5, 5)
        stats.record([True,True,True])
        self.assertFalse(stats.ready)
        stats.record([False,False])
        self.assertTrue(stats.ready)
        stats.record([False])
        self.assertFalse(stats.ready)
        self.assertAlmostEqual(stats.rate, .4)
        restored = module.RecentCleanCrossings(.6, 5, 5)
        restored.restore(stats.summary())
        self.assertEqual(restored.summary(), stats.summary())

class StageIntegrationTests(unittest.TestCase):
    def setUp(self):
        import sys
        sys.path.insert(0,str(ROOT))
        sys.path.insert(0,str(ROOT/'hpc'))

    def test_default_is_physical_stage2_and_legacy_stage_switch_still_works(self):
        from tasks.manager_based.humanoid_robot_policy_rsl_rl.fixed_stick_env_cfg import FixedStickEnvCfg, configure_fixed_stick_stage
        cfg=FixedStickEnvCfg()
        self.assertTrue(cfg.scene.wooden_bar.spawn.collision_props.collision_enabled)
        self.assertFalse(cfg.collisionless_mode)
        self.assertTrue(cfg.direct_curriculum_enabled)
        configure_fixed_stick_stage(cfg,1)
        self.assertFalse(cfg.scene.wooden_bar.spawn.collision_props.collision_enabled)
        self.assertIsNone(cfg.scene.bar_contacts)
        self.assertTrue(cfg.collisionless_mode)
        configure_fixed_stick_stage(cfg,2)
        self.assertTrue(cfg.scene.wooden_bar.spawn.collision_props.collision_enabled)
        self.assertIsNotNone(cfg.scene.bar_contacts)
        self.assertFalse(cfg.collisionless_mode)
        self.assertEqual(cfg.training_stage,'fixed_stick_stage2_v32')
        self.assertEqual(cfg.rewards.collisionless_hit_penalty.weight,0.)
        configure_fixed_stick_stage(cfg,1)
        self.assertEqual(cfg.rewards.collisionless_hit_penalty.weight,-100.)
        self.assertEqual(cfg.fixed_walk_step,.10)
        self.assertEqual(cfg.fixed_initial_gap,.08)

    def test_resume_rejects_cross_stage_but_direct_stage2_accepts_old_actor(self):
        from fixed_stick_stages import checkpoint_stage, validate_stage_checkpoint
        one={'entropy_schedule_state':{'training_stage':'fixed_stick_stage1_v32'},
             'infos':{'fixed_stick_training':{'ready':True}}}
        self.assertEqual(checkpoint_stage(one),1)
        validate_stage_checkpoint(one,2,False)
        with self.assertRaises(ValueError):
            validate_stage_checkpoint(one,2,True)
        self.assertIsNone(validate_stage_checkpoint({},2,False))
        validate_stage_checkpoint(one,1,True)

    def test_stage1_adapter_detects_geometry_without_a_bar_contact_sensor(self):
        from types import SimpleNamespace as NS
        from unittest.mock import patch, MagicMock
        from tasks.manager_based.humanoid_robot_policy_rsl_rl.mdp import fixed_stick as mdp
        state=module.FixedStickState(1,'cpu',.02,collisionless_mode=True)
        front=torch.tensor([[.20,0.]])
        state.reset(torch.tensor([0]),torch.zeros(1,2))
        state.progress=torch.zeros(1)
        state.previous_root_x=torch.zeros(1)
        state.bar_pose=torch.tensor([[.145,0.,.015,0.,0.,0.,1.]])
        state.feet_cfg=NS(body_ids=[0,1])
        state.sensor_cfg=NS(body_ids=[0,1])
        robot=NS(data=NS(root_pos_w=torch.tensor([[0.,0.,.32]]),
                         projected_gravity_b=torch.tensor([[0.,0.,-1.]]),
                         body_lin_vel_w=torch.zeros(1,2,3)))
        scene=MagicMock()
        scene.__getitem__.return_value=robot
        scene.env_origins=torch.zeros(1,3)
        scene.sensors={'contact_forces':NS(data=NS(current_contact_time=torch.ones(1,2)))}
        env=NS(common_step_counter=1,scene=scene,step_dt=.02,
               episode_length_buf=torch.ones(1),max_episode_length=400,command_manager=MagicMock())
        soles=torch.tensor([[[[.10,-.02,0.],[.20,-.02,0.],[.20,.02,0.],[.10,.02,0.]],
                             [[-.148,-.02,0.],[0.,-.02,0.],[0.,.02,0.],[-.148,.02,0.]]]])
        with patch.object(mdp,'_measure',return_value=(state,soles,front,front-.148)):
            mdp.update_fixed_stick(env)
            self.assertTrue(state.hit[0])
            self.assertFalse(state.failed[0])
            self.assertEqual(mdp.collisionless_hit_penalty(env).item(),1.)
            state.current_hit.zero_()
            self.assertEqual(mdp.collisionless_hit_penalty(env).item(),0.)


class TerminalMetricTests(unittest.TestCase):
    def test_startup_and_manual_reset_do_not_pollute_completed_episode_window(self):
        import sys
        from types import SimpleNamespace as NS
        from unittest.mock import MagicMock, patch
        sys.path.insert(0,str(ROOT))
        from tasks.manager_based.humanoid_robot_policy_rsl_rl.curriculum_aware_env import CurriculumAwareManagerBasedRLEnv as Env
        from isaaclab.envs import ManagerBasedRLEnv
        env=object.__new__(Env)
        env._is_closed=True
        state=module.FixedStickState(2,'cpu',.02,collisionless_mode=True)
        state.initialized=torch.ones(2,dtype=torch.bool)
        state.success[1]=True
        state.walk_completed[1]=True
        state.stage[1]=state.DONE
        state.clean_stats=module.RecentCleanCrossings(.6,1,5)
        env._fixed_stick_state=state
        env.episode_length_buf=torch.tensor([0,0])
        env.extras={'log':{}}
        env.scene=MagicMock()
        env.scene.__getitem__.return_value=NS(data=NS(root_pos_w=torch.zeros(2,3),joint_pos=torch.zeros(2,12)))
        env.termination_manager=NS(active_terms=[])
        ids=torch.tensor([0,1])
        with patch.object(ManagerBasedRLEnv,'_reset_idx'):
            env._reset_idx(ids)
            self.assertEqual(len(state.clean_stats.outcomes),0)
            env.reset_buf=torch.tensor([False,False])
            env.episode_length_buf[:]=10
            env._reset_idx(ids)
            self.assertEqual(len(state.clean_stats.outcomes),0)
            env.reset_buf[1]=True
            env._reset_idx(ids)
            self.assertEqual(list(state.clean_stats.outcomes),[True])
            self.assertEqual(env.extras['log']['Task/clean_crossing_success_rate'],1.)
            state.hit[1]=True
            env._reset_idx(torch.tensor([1]))
            self.assertEqual(list(state.clean_stats.outcomes),[True,False])


class PhysicsHitTests(unittest.TestCase):
    def test_short_substep_penetration_is_retained_without_failure(self):
        state=module.FixedStickState(1,'cpu',.02,collisionless_mode=True)
        front=torch.zeros(1,2)
        state.reset(torch.tensor([0]),front)
        for sole_height in [.04,.029,.04,.04]:
            state.record_physics_hit(torch.tensor([sole_height <= .03]))
        state.update(1,front,front-.148,torch.ones(1,2,dtype=torch.bool),torch.tensor([False]))
        self.assertTrue(state.hit[0])
        self.assertTrue(state.current_hit[0])
        self.assertFalse(state.failed[0])
        self.assertFalse(state.physics_hit.any())
        state.update(2,front,front-.148,torch.ones(1,2,dtype=torch.bool),torch.tensor([False]))
        self.assertTrue(state.hit[0])
        self.assertFalse(state.current_hit[0])


class CleanBonusTests(unittest.TestCase):
    def test_dirty_completion_has_no_final_bonus_but_clean_completion_has_100(self):
        import sys
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        sys.path.insert(0,str(ROOT))
        from tasks.manager_based.humanoid_robot_policy_rsl_rl.mdp import fixed_stick as mdp
        state=NS(success_event=torch.tensor([True,True]),hit=torch.tensor([True,False]))
        with patch.object(mdp,'update_fixed_stick',return_value=state):
            points=100*.02*mdp.fixed_success_reward(NS(step_dt=.02))
        torch.testing.assert_close(points,torch.tensor([0.,100.]))


if __name__ == '__main__':
    unittest.main()
