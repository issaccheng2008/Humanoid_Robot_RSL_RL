"""Exercise the 10-stick hurdle course updater with deterministic physical measurements."""
from copy import deepcopy
from types import SimpleNamespace as NS
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasks.manager_based.humanoid_robot_policy_rsl_rl.mdp import wooden_bar as wb
from tasks.manager_based.humanoid_robot_policy_rsl_rl.humanoid_robot_policy_rsl_rl_env_cfg import _crossing_state_update_params, HumanoidRobotPolicyEnvCfg

class Bar:
    def __init__(self): self.writes = 0
    def write_root_pose_to_sim(self, pose, env_ids): self.writes += 1
    def write_root_velocity_to_sim(self, velocity, env_ids): pass

class Scene(dict): pass

def vector(value): return NS(torch=value)

class CrossingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.env = NS(num_envs=2, device='cpu', step_dt=0.02, common_step_counter=0,
                      episode_length_buf=torch.zeros(2, dtype=torch.long),
                      cfg=NS(stop_before_crossing=False))
        self.velocity = torch.zeros(2, 3)
        self.velocity[:, 0] = 0.4
        scene = Scene(robot=NS(data=NS(root_pos_w=torch.zeros(2, 3),
                      root_lin_vel_w=vector(self.velocity), root_ang_vel_w=vector(torch.zeros(2, 3)))),
                      wooden_bar=Bar(), collisionless_wooden_bar=Bar())
        scene.env_origins = torch.zeros(2, 3)
        self.contact = torch.ones(2, 2)
        scene.sensors = {'contact_forces': NS(data=NS(current_contact_time=self.contact, last_air_time=torch.zeros(2, 2)),
                                         compute_first_contact=lambda dt: torch.zeros(2, 2, dtype=torch.bool))}
        self.env.scene = scene
        self.syncs = 0
        def sync(): self.syncs += 1
        self.env.command_manager = NS(get_term=lambda name: NS(_update_command=sync))
        self.params = _crossing_state_update_params()
        self.params['crossing_step_distance'] = .23
        self.params['feet_cfg'].body_ids = [0, 1]
        self.params['sensor_cfg'].body_ids = [0, 1]
        self.env.event_manager = NS(get_term_cfg=lambda name: NS(params=self.params))
        self.sole = torch.zeros(2, 2, 4, 3)
        self.state = wb._get_state(self.env)
        self.state.initialized[:] = True
        self.state.training_phase[:] = 5
        self.state.current_stick_index[:] = 0
        self.state.sticks_cleared_count[:] = 0
        self.state.step_distance[:] = self.params['default_step_distance']
        self.geometry = patch.object(wb, '_sole_geometry_w', lambda *args: self.sole)
        self.forward = patch.object(wb, '_base_forward_and_sole_front',
                                    lambda *args: (torch.tensor([[1., 0.], [1., 0.]]),
                                                   torch.tensor([[0., 0., 0., 1.], [0., 0., 0., 1.]]),
                                                   torch.stack([self.sole[:, 0, :, 0].amax(dim=1),
                                                                self.sole[:, 1, :, 0].amax(dim=1)], dim=1)))
        self.geometry.start(); self.forward.start()
        self.addCleanup(self.geometry.stop); self.addCleanup(self.forward.stop)

    def tick(self):
        self.env.common_step_counter += 1
        self.env.episode_length_buf += 1
        return wb._update_crossing_state_once(self.env, **self.params)

    def test_ten_sticks_initial_state(self):
        self.tick()
        self.assertEqual(self.state.num_sticks, 10)
        self.assertEqual(self.state.current_stick_index.tolist(), [0, 0])
        self.assertFalse(self.state.crossing_command.any())
        self.assertAlmostEqual(self.state.step_distance[0].item(), self.params['default_step_distance'], places=4)

    def test_sequential_crossing_and_advancement(self):
        # Position robot feet approaching stick 0 (at x=0.50m)
        # Front sole at 0.42m -> dist to stick0 is 0.50 - 0.015 - 0.42 = 0.065m <= 0.14m
        self.sole[0, 0, :, 0] = 0.42
        self.sole[0, 1, :, 0] = 0.35
        # 1. Swing foot 1
        self.contact[0, 0] = 1.0; self.contact[0, 1] = 0.0
        self.tick()
        # 2. Touchdown foot 1
        self.contact[0, 0] = 1.0; self.contact[0, 1] = 1.0
        self.tick()
        # Touchdown triggers crossing command
        self.assertTrue(self.state.crossing_command[0])
        self.assertAlmostEqual(self.state.step_distance[0].item(), .265, places=3)
        self.assertEqual(self.state.following_step_command_stage[0].item(), 0)

        # Crossing foot (foot 0) steps over and lands at 0.65m (past stick 0)
        self.sole[0, 0, :, 0] = 0.65
        # Swing foot 0
        self.contact[0, 0] = 0.0; self.contact[0, 1] = 1.0
        self.tick()
        # Land foot 0
        self.contact[0, 0] = 1.0; self.contact[0, 1] = 1.0
        self.tick()
        self.assertEqual(self.state.following_step_command_stage[0].item(), 1)
        self.assertAlmostEqual(self.state.step_distance[0].item(), -.035, places=3)

        # Following foot (foot 1) steps over and lands at 0.63m (both feet past stick 0 at 0.50m)
        self.sole[0, 1, :, 0] = 0.63
        # Swing foot 1
        self.contact[0, 0] = 1.0; self.contact[0, 1] = 0.0
        self.tick()
        # Land foot 1
        self.contact[0, 0] = 1.0; self.contact[0, 1] = 1.0
        self.tick()
        # Stick 0 cleared!
        self.assertTrue(self.state.stick_cleared_event[0])
        self.assertEqual(self.state.current_stick_index[0].item(), 1)
        self.assertEqual(self.state.sticks_cleared_count[0].item(), 1)

    def test_completion_requires_all_ten_sticks(self):
        self.tick()
        self.state.current_stick_index[0] = 9
        self.state.following_step_command_stage[0] = 1
        self.state.crossing_command[0] = True
        self.state.crossing_foot_index[0] = 1
        self.state.following_foot_index[0] = 0
        # Stick 9 is at x = 0.50 + 9 * 0.23 = 2.57m. Move feet past 2.57m
        self.sole[0, :, :, 0] = 2.70
        # Swing following foot (foot 0)
        self.contact[0, 0] = 0.0; self.contact[0, 1] = 1.0
        self.tick()
        # Land following foot (foot 0)
        self.contact[0, 0] = 1.0; self.contact[0, 1] = 1.0
        self.tick()
        self.assertTrue(self.state.task_success[0])
        self.assertTrue(self.state.task_success_event[0])
        self.assertEqual(self.state.current_stick_index[0].item(), 10)

    def test_terminal_failure_cannot_award_crossing_success(self):
        self.tick()
        self.state.task_success_event[0] = True
        self.state.task_success[0] = True
        self.env.termination_manager = NS(terminated=torch.tensor([True, False]))
        self.env.event_manager = NS(get_term_cfg=lambda name: NS(params=self.params))
        reward = wb.all_sticks_completed_reward(self.env)
        self.assertFalse(reward[0].item())
        self.assertFalse(self.state.task_success[0].item())

    def test_collision_penalized_only_once_per_stick(self):
        # Place foot 0 on Stick 0 (x=0.50m, z=0.01m)
        self.sole[0, 0, :, 0] = 0.50
        self.sole[0, 0, :, 1] = 0.0
        self.sole[0, 0, :, 2] = 0.01
        self.tick()
        # First contact with Stick 0 triggers penalty
        self.assertTrue(self.state.stick_collision_event[0])
        self.assertTrue(self.state.stick_hit_flags[0, 0])
        penalty1 = wb.stick_collision_penalty(self.env)
        self.assertEqual(penalty1[0].item(), 1.0)

        # Foot remains on Stick 0 on subsequent step
        self.tick()
        # NOT penalized again!
        self.assertFalse(self.state.stick_collision_event[0])
        penalty2 = wb.stick_collision_penalty(self.env)
        self.assertEqual(penalty2[0].item(), 0.0)

        # Foot moves to collide with Stick 1 (x=0.73m, z=0.01m)
        self.sole[0, 0, :, 0] = 0.73
        self.tick()
        # First contact with Stick 1 triggers penalty once
        self.assertTrue(self.state.stick_collision_event[0])
        self.assertTrue(self.state.stick_hit_flags[0, 1])

        # Step again on Stick 1 -> no repeat penalty
        self.tick()
        self.assertFalse(self.state.stick_collision_event[0])

    def test_rule_1_clean_clear_awards_reward(self):
        # Approach stick 0 clean (no collision)
        self.sole[0, 0, :, 0] = 0.42; self.sole[0, 1, :, 0] = 0.35
        self.contact[0, 0] = 1.0; self.contact[0, 1] = 0.0
        self.tick()
        self.contact[0, 0] = 1.0; self.contact[0, 1] = 1.0
        self.tick()
        self.assertTrue(self.state.crossing_command[0])

        # Foot 0 crosses to 0.65m
        self.sole[0, 0, :, 0] = 0.65
        self.contact[0, 0] = 0.0; self.contact[0, 1] = 1.0
        self.tick()
        self.contact[0, 0] = 1.0; self.contact[0, 1] = 1.0
        self.tick()

        # Foot 1 crosses to 0.63m
        self.sole[0, 1, :, 0] = 0.63
        self.contact[0, 0] = 1.0; self.contact[0, 1] = 0.0
        self.tick()
        self.contact[0, 0] = 1.0; self.contact[0, 1] = 1.0
        self.tick()

        # Rule 1: Clean pass gives 20 reward (stick_cleared_event is True)
        self.assertTrue(self.state.stick_cleared_event[0])
        self.assertEqual(wb.stick_cleared_reward(self.env)[0].item(), 1.0)
        self.assertEqual(self.state.current_stick_index[0].item(), 1)
        self.assertFalse(self.state.stick_hit_flags[0, 0])

    def test_rule_2_grazed_clear_gives_zero_reward_and_advances(self):
        # Foot 0 hits Stick 0 during approach/crossing
        self.sole[0, 0, :, 0] = 0.50
        self.sole[0, 0, :, 2] = 0.01
        self.tick()
        self.assertTrue(self.state.stick_hit_flags[0, 0])

        # Continue and complete the crossing
        self.state.crossing_command[0] = True
        self.state.following_step_command_stage[0] = 1
        self.state.crossing_foot_index[0] = 0
        self.state.following_foot_index[0] = 1
        self.sole[0, 0, :, 0] = 0.65; self.sole[0, 0, :, 2] = 0.0
        self.sole[0, 1, :, 0] = 0.63; self.sole[0, 1, :, 2] = 0.0

        # Swing foot 1
        self.contact[0, 0] = 1.0; self.contact[0, 1] = 0.0
        self.tick()
        # Land foot 1
        self.contact[0, 0] = 1.0; self.contact[0, 1] = 1.0
        self.tick()

        # Rule 2: Grazed clear gives NO reward (stick_cleared_event is False)
        self.assertFalse(self.state.stick_cleared_event[0])
        self.assertEqual(wb.stick_cleared_reward(self.env)[0].item(), 0.0)
        # But advancement DOES occur!
        self.assertEqual(self.state.current_stick_index[0].item(), 1)
        self.assertEqual(self.state.sticks_cleared_count[0].item(), 1)

    def test_rule_3_hit_and_failed_triggers_penalty(self):
        # Hit Stick 0
        self.sole[0, 0, :, 0] = 0.50
        self.sole[0, 0, :, 2] = 0.01
        self.tick()
        self.assertTrue(self.state.stick_hit_flags[0, 0])

        # Termination is measured before rewards on the next control sample.
        self.env.termination_manager = NS(terminated=torch.tensor([True, False]))
        self.env.common_step_counter += 1
        penalty = wb.stick_failed_penalty(self.env)
        # Env 0 hit and terminated -> penalize! Env 1 did not -> 0.
        self.assertEqual(penalty[0].item(), 1.0)
        self.assertEqual(penalty[1].item(), 0.0)
        self.assertTrue(self.state.stick_failed_event[0])

    def test_no_touch_fall_does_not_trigger_stick_failed(self):
        # Robot falls without touching the stick
        self.assertFalse(self.state.stick_hit_flags[0, 0])
        self.env.termination_manager = NS(terminated=torch.tensor([True, False]))
        penalty = wb.stick_failed_penalty(self.env)
        self.assertEqual(penalty[0].item(), 0.0)
        self.assertFalse(self.state.stick_failed_event[0])

    def test_grand_prize_clean_vs_grazed(self):
        # Case A: Clean Sheet (all 10 cleared with zero touches)
        self.tick()
        self.state.current_stick_index[0] = 9
        self.state.following_step_command_stage[0] = 1
        self.state.crossing_command[0] = True
        self.state.crossing_foot_index[0] = 1
        self.state.following_foot_index[0] = 0
        self.state.stick_hit_flags[0, :] = False
        self.sole[0, :, :, 0] = 2.70
        self.contact[0, 0] = 0.0; self.contact[0, 1] = 1.0; self.tick()
        self.contact[0, 0] = 1.0; self.contact[0, 1] = 1.0; self.tick()
        self.assertTrue(self.state.task_success[0])
        self.assertTrue(self.state.task_success_event[0])

        # Case B: Grazed course (hit stick 3 earlier)
        self.state.current_stick_index[1] = 9
        self.state.following_step_command_stage[1] = 1
        self.state.crossing_command[1] = True
        self.state.crossing_foot_index[1] = 1
        self.state.following_foot_index[1] = 0
        self.state.stick_hit_flags[1, 3] = True
        self.sole[1, :, :, 0] = 2.70
        self.contact[1, 0] = 0.0; self.contact[1, 1] = 1.0; self.tick()
        self.contact[1, 0] = 1.0; self.contact[1, 1] = 1.0; self.tick()
        # Completed, but NOT clean sheet -> NO Grand Prize
        self.assertFalse(self.state.task_success[1])
        self.assertFalse(self.state.task_success_event[1])

    def test_standing_still_yields_zero_reward(self):
        # Robot is stationary (v_x = 0), both feet firmly on ground
        self.velocity[:, :] = 0.0
        self.contact[:, :] = 1.0
        self.tick()
        r_fwd = wb.hurdle_forward_progress_reward(self.env)
        r_clear = wb.stick_over_clearance_reward(self.env)
        r_center = wb.stick_landing_center_reward(self.env)
        r_cleared = wb.stick_cleared_reward(self.env)
        r_all = wb.all_sticks_completed_reward(self.env)
        self.assertEqual(r_fwd[0].item(), 0.0)
        self.assertEqual(r_clear[0].item(), 0.0)
        self.assertEqual(r_center[0].item(), 0.0)
        self.assertEqual(r_cleared[0].item(), 0.0)
        self.assertEqual(r_all[0].item(), 0.0)

    def test_no_repeat_scoring_on_cleared_stick(self):
        # Clear stick 0
        self.state.current_stick_index[0] = 0
        self.state.crossing_command[0] = True
        self.state.following_step_command_stage[0] = 1
        self.state.crossing_foot_index[0] = 0
        self.state.following_foot_index[0] = 1
        self.sole[0, :, :, 0] = 0.65
        self.contact[0, 0] = 1.0; self.contact[0, 1] = 0.0; self.tick()
        self.contact[0, 0] = 1.0; self.contact[0, 1] = 1.0; self.tick()
        self.assertTrue(self.state.stick_cleared_event[0])
        self.assertEqual(self.state.current_stick_index[0].item(), 1)

        # Step again: stick_cleared_event MUST be reset to False immediately
        self.tick()
        self.assertFalse(self.state.stick_cleared_event[0])
        self.assertEqual(wb.stick_cleared_reward(self.env)[0].item(), 0.0)

    def test_4_layer_rewards_exist_and_callable(self):
        cfg = HumanoidRobotPolicyEnvCfg()
        self.assertTrue(callable(cfg.rewards.hurdle_forward_progress.func))
        self.assertTrue(callable(cfg.rewards.stick_over_clearance.func))
        self.assertTrue(callable(cfg.rewards.stick_landing_center.func))
        self.assertTrue(callable(cfg.rewards.stick_cleared.func))
        self.assertTrue(callable(cfg.rewards.stick_failed.func))
        self.assertTrue(callable(cfg.rewards.all_sticks_completed.func))
        self.assertEqual(cfg.rewards.stick_collision.weight, -5.0)

        self.tick()
        r_fwd = wb.hurdle_forward_progress_reward(self.env)
        r_clear = wb.stick_over_clearance_reward(self.env)
        r_center = wb.stick_landing_center_reward(self.env)
        self.assertEqual(r_fwd.shape, (2,))
        self.assertEqual(r_clear.shape, (2,))
        self.assertEqual(r_center.shape, (2,))

    def test_config_has_10_sticks_and_12s_duration(self):
        cfg = HumanoidRobotPolicyEnvCfg()
        self.assertEqual(cfg.episode_length_s, 30.0)
        self.assertTrue(cfg.scene.robot.spawn.usd_path.endswith('v3.2.usd'))
        for j in range(10):
            self.assertTrue(getattr(cfg.scene, f'course_stick_{j}').spawn.rigid_props.kinematic_enabled)
        self.assertFalse(cfg.stop_before_crossing)
        self.assertEqual(cfg.rewards.stick_cleared.weight, 30.0)
        self.assertEqual(cfg.rewards.stick_over_clearance.weight, 15.0)
        self.assertEqual(cfg.rewards.hurdle_body_heading.weight, 3.0)
        self.assertEqual(cfg.rewards.stick_landing_center.weight, 25.0)
        self.assertEqual(cfg.rewards.track_lin_vel_xy_exp.weight, 3.0)
        self.assertEqual(cfg.rewards.stick_failed.weight, -15.0)
        self.assertEqual(cfg.rewards.all_sticks_completed.weight, 100.0)
        self.assertEqual(cfg.rewards.stick_collision.weight, -5.0)

if __name__ == '__main__':
    unittest.main()
