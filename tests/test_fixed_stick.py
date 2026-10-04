"""The fixed obstacle must work with a moving, distance-blind policy."""
import importlib.util
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / 'tasks/manager_based/humanoid_robot_policy_rsl_rl/mdp/fixed_stick_state.py'


class FixedStickTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(STATE_PATH.is_file(), 'fixed moving-crossing state is missing')
        spec = importlib.util.spec_from_file_location('fixed_stick_state', STATE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.state = module.FixedStickState(2, 'cpu', .02)
        self.front = torch.zeros(2, 2)
        self.rear = self.front - .148
        self.contact = torch.ones(2, 2, dtype=torch.bool)
        self.hit = torch.zeros(2, dtype=torch.bool)
        self.state.reset(torch.tensor([0, 1]), self.front)

    def tick(self, step):
        return self.state.update(step, self.front, self.rear, self.contact, self.hit)

    def touchdown(self, foot, step, front):
        self.contact[0, foot] = False
        self.tick(step)
        self.tick(step + 1)
        self.front[0, foot] = front
        self.rear[0, foot] = front - .148
        self.contact[0, foot] = True
        self.tick(step + 2)

    def begin_crossing(self):
        self.touchdown(0, 1, .10)

    def test_initial_bar_near_edge_is_8cm_from_frontmost_toe(self):
        self.front[1] = torch.tensor([1.1, 1.0])
        self.state.reset(torch.tensor([1]), self.front)
        torch.testing.assert_close(self.state.bar_near, torch.tensor([.08, 1.18]))
        torch.testing.assert_close(self.state.step_distance, torch.tensor([.10, .10]))
        self.assertFalse(self.state.crossing_command.any())

    def test_first_10cm_touchdown_switches_while_walking_and_keeps_bar_fixed(self):
        original = self.state.bar_near.clone()
        self.begin_crossing()
        self.assertEqual(self.state.stage.tolist(), [1, 0])
        self.assertEqual(self.state.expected_foot[0].item(), 1)
        self.assertTrue(self.state.crossing_command[0])
        self.assertAlmostEqual(self.state.step_distance[0].item(), .23, places=5)
        torch.testing.assert_close(self.state.bar_near, original)

    def test_initial_ground_contact_and_brief_contact_flicker_do_not_issue_cross_command(self):
        self.tick(1)
        self.contact[0, 0] = False
        self.tick(2)
        self.front[0, 0] = .03
        self.rear[0, 0] = -.118
        self.contact[0, 0] = True
        self.tick(3)
        self.assertFalse(self.state.crossing_command.any())
        self.assertFalse(self.state.success.any())

    def test_support_transfer_without_double_support_still_issues_crossing(self):
        self.contact[0, 0] = False
        self.tick(1)
        self.tick(2)
        self.front[0, 0] = .10
        self.rear[0, 0] = -.048
        self.contact[0] = torch.tensor([True, False])
        self.tick(3)
        self.assertTrue(self.state.crossing_command[0], 'moving gait must not require a stop or double support')

    def test_same_control_step_cannot_count_touchdown_twice(self):
        self.begin_crossing()
        original = self.state.stage.clone()
        self.tick(3)
        torch.testing.assert_close(self.state.stage, original)
        self.assertEqual(self.state.walk_completed_event[0].item(), True)

    def test_one_foot_crossing_is_not_success(self):
        self.begin_crossing()
        self.touchdown(1, 4, .43)
        self.assertEqual(self.state.stage[0].item(), 2)
        self.assertAlmostEqual(self.state.step_distance[0].item(), 0., places=5)
        self.assertFalse(self.state.success[0])

    def test_full_sole_and_two_support_samples_required(self):
        self.begin_crossing()
        self.touchdown(1, 4, .43)
        self.touchdown(0, 7, .43)
        self.assertFalse(self.state.success[0])
        self.tick(10)
        self.assertTrue(self.state.success[0])
        self.assertTrue(self.state.success_event[0])
        self.tick(11)
        self.assertFalse(self.state.success_event[0])

    def test_front_only_over_bar_or_going_back_does_not_succeed(self):
        self.begin_crossing()
        self.touchdown(1, 4, .43)
        self.rear[0, 0] = self.state.bar_far[0] - .01
        self.front[0, 0] = self.rear[0, 0] + .148
        self.tick(7)
        self.tick(8)
        self.assertFalse(self.state.success[0])

    def test_hit_on_success_sample_has_failure_priority(self):
        self.begin_crossing()
        self.touchdown(1, 4, .43)
        self.touchdown(0, 7, .43)
        self.hit[0] = True
        self.tick(10)
        self.assertTrue(self.state.failed[0])
        self.assertFalse(self.state.success[0])
        self.assertFalse(self.state.success_event[0])

    def test_other_failure_suppresses_success_on_same_sample(self):
        self.begin_crossing()
        self.touchdown(1, 4, .43)
        self.touchdown(0, 7, .43)
        self.state.update(10, self.front, self.rear, self.contact, self.hit,
                          failed=torch.tensor([True, False]))
        self.assertFalse(self.state.success[0])

    def test_partial_reset_does_not_reset_other_environment_or_move_its_bar(self):
        self.begin_crossing()
        near = self.state.bar_near.clone()
        self.front[1] += 2.
        self.state.reset(torch.tensor([1]), self.front)
        self.assertEqual(self.state.stage[0].item(), 1)
        self.assertEqual(self.state.bar_near[0].item(), near[0].item())
        self.assertAlmostEqual(self.state.bar_near[1].item(), 2.08, places=5)

    def test_commands_do_not_track_live_obstacle_distance(self):
        self.begin_crossing()
        command = self.state.step_distance.clone()
        self.front += .01
        self.rear += .01
        self.tick(4)
        torch.testing.assert_close(self.state.step_distance, command)


if __name__ == '__main__':
    unittest.main()


class FixedStickConfigTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'tasks/manager_based/humanoid_robot_policy_rsl_rl/fixed_stick_env_cfg.py'
        self.assertTrue(path.is_file(), 'fixed-stick Isaac environment config is missing')
        import sys
        sys.path.insert(0, str(ROOT))
        from tasks.manager_based.humanoid_robot_policy_rsl_rl.fixed_stick_env_cfg import FixedStickEnvCfg
        from tasks.manager_based.humanoid_robot_policy_rsl_rl.fixed_stick_env_cfg import configure_fixed_stick_stage
        self.cfg = configure_fixed_stick_stage(FixedStickEnvCfg(), 2)

    def test_one_fixed_bar_on_flat_ground_and_exact_old_pose(self):
        cfg = self.cfg
        self.assertEqual(cfg.scene.terrain.terrain_type, 'plane')
        for j in range(10):
            self.assertIsNone(getattr(cfg.scene, f'course_stick_{j}'))
        self.assertIsNone(cfg.scene.collisionless_wooden_bar)
        self.assertTrue(cfg.scene.wooden_bar.spawn.rigid_props.kinematic_enabled)
        self.assertTrue(cfg.scene.wooden_bar.spawn.collision_props.collision_enabled)
        self.assertTrue(cfg.scene.robot.spawn.usd_path.endswith('v3.2/v3.2.usd') or
                        cfg.scene.robot.spawn.usd_path.endswith('v3.2\\v3.2.usd'))
        self.assertEqual(cfg.scene.robot.init_state.pos, (0., 0., .32))
        self.assertEqual(cfg.scene.robot.init_state.joint_pos['r_knee_pitch_joint'], .30)
        self.assertEqual(cfg.events.reset_base.params['pose_range'], {})
        self.assertIsNone(cfg.events.update_crossing_state)
        self.assertIsNone(cfg.curriculum.phase_5_ang_vel_z)

    def test_commands_start_10cm_curriculum_and_nonzero_crossing_velocity(self):
        self.assertEqual(self.cfg.fixed_walk_step, .10)
        self.assertEqual(self.cfg.fixed_initial_gap, .08)
        self.assertFalse(self.cfg.stop_before_crossing)
        self.assertEqual(self.cfg.commands.base_velocity.ranges.lin_vel_x, (.20, .20))
        self.assertIsNone(self.cfg.terminations.stop_failed)
        self.assertIsNone(self.cfg.rewards.hurdle_wrong_landing)
        self.assertIsNone(self.cfg.rewards.stick_landing_center)

    def test_contact_filter_has_one_rigid_body_per_expression(self):
        from pxr import Usd, UsdPhysics
        stage = Usd.Stage.Open(self.cfg.scene.robot.spawn.usd_path)
        links = {str(p.GetPath()).split('/yuerobot/')[-1]
                 for p in Usd.PrimRange(stage.GetDefaultPrim()) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        filters = self.cfg.scene.bar_contacts.filter_prim_paths_expr
        self.assertFalse(any('.*' in p for p in filters), 'PhysX filter cannot match multiple bodies per env')
        self.assertEqual({p.split('/Robot/')[-1] for p in filters}, links)

    def test_tilted_foot_with_grounded_heel_behind_bar_is_not_a_collision(self):
        from tasks.manager_based.humanoid_robot_policy_rsl_rl.mdp import fixed_stick as mdp
        self.assertTrue(hasattr(mdp, 'fixed_soles_hit'), 'exact sole/bar intersection is missing')
        soles = torch.tensor([[[[.18, -.02, .00], [.36, -.02, .18],
                                [.36, .02, .18], [.18, .02, .00]]]])
        bar = torch.tensor([[.245, 0., .015, 0., 0., 0., 1.]])
        self.assertFalse(mdp.fixed_soles_hit(soles, bar).any())
        soles[..., 2] = 0.
        self.assertTrue(mdp.fixed_soles_hit(soles, bar).all())

    def test_brief_collision_inside_control_decimation_is_retained(self):
        from tasks.manager_based.humanoid_robot_policy_rsl_rl.mdp import fixed_stick as mdp
        self.assertFalse(self.cfg.scene.lazy_sensor_update, 'brief contacts need physics-rate sensor updates')
        self.assertGreaterEqual(self.cfg.scene.bar_contacts.history_length, self.cfg.decimation)
        self.assertTrue(hasattr(mdp, 'fixed_bar_force_hit'))
        forces = torch.zeros(2, 4, 1, 13, 3)
        forces[0, 0, 0, 3, 0] = 2.0  # Brief knee hit, zero force at final substep.
        self.assertEqual(mdp.fixed_bar_force_hit(forces).tolist(), [True, False])

    def test_completion_beats_stalling_and_deadline_is_not_bootstrapped(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from tasks.manager_based.humanoid_robot_policy_rsl_rl.mdp import fixed_stick as mdp
        env = SimpleNamespace(step_dt=.02)
        state = SimpleNamespace(success_event=torch.tensor([True]), hit=torch.tensor([False]),
                                walk_completed_event=torch.tensor([True]),
                                lead_completed_event=torch.tensor([True]))
        with patch.object(mdp, 'update_fixed_stick', return_value=state):
            reward = self.cfg.rewards.fixed_success.weight * mdp.fixed_success_reward(env)[0].item() * env.step_dt
            self.assertAlmostEqual(reward, 100., places=4)
            for name, expected in [('fixed_walk_completed', 25.), ('fixed_lead_completed', 30.)]:
                term = getattr(self.cfg.rewards, name)
                self.assertAlmostEqual(term.weight * term.func(env)[0].item() * env.step_dt, expected, places=4)
        passive_rate = (self.cfg.rewards.track_ang_vel_z_exp.weight
                        + self.cfg.rewards.hurdle_body_heading.weight
                        + self.cfg.rewards.ground_contact_flatness.weight)
        discount, delay = .99, 50
        stalled = passive_rate * env.step_dt * (1 - discount**delay) / (1 - discount) + discount**delay * reward
        self.assertGreater(reward, stalled)
        self.assertTrue(self.cfg.is_finite_horizon)
        self.assertIsNone(self.cfg.terminations.time_out)

    def test_original_policy_order_and_default_registration_match(self):
        import gymnasium as gym
        cfg = self.cfg
        self.assertEqual(cfg.observations.policy.step_distance.func.__name__, 'fixed_step_command')
        self.assertEqual(cfg.observations.policy.crossing_command.func.__name__, 'fixed_crossing_command')
        self.assertEqual(cfg.curriculum.policy_observation_shape.params['expected_dim'], 49)
        self.assertTrue(gym.spec('Humanoid-Robot-RSLRL-v0').kwargs['env_cfg_entry_point'].endswith(
            'fixed_stick_env_cfg:FixedStickEnvCfg'))


class RuntimeTests(unittest.TestCase):
    def test_hpc_runtime_mismatch_is_rejected_before_simulation(self):
        path = ROOT / 'hpc/runtime_compat.py'
        self.assertTrue(path.is_file(), 'runtime compatibility check is missing')
        spec = importlib.util.spec_from_file_location('runtime_compat', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runtime = {'python': '3.12.10', 'isaacsim': '6.0.1.0',
                   'isaaclab_framework': '3.0.0', 'rsl_rl': '5.4.1'}
        module.validate_versions(runtime)
        for name, bad in [('python', '3.11.0'), ('isaacsim', '5.1.0'),
                          ('isaaclab_framework', '2.3.0'), ('rsl_rl', '3.1.0')]:
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                module.validate_versions({**runtime, name: bad})
