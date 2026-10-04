import importlib.util
import ast
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deployment.phase_clock import PhaseClockConfig, PhaseClock
from deployment.policy_interface import PolicyController, DEFAULT_JOINT_POS, build_observation, projected_gravity

spec = importlib.util.spec_from_file_location('clock_test_state', ROOT / 'tasks/manager_based/humanoid_robot_policy_rsl_rl/mdp/fixed_stick_state.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ClockTests(unittest.TestCase):
    def test_autoreset_velocity_is_walk_before_sole_measurement(self):
        # Isaac updates commands before the observation term performs reset FK.
        from types import SimpleNamespace
        path = ROOT/'tasks/manager_based/humanoid_robot_policy_rsl_rl/mdp/fixed_stick.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FixedStickVelocityCommand')
        class Base:
            def _update_command(self):
                pass
        ns = {'UniformVelocityCommand':Base, 'torch':torch}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), ns)
        obj = ns['FixedStickVelocityCommand']()
        state = SimpleNamespace(forward_command=torch.tensor([0., .2]),
                                pending_reset=torch.tensor([True,False]))
        obj._env = SimpleNamespace(_fixed_stick_state=state)
        obj.cfg = SimpleNamespace(ranges=SimpleNamespace(lin_vel_x=(.2,.2)))
        obj.is_standing_env = torch.zeros(2,dtype=torch.bool)
        obj.vel_command_b = torch.zeros(2,3)
        obj._update_command()
        torch.testing.assert_close(obj.vel_command_b[:,0], torch.tensor([.2,.2]))
        state.pending_reset[:] = False
        obj._update_command()
        torch.testing.assert_close(obj.vel_command_b[:,0], torch.tensor([0.,.2]))

    def test_boundaries_and_same_vector_schedule(self):
        cfg = PhaseClockConfig()
        ticks = [0, 6, 7, 21, 22, 37, 38, 100]
        expected = [0, 0, 1, 1, 2, 2, 3, 3]
        self.assertEqual([cfg.phase_at_tick(t) for t in ticks], expected)
        self.assertEqual(cfg.phase_at_tick(torch.tensor(ticks)).tolist(), expected)
        self.assertEqual(cfg.command_at_tick(7), (.2, 0., .23, 1.))
        self.assertEqual(cfg.command_at_tick(38), (0., 0., 0., 0.))

    def test_start_latched_until_explicit_reset(self):
        clock = PhaseClock()
        self.assertFalse(clock.read(10.).active)
        self.assertTrue(clock.start(10.))
        self.assertFalse(clock.start(10.1))
        self.assertEqual(clock.read(10.14).phase, 1)
        end = clock.read(10.76)
        self.assertTrue(end.sequence_finished)
        self.assertFalse(end.active)
        self.assertFalse(clock.start(12.))
        clock.reset()
        self.assertTrue(clock.start(12.))

    def test_large_monotonic_origin_does_not_delay_boundaries(self):
        for origin in (1048576., 2097152.):
            clock = PhaseClock()
            clock.start(origin)
            for seconds, phase in ((.12,0), (.14,1), (.42,1), (.44,2), (.74,2), (.76,3)):
                self.assertEqual(clock.read(origin+seconds).phase, phase)

    def test_invalid_times_rejected(self):
        for args in ({'walk_end_s': float('nan')}, {'lead_end_s': .1},
                     {'walk_end_s': .011, 'lead_end_s': .019}):
            with self.assertRaises(ValueError):
                PhaseClockConfig(**args)
        clock = PhaseClock()
        clock.start(10.)
        clock.read(10.2)
        with self.assertRaises(ValueError):
            clock.read(10.1)

    def test_closed_loop_contract_and_no_action_after_clock_end(self):
        captured = []
        def infer(obs):
            captured.append(obs.copy())
            return np.arange(12, dtype=np.float32)[None, :] / 10
        controller = PolicyController(infer)
        controller.start(10.)
        sensors = dict(acc=[0,0,9.81], gyro=[.1,.2,.3], gravity=[0,0,-1],
                       q=DEFAULT_JOINT_POS, qd=np.zeros(12))
        first = controller.tick(now=10., **sensors)
        np.testing.assert_allclose(first.joint_targets, DEFAULT_JOINT_POS + .25 * first.action)
        np.testing.assert_allclose(captured[0][0, 9:13], [.2,0,.1,0])
        np.testing.assert_allclose(captured[0][0, 37:49], 0.)
        controller.tick(now=10.14, **sensors)
        np.testing.assert_allclose(captured[1][0, 9:13], [.2,0,.23,1])
        np.testing.assert_allclose(captured[1][0, 37:49], first.action)
        end = controller.tick(now=10.76, **sensors)
        self.assertTrue(end.sequence_finished)
        self.assertIsNone(end.joint_targets)
        self.assertEqual(len(captured), 2)
        np.testing.assert_allclose(projected_gravity([0,0,0,1]), [0,0,-1])
        np.testing.assert_allclose(captured[0][0, 0:3], [0,0,.981])
        with self.assertRaises(ValueError):
            build_observation([float('nan')]*3, sensors['gyro'], sensors['gravity'],
                              (.2,0,.1,0), sensors['q'], sensors['qd'], np.zeros(12))


class ClockStateTests(unittest.TestCase):
    def setUp(self):
        self.state = module.FixedStickState(2, 'cpu', .02, command_mode='phase_clock', collisionless_mode=True)
        self.front = torch.zeros(2,2)
        self.rear = self.front - .148
        self.contact = torch.ones(2,2,dtype=torch.bool)
        self.hit = torch.zeros(2,dtype=torch.bool)
        self.state.reset(torch.arange(2), self.front, step=0)

    def tick(self, step):
        self.state.update(step, self.front, self.rear, self.contact, self.hit)

    def test_commands_do_not_depend_on_touchdown_or_bar_geometry(self):
        self.contact[0,0] = False
        for step in range(1,9):
            if step == 8:
                self.front[0,0] = .1
                self.contact[0,0] = True
            self.tick(step)
            if step == 7:
                self.assertEqual(self.state.command_stage.tolist(), [1,1])
                self.assertEqual(self.state.stage.tolist(), [0,0])
        self.assertEqual(self.state.stage.tolist(), [1,0])
        self.assertAlmostEqual(self.state.touchdown_error[0,0].item(), 0., places=6)
        torch.testing.assert_close(self.state.step_distance, torch.tensor([.23,.23]))
        self.assertTrue(self.state.crossing_command.all())

    def test_no_false_success_at_sequence_end_and_no_double_tick(self):
        for step in range(1,39):
            self.tick(step)
            self.tick(step)
        self.assertEqual(self.state.control_steps.tolist(), [38,38])
        self.assertTrue(self.state.sequence_finished.all())
        self.assertTrue(self.state.failed.all())
        self.assertFalse(self.state.success.any())
        self.assertFalse(self.state.success_event.any())

    def test_partial_reset_restarts_only_one_clock(self):
        for step in range(1,10):
            self.tick(step)
        self.state.reset(torch.tensor([0]), self.front, step=9)
        self.tick(10)
        self.assertEqual(self.state.control_steps.tolist(), [1,10])
        self.assertEqual(self.state.command_stage.tolist(), [0,1])

    def test_geometry_success_must_still_be_valid_at_handoff(self):
        # Three real swing/touchdowns, then full support on the far side.
        for step in range(1,39):
            if step in (1,4,7):
                self.contact[0, 0 if step != 4 else 1] = False
            if step in (3,6,9):
                foot = 1 if step == 6 else 0
                self.front[0,foot] = .1 if step == 3 else .43
                self.rear[0,foot] = self.front[0,foot] - .148
                self.contact[0,foot] = True
            self.tick(step)
            if step == 10:
                self.assertFalse(self.state.success[0])
        self.assertTrue(self.state.success[0])
        self.assertTrue(self.state.clean_success[0])
        self.assertFalse(self.state.failed[0])
        self.assertTrue(self.state.failed[1])


if __name__ == '__main__':
    unittest.main()
