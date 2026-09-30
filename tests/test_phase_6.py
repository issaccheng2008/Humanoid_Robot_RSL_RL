"""CPU command/configuration checks; no Isaac Sim process is required."""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
    BACKEND = "PyTorch CPU"
except ImportError:
    BACKEND = "NumPy tensor adapter (PyTorch unavailable)"
    rng = np.random.default_rng(42)

    class Tensor(np.ndarray):
        def uniform_(self, low, high):
            self[:] = low if low == high else rng.uniform(low, high, self.shape)
            return self

        def nonzero(self, as_tuple=False):
            if as_tuple:
                return tuple(x.view(Tensor) for x in np.nonzero(np.asarray(self)))
            return np.argwhere(np.asarray(self)).view(Tensor)

        def clone(self):
            return self.copy()

    def tensor(value, dtype=None, device=None):
        return np.asarray(value, dtype=dtype).view(Tensor)

    def zeros(size, dtype=None, device=None):
        return np.zeros(size, dtype=dtype or np.float32).view(Tensor)

    torch = ModuleType("torch")
    torch.Tensor = Tensor
    torch.bool = np.bool_
    torch.long = np.int64
    torch.as_tensor = tensor
    torch.zeros = zeros
    torch.empty = lambda size, device=None: np.empty(size, dtype=np.float32).view(Tensor)
    torch.rand = lambda size, device=None: rng.random(size).astype(np.float32).view(Tensor)
    torch.arange = lambda size, device=None: np.arange(size).view(Tensor)
    torch.norm = lambda value, dim: np.linalg.norm(value, axis=dim).view(Tensor)
    torch.abs = lambda value: np.abs(value).view(Tensor)


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/manager_based/humanoid_robot_policy_rsl_rl"


class Node(SimpleNamespace):
    Ranges = SimpleNamespace


class UniformVelocityCommandStub:
    """Only the command-manager lifecycle and base standing behavior are stubbed."""

    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env
        self.device, self.num_envs = env.device, env.num_envs
        self.robot = env.scene[cfg.asset_name]
        self.vel_command_b = torch.zeros((self.num_envs, 3))
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool)
        self.command_counter = torch.zeros(self.num_envs, dtype=torch.long)
        self.time_left = torch.zeros(self.num_envs)
        self.metrics = {name: torch.zeros(self.num_envs) for name in ("error_vel_xy", "error_vel_yaw")}

    def _resample(self, env_ids):
        self.time_left[env_ids] = self.cfg.resampling_time_range[0]
        self._resample_command(env_ids)
        self.command_counter[env_ids] += 1

    def reset(self, env_ids):
        self.command_counter[env_ids] = 0
        self._resample(env_ids)
        return {}

    def compute(self, dt):
        self._update_metrics()
        self.time_left -= dt
        env_ids = (self.time_left <= 0).nonzero(as_tuple=False).flatten()
        if len(env_ids):
            self._resample(env_ids)
        self._update_command()

    def _update_command(self):
        self.vel_command_b[self.is_standing_env] = 0.0


class PPOStub:
    def __init__(self, *args, **kwargs):
        self.entropy_coef = kwargs["entropy_coef"]

    def load(self, saved, *args, **kwargs):
        return saved.get("iteration", 0)

    def save(self):
        return {}

    def update(self):
        return {}


def load_module(name, path, stubs):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


commands = load_module("phase6_commands_tested", TASK / "mdp/phase_6_commands.py", {
    "torch": torch,
    "isaaclab": ModuleType("isaaclab"),
    "isaaclab.envs": ModuleType("isaaclab.envs"),
    "isaaclab.envs.mdp": SimpleNamespace(
        UniformVelocityCommand=UniformVelocityCommandStub,
        UniformVelocityCommandCfg=Node,
    ),
    "isaaclab.utils": SimpleNamespace(configclass=lambda cls: cls),
})
entropy = load_module("phase6_entropy_tested", TASK / "agents/entropy_schedule.py", {
    "rsl_rl": ModuleType("rsl_rl"),
    "rsl_rl.algorithms": SimpleNamespace(PPO=PPOStub),
})


def arrays_equal(actual, expected):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-7)


def setUpModule():
    print(f"Phase 6 test backend: {BACKEND}")


class Phase6CommandTests(unittest.TestCase):
    def make_command(self, size=8, **overrides):
        cfg = Node(
            asset_name="robot", resampling_time_range=(math.inf, math.inf),
            command_update_interval_s=1.0, yaw_change_probability=0.2,
            speed_change_probability=0.1, stop_probability=0.3, stop_time_s=7.0,
            ranges=Node(lin_vel_x=(0.05, 0.4), ang_vel_z=(-1.0, 1.0)),
        )
        for key, value in overrides.items():
            setattr(cfg, key, value)
        env = Node(num_envs=size, device="cpu", step_dt=0.02, max_episode_length=500,
                   episode_length_buf=torch.zeros(size, dtype=torch.long))
        robot_data = Node(root_lin_vel_b=torch.zeros((size, 3)), root_ang_vel_b=torch.zeros((size, 3)))
        env.scene = {"robot": Node(data=robot_data)}
        command = commands.Phase6VelocityCommand(cfg, env)
        env.command_manager = Node(get_command=lambda name: command.vel_command_b)
        command.reset()
        return command, env

    def test_defaults_and_initial_distribution(self):
        cls = commands.Phase6VelocityCommandCfg
        self.assertEqual((cls.yaw_change_probability, cls.speed_change_probability, cls.stop_probability),
                         (0.2, 0.1, 0.3))
        command, env = self.make_command(20000)
        values = np.asarray(command.vel_command_b)
        self.assertTrue(np.all((values[:, 0] >= 0.05) & (values[:, 0] <= 0.4)))
        self.assertTrue(np.all((values[:, 2] >= -1) & (values[:, 2] <= 1)))
        self.assertAlmostEqual(float(values[:, 0].mean()), 0.225, delta=0.004)
        self.assertAlmostEqual(float(values[:, 2].mean()), 0.0, delta=0.025)
        self.assertAlmostEqual(float(np.asarray(command._stop_episode).mean()), 0.30, delta=0.02)
        arrays_equal(values[:, 1], 0)
        arrays_equal(commands.phase_6_crossing_command(env), np.zeros((20000, 1)))
        self.assertFalse(hasattr(env, "_wooden_bar_state"))

    def test_one_second_boundary_and_no_double_update(self):
        command, env = self.make_command(stop_probability=0, yaw_change_probability=1, speed_change_probability=1)
        old = command.vel_command_b.clone()
        for step in range(1, 50):
            env.episode_length_buf[:] = step
            command.compute(env.step_dt)
        arrays_equal(command.vel_command_b, old)
        env.episode_length_buf[:] = 50
        command.compute(env.step_dt)
        self.assertFalse(np.array_equal(np.asarray(command.vel_command_b), np.asarray(old)))
        updated = command.vel_command_b.clone()
        command.compute(env.step_dt)
        arrays_equal(command.vel_command_b, updated)
        arrays_equal(command.command_counter, 1)

    def test_independent_probability_masks(self):
        command, env = self.make_command(4, stop_probability=0)
        command.vel_command_b[:, 0] = 0.8  # Sentinels outside sampling ranges.
        command.vel_command_b[:, 2] = 2.0
        env.episode_length_buf[:] = 50
        draws = [torch.as_tensor([0.1, 0.3, 0.1, 0.3]), torch.as_tensor([0.2, 0.05, 0.05, 0.2])]
        with patch.object(torch, "rand", side_effect=draws):
            command._update_command()
        values = np.asarray(command.vel_command_b)
        np.testing.assert_array_equal(values[:, 2] != 2, [True, False, True, False])
        np.testing.assert_array_equal(values[:, 0] != 0.8, [False, True, True, False])

    def test_update_probabilities_over_many_episodes(self):
        command, env = self.make_command(20000, stop_probability=0)
        command.vel_command_b[:, 0] = 0.8
        command.vel_command_b[:, 2] = 2.0
        env.episode_length_buf[:] = 50
        command._update_command()
        values = np.asarray(command.vel_command_b)
        yaw, speed = values[:, 2] != 2, values[:, 0] != np.float32(0.8)
        self.assertAlmostEqual(float(yaw.mean()), 0.20, delta=0.02)
        self.assertAlmostEqual(float(speed.mean()), 0.10, delta=0.02)
        self.assertAlmostEqual(float((yaw & speed).mean()), 0.02, delta=0.008)

    def test_stop_boundary_persistence_and_reset(self):
        command, env = self.make_command(2, stop_probability=1, yaw_change_probability=1, speed_change_probability=1)
        command._stop_episode[1] = False
        env.episode_length_buf[:] = 349
        command._update_command()
        self.assertTrue(np.all(np.asarray(command.vel_command_b)[:, 0] > 0))
        for step in (350, 351, 400, 450, 499):
            env.episode_length_buf[:] = step
            command._update_command()
            arrays_equal(command.vel_command_b[0], [0, 0, 0])
            arrays_equal(commands.phase_6_step_distance_command(env, "base_velocity", 0.08, 0.4)[0], [0])
            self.assertGreater(float(command.vel_command_b[1, 0]), 0)
        old_other = command.vel_command_b[1].clone()
        # Isaac resets managers before clearing the episode length buffer.
        command.reset([0])
        env.episode_length_buf[0] = 0
        command._update_command()
        self.assertGreater(float(command.vel_command_b[0, 0]), 0)
        arrays_equal(command.vel_command_b[1], old_other)
        self.assertEqual(int(command._last_update_tick[0]), 0)

    def test_asynchronous_episode_clocks(self):
        command, env = self.make_command(2, stop_probability=0, yaw_change_probability=1, speed_change_probability=1)
        command.vel_command_b[:, 0] = 0.8
        env.episode_length_buf[:] = torch.as_tensor([49, 50])
        command._update_command()
        self.assertAlmostEqual(float(command.vel_command_b[0, 0]), 0.8)
        self.assertLessEqual(float(command.vel_command_b[1, 0]), 0.4)
        command.reset(slice(0, 1))
        env.episode_length_buf[0] = 0
        arrays_equal(command._last_update_tick, [0, 1])

    def test_step_scaling_tracks_speed_immediately(self):
        command, env = self.make_command(3, stop_probability=0)
        command.vel_command_b[:, 0] = torch.as_tensor([0.05, 0.4, 0])
        target = commands.phase_6_step_distance_command(env, "base_velocity", 0.08, 0.4)
        arrays_equal(target, [[0.01], [0.08], [0]])
        command.vel_command_b[:, 0] = 0.2
        arrays_equal(commands.phase_6_step_distance_command(env, "base_velocity", 0.08, 0.4), 0.04)

    def test_tracking_metrics_remain_nonzero(self):
        command, env = self.make_command(2, stop_probability=0)
        command.vel_command_b[:, 0] = 0.4
        command.vel_command_b[:, 2] = 1.0
        command._update_metrics()
        arrays_equal(command.metrics["error_vel_xy"], 0.4 / 500)
        arrays_equal(command.metrics["error_vel_yaw"], 1 / 500)


class Phase6ConfigAndResumeTests(unittest.TestCase):
    def test_phase6_wiring_and_observation_slots(self):
        source = ast.parse((TASK / "humanoid_robot_policy_rsl_rl_env_cfg.py").read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "HumanoidRobotPolicyEnvCfg")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_configure_phase_6")
        namespace = {
            "mdp": Node(Phase6VelocityCommandCfg=Node,
                        phase_6_step_distance_command=commands.phase_6_step_distance_command,
                        phase_6_crossing_command=commands.phase_6_crossing_command),
            "PHASE_6_EPISODE_LENGTH_S": 10.0, "PHASE_6_LIN_VEL_X_RANGE": (0.05, 0.4),
            "PHASE_6_ANG_VEL_Z_RANGE": (-1.0, 1.0), "PHASE_6_REFERENCE_SPEED": 0.4,
            "DEFAULT_STEP_DISTANCE": 0.08,
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), "phase6_cfg", "exec"), namespace)
        policy = Node(step_distance=Node(), crossing_command=Node(), velocity_commands=Node())
        cfg = Node(scene=Node(), events=Node(), terminations=Node(),
                   rewards=Node(track_lin_vel_xy_exp="retained", track_ang_vel_z_exp="retained"),
                   curriculum=Node(policy_observation_shape="retained"), commands=Node(),
                   observations=Node(policy=policy))
        namespace["_configure_phase_6"](cfg)
        self.assertEqual(cfg.episode_length_s, 10)
        self.assertIsNone(cfg.scene.wooden_bar)
        self.assertIsNone(cfg.scene.collisionless_wooden_bar)
        self.assertTrue(all(value is None for value in vars(cfg.events).values()))
        self.assertIsNone(cfg.rewards.step_distance_tracking_reward)
        self.assertTrue(all(value is None for name, value in vars(cfg.rewards).items() if "bar" in name or "band" in name))
        self.assertEqual(cfg.rewards.track_lin_vel_xy_exp, "retained")
        self.assertEqual(cfg.curriculum.policy_observation_shape, "retained")
        self.assertIsNone(cfg.curriculum.step_distance_gaussian)
        self.assertEqual(cfg.commands.base_velocity.ranges.lin_vel_x, (0.05, 0.4))
        self.assertEqual(cfg.commands.base_velocity.ranges.ang_vel_z, (-1, 1))
        self.assertEqual(policy.step_distance.params["reference_step_distance"], 0.08)
        self.assertIs(policy.crossing_command.func, commands.phase_6_crossing_command)
        policy_cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "ObservationsCfg")
        nested = next(n for n in policy_cls.body if isinstance(n, ast.ClassDef))
        observation_names = [n.targets[0].id for n in nested.body if isinstance(n, ast.Assign)]
        self.assertEqual(observation_names, ["base_lin_acc", "base_ang_vel", "projected_gravity", "velocity_commands",
                                             "step_distance", "crossing_command", "joint_pos", "joint_vel", "actions"])
        self.assertEqual(sum([3, 3, 3, 2, 1, 1, 12, 12, 12]), 49)

    def test_phase5_to_phase6_entropy_resume(self):
        algorithm = entropy.EntropyScheduledPPO(training_phase=6, entropy_schedule=((0, 0.0005),))
        algorithm.load({"entropy_schedule_state": {"training_phase": 5, "phase_iteration": 10000}})
        self.assertEqual(algorithm.entropy_schedule_iteration, 0)
        self.assertEqual(algorithm.entropy_coef, 0.0005)
        algorithm.update()
        saved = algorithm.save()
        restored = entropy.EntropyScheduledPPO(training_phase=6, entropy_schedule=((0, 0.0005),))
        restored.load(saved)
        self.assertEqual(restored.entropy_schedule_iteration, 1)
        self.assertEqual(restored.entropy_coef, 0.0005)

    def test_legacy_entropy_phases_still_resume(self):
        for phase in range(1, 6):
            algorithm = entropy.EntropyScheduledPPO(training_phase=phase, entropy_schedule=((0, 0.002),))
            algorithm.load({"entropy_schedule_state": {"training_phase": phase, "phase_iteration": 25}})
            self.assertEqual(algorithm.entropy_schedule_iteration, 25)


if __name__ == "__main__":
    unittest.main()
