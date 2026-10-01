"""Small offline checks for the HPC obstacle recording script."""

import os
import unittest
from types import SimpleNamespace

import numpy as np

from obstacle_software_video import render_side_frame
from play_obstacle_v31 import camera_view_for_root, configure_obstacle_episode, enable_video_capture, frame_has_content, joint_csv_header, joint_csv_row


class ObstacleRecordingTests(unittest.TestCase):
    def test_camera_uses_world_space_robot_position(self):
        eye, target = camera_view_for_root((-12.0, -75.0, 0.33))
        self.assertEqual(eye, (-11.9, -74.1, 0.75))
        self.assertEqual(target, (-11.9, -75.0, 0.25))

    def test_software_frame_draws_robot_and_obstacle_without_kit(self):
        frame = render_side_frame(
            np.array([0.0, 0.0, 0.32]),
            ["l_knee_pitch_link", "l_ankle_roll_link", "r_knee_pitch_link", "r_ankle_roll_link"],
            np.array([[0.03, 0.04, 0.18], [0.08, 0.04, 0.04], [0.0, -0.04, 0.18], [0.02, -0.04, 0.04]]),
            np.array([0.25, 0.0, 0.015]),
            False,
        )
        self.assertEqual(frame.shape, (544, 960, 3))
        self.assertTrue(np.any(np.all(frame == (214, 65, 42), axis=2)))
        self.assertTrue(np.any(np.all(frame == (44, 112, 214), axis=2)))
        self.assertTrue(np.any(np.all(frame == (35, 158, 121), axis=2)))

    def test_frame_check_rejects_black_capture(self):
        self.assertFalse(frame_has_content(np.zeros((4, 4, 3), dtype=np.uint8)))
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[1, 1, 0] = 128
        self.assertTrue(frame_has_content(frame))

    def test_video_capture_keeps_headless_viewport_active(self):
        args = SimpleNamespace(enable_cameras=False, video=False)
        enable_video_capture(args)
        self.assertTrue(args.enable_cameras)
        self.assertTrue(args.video)

    def test_configure_obstacle_episode_forces_physical_bar(self):
        reset = SimpleNamespace(params={"training_phase": 5, "phase_5_bar_episode_probability": 0.5})
        cfg = SimpleNamespace(
            scene=SimpleNamespace(num_envs=512, robot=SimpleNamespace(spawn=SimpleNamespace(usd_path="old"))),
            events=SimpleNamespace(reset_crossing_state=reset),
            commands=SimpleNamespace(base_velocity=SimpleNamespace(debug_vis=True)),
            viewer=SimpleNamespace(origin_type="world", asset_name=None, eye=None, lookat=None),
            video_recorder=SimpleNamespace(eye=None, lookat=None),
            sim=SimpleNamespace(device="cpu"),
            seed=0,
        )
        configure_obstacle_episode(cfg, "/robot/v3.1.usd", 7, "cuda:0")
        self.assertEqual(cfg.scene.num_envs, 1)
        self.assertEqual(cfg.scene.robot.spawn.usd_path, os.path.abspath("/robot/v3.1.usd"))
        self.assertEqual(reset.params["phase_5_bar_episode_probability"], 1.0)
        self.assertEqual(cfg.seed, 7)
        self.assertEqual(cfg.sim.device, "cuda:0")
        self.assertEqual(cfg.viewer.origin_type, "asset_root")
        self.assertEqual(cfg.viewer.eye, (0.1, 0.9, 0.55))
        self.assertEqual(cfg.video_recorder.eye, (0.1, 0.9, 0.75))
        self.assertEqual(cfg.video_recorder.lookat, (0.1, 0.0, 0.25))

    def test_rejects_non_phase_five_task(self):
        cfg = SimpleNamespace(events=SimpleNamespace(reset_crossing_state=SimpleNamespace(params={"training_phase": 4})))
        with self.assertRaisesRegex(ValueError, "Phase 5"):
            configure_obstacle_episode(cfg, "/robot/v3.1.usd", 7, None)

    def test_joint_columns_include_units_and_robot_order(self):
        self.assertEqual(
            joint_csv_header(["r_knee_pitch_joint", "l_knee_pitch_joint"]),
            ["step", "time_s", "crossed", "r_knee_pitch_joint_deg", "l_knee_pitch_joint_deg"],
        )

    def test_joint_row_converts_measured_radians_to_degrees(self):
        row = joint_csv_row(10, 0.02, True, [0.0, 3.141592653589793 / 2])
        self.assertEqual(row, [10, 0.2, 1, 0.0, 90.0])


if __name__ == "__main__":
    unittest.main()
