"""Record one v3.1 USD physical-bar crossing episode and actual joint angles."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import math
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK = "Humanoid-Robot-RSLRL-Legacy-Play-v0"


def configure_obstacle_episode(env_cfg, usd_path: str, seed: int, device: str | None) -> None:
    """Use one environment and force Phase 5's physical-bar branch on every reset."""
    reset_params = env_cfg.events.reset_crossing_state.params
    if reset_params["training_phase"] != 5:
        raise ValueError("This recording requires the Phase 5 crossing task.")
    reset_params["phase_5_bar_episode_probability"] = 1.0
    env_cfg.scene.num_envs = 1
    env_cfg.scene.robot.spawn.usd_path = os.path.abspath(usd_path)
    env_cfg.commands.base_velocity.debug_vis = False
    # Follow the root from the side with a clear view of both the robot and upcoming hurdles.
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (0.2, 1.4, 0.6)
    env_cfg.viewer.lookat = (0.45, 0.0, 0.0)
    # The Kit video recorder has its own world-space camera, separate from viewer.
    env_cfg.video_recorder.eye = (0.2, 1.4, 0.6)
    env_cfg.video_recorder.lookat = (0.45, 0.0, 0.0)
    env_cfg.seed = seed
    if device is not None:
        env_cfg.sim.device = device


def joint_csv_header(joint_names: list[str]) -> list[str]:
    return ["step", "time_s", "crossed", *(f"{name}_deg" for name in joint_names)]


def joint_csv_row(step: int, control_dt: float, crossed: bool, joint_pos_rad: list[float]) -> list:
    return [
        step,
        step * control_dt,
        int(crossed),
        *(math.degrees(value) for value in joint_pos_rad),
    ]


def enable_video_capture(launcher_args) -> None:
    """Keep Kit's offscreen renderer and default viewport active for RecordVideo."""
    launcher_args.enable_cameras = True
    launcher_args.video = True


def frame_has_content(frame) -> bool:
    """Return whether an RGB camera frame contains a visible pixel."""
    if frame is None:
        return False
    pixels = np.asarray(frame)
    return pixels.ndim == 3 and pixels.shape[-1] >= 3 and bool(np.any(pixels[..., :3] > 2))


def camera_view_for_root(root_position) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """World-space side camera that keeps the robot and upcoming hurdles in clear view."""
    x, y, z = (float(value) for value in root_position)
    return (x + 0.2, y + 1.4, z + 0.35), (x + 0.45, y, z - 0.15)


def move_kit_recording_camera(env, root_position) -> None:
    from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view

    eye, target = camera_view_for_root(root_position)
    set_kit_renderer_camera_view(eye, target, camera_prim_path=env.cfg.viewer.cam_prim_path)


def main() -> None:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import setup_preset_cli
    from isaaclab_tasks.utils.hydra import hydra_task_config

    sys.path.insert(0, str(PROJECT_ROOT))
    import tasks  # noqa: F401  Register the custom Gym task.

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "model_33999.pt"))
    parser.add_argument("--usd", default=str(PROJECT_ROOT / "v3.1" / "v3.1.usd"))
    parser.add_argument("--steps", type=int, default=250, help="Maximum control steps in the first episode.")
    parser.add_argument("--video-backend", choices=("kit", "software"), default="kit")
    parser.add_argument("--video-stride", type=int, default=4, help="Control steps per 3-D video frame.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "recordings" / f"v31_obstacle_{os.environ.get('SLURM_JOB_ID', 'manual')}"),
    )
    add_launcher_args(parser)
    args_cli, remaining_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *remaining_args]
    if args_cli.video_backend == "kit":
        enable_video_capture(args_cli)

    @hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
    def run(env_cfg, agent_cfg) -> None:
        if args_cli.steps <= 0:
            raise ValueError("--steps must be positive")
        if args_cli.video_stride <= 0:
            raise ValueError("--video-stride must be positive")
        for label, filename in (("checkpoint", args_cli.checkpoint), ("USD", args_cli.usd)):
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"{label} missing: {filename}")

        configure_obstacle_episode(env_cfg, args_cli.usd, agent_cfg.seed, args_cli.device)
        if args_cli.video_backend == "kit":
            # A 4 GB RTX 3050 cannot sustain 1280x720 rendering at every 50 Hz control step.
            env_cfg.video_recorder.window_width = 640
            env_cfg.video_recorder.window_height = 360
            env_cfg.sim.render_interval = env_cfg.decimation * args_cli.video_stride
        output_dir = Path(args_cli.output_dir).resolve()
        video_dir = output_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "joint_angles_deg.csv"
        control_dt = env_cfg.sim.dt * env_cfg.decimation

        with launch_simulation(env_cfg, args_cli):
            agent_cfg = handle_deprecated_rsl_rl_cfg(
                agent_cfg, importlib.metadata.version("rsl-rl-lib")
            )
            env = gym.make(
                TASK,
                cfg=env_cfg,
                render_mode="rgb_array" if args_cli.video_backend == "kit" else None,
            )
            try:
                if args_cli.video_backend == "kit":
                    # The recorder copies its own world-space camera config at env creation.
                    # Scene origins can be far from (0, 0, 0), so configure it before reset.
                    origin = env.unwrapped.scene.env_origins[0].detach().cpu().tolist()
                    initial_root = (origin[0], origin[1], origin[2] + 0.33)
                    eye, target = camera_view_for_root(initial_root)
                    capture = env.unwrapped.video_recorder._capture
                    if capture is None:
                        raise RuntimeError("Kit video capture was not initialized")
                    capture.cfg.eye = eye
                    capture.cfg.lookat = target
                    print(f"[RECORD] initial camera eye={eye} target={target}", flush=True)
                env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
                if agent_cfg.class_name != "OnPolicyRunner":
                    raise ValueError(f"Unexpected runner: {agent_cfg.class_name}")
                runner = OnPolicyRunner(
                    env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
                )
                runner.load(os.path.abspath(args_cli.checkpoint))
                policy = runner.get_inference_policy(device=env.unwrapped.device)
                obs = env.get_observations()
                if obs["policy"].shape[-1] != 49:
                    raise RuntimeError(
                        f"Expected 49 policy observations, got {obs['policy'].shape[-1]}"
                    )

                robot = env.unwrapped.scene["robot"]
                crossing_state = env.unwrapped._wooden_bar_state
                if bool(crossing_state.phase_5_no_bar_episode[0]):
                    raise RuntimeError("The first episode has no obstacle.")
                print(f"[RECORD] USD: {env_cfg.scene.robot.spawn.usd_path}", flush=True)
                print(f"[RECORD] joints: {robot.data.joint_names}", flush=True)
                if args_cli.video_backend == "kit":
                    print(
                        f"[RECORD] root position: {robot.data.root_pos_w[0].detach().cpu().tolist()}",
                        flush=True,
                    )

                from obstacle_software_video import SoftwareVideoWriter

                crossed = False
                visible_frame_seen = False
                captured_frames = 0
                if args_cli.video_backend == "software":
                    from obstacle_software_video import render_side_frame

                    bar = env.unwrapped.scene["wooden_bar"]
                    software_video_path = video_dir / "obstacle-software.mp4"
                    video_context = SoftwareVideoWriter(
                        str(software_video_path), fps=1.0 / control_dt
                    )
                    print("[RECORD] software side-view video enabled", flush=True)
                else:
                    kit_video_path = video_dir / "obstacle-3d.mp4"
                    video_context = SoftwareVideoWriter(
                        str(kit_video_path),
                        fps=1.0 / (control_dt * args_cli.video_stride),
                        width=env_cfg.video_recorder.window_width,
                        height=env_cfg.video_recorder.window_height,
                    )
                    print(
                        f"[RECORD] 3-D recording every {args_cli.video_stride} control steps",
                        flush=True,
                    )
                with video_context as video_writer, csv_path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(joint_csv_header(robot.data.joint_names))
                    for step in range(1, args_cli.steps + 1):
                        with torch.inference_mode():
                            actions = policy(obs)
                            obs, _, dones, _ = env.step(actions)
                            policy.reset(dones)
                        crossed |= bool(crossing_state.crossing_completed[0])
                        joint_pos_rad = robot.data.joint_pos[0].detach().cpu().tolist()
                        writer.writerow(joint_csv_row(step, control_dt, crossed, joint_pos_rad))
                        if args_cli.video_backend == "software":
                            frame = render_side_frame(
                                robot.data.root_pos_w[0].detach().cpu().numpy(),
                                robot.data.body_names,
                                robot.data.body_pos_w[0].detach().cpu().numpy(),
                                bar.data.root_pos_w[0].detach().cpu().numpy(),
                                crossed,
                            )
                            video_writer.append_data(frame)
                        elif step == 1 or step % args_cli.video_stride == 0 or bool(dones[0]):
                            move_kit_recording_camera(
                                env.unwrapped,
                                robot.data.root_pos_w[0].detach().cpu().tolist(),
                            )
                            frame = env.unwrapped.render(recompute=True)
                            if frame_has_content(frame):
                                video_writer.append_data(frame)
                                captured_frames += 1
                                visible_frame_seen = True
                            if step == 1 or step % 50 == 0:
                                print(
                                    f"[RECORD] 3-D frame step={step} visible={frame_has_content(frame)}",
                                    flush=True,
                                )
                        if step == 1 or step % 25 == 0 or bool(dones[0]):
                            print(
                                f"[RECORD] step={step} crossed={crossed} done={bool(dones[0])}",
                                flush=True,
                            )
                        if bool(dones[0]):
                            print(f"[RECORD] Episode terminated at step={step}:", flush=True)
                            if hasattr(env.unwrapped, "termination_manager"):
                                for term_name in env.unwrapped.termination_manager.active_terms:
                                    term_val = env.unwrapped.termination_manager.get_term(term_name)[0].item()
                                    if term_val:
                                        print(f"  [TRIGGERED TERMINATION] {term_name}: {term_val}", flush=True)
                                    else:
                                        print(f"  - {term_name}: {term_val}", flush=True)
                            break
                print(f"[RECORD] joint angles: {csv_path}", flush=True)
                print(f"[RECORD] crossing completed: {crossed}", flush=True)
                if args_cli.video_backend == "kit":
                    print(f"[RECORD] 3-D frames: {captured_frames}", flush=True)
            finally:
                env.close()

            videos = sorted(video_dir.glob("*.mp4"))
            if not videos:
                raise RuntimeError(f"No MP4 was produced in {video_dir}")
            for video in videos:
                print(f"[RECORD] video: {video}", flush=True)
            import cv2

            video_path = software_video_path if args_cli.video_backend == "software" else kit_video_path
            capture = cv2.VideoCapture(str(video_path))
            decoded, first_frame = capture.read()
            capture.release()
            if not decoded or not np.any(first_frame > 2):
                raise RuntimeError(f"Video could not be decoded or is black: {video_path}")
            if args_cli.video_backend == "kit" and (
                not visible_frame_seen or int(np.max(first_frame)) < 150
            ):
                raise RuntimeError(f"3-D video has no visible robot in its first frame: {video_path}")
            print("[RECORD] MP4 first frame decoded and visible", flush=True)

    run()


if __name__ == "__main__":
    main()
