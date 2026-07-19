# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based rough-terrain locomotion environment config for the custom humanoid robot.

Suggested file name:
    humanoid_robot_policy_env_cfg.py

This file is designed for the project:
    Humanoid_Robot_Policy

Self-collision note:
    Fall detection is based on root orientation and root height. Contact sensors
    are used only for foot stepping rewards, not whole-body fall detection.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

from .humanoid_robot import HUMANOID_ROBOT_CFG

SMALL_RANDOM_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    # Size of each generated terrain patch.
    size=(8.0, 8.0),

    # Flat border around the complete terrain grid.
    border_width=10.0,

    # Creates 10 Ã— 20 = 200 terrain patches.
    num_rows=10,
    num_cols=20,

    # Resolution of the generated terrain.
    horizontal_scale=0.05,  # one mesh point every 5 cm
    vertical_scale=0.001,   # height resolution of 1 mm

    slope_threshold=0.75,
    curriculum=False,
    use_cache=False,

    sub_terrains={
        "small_random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,

            # Ground elevation varies from -5 mm to +5 mm.
            noise_range=(-0.005, 0.005),

            # Heights are generated in 1 mm increments.
            noise_step=0.001,

            # Random samples are generated every 10 cm and interpolated.
            # This produces smoother deviations instead of sharp noise.
            downsampled_scale=0.10,

            # Flat padding around each individual patch.
            border_width=0.25,
        ),
    },
)



##
# Robot-specific names
##

LEG_JOINT_NAMES = [
    # Right leg
    "r_leg_pitch_joint",
    "r_leg_roll_joint",
    "r_leg_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",

    # Left leg
    "l_leg_pitch_joint",
    "l_leg_roll_joint",
    "l_leg_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
]



BASE_BODY_NAME = "base_link"

FOOT_BODY_NAMES = [
    "l_ankle_roll_link",
    "r_ankle_roll_link",
]

TARGET_BASE_HEIGHT = 0.32
MIN_BASE_HEIGHT = 0.20
MAX_BASE_TILT = math.radians(65.0)

WOODEN_BAR_LENGTH = 0.35
WOODEN_BAR_WIDTH = 0.01
# Nine evenly spaced heights from 2 mm to 20 mm.
WOODEN_BAR_HEIGHTS = (
    0.00100,  # 2.00 mm
    0.00225,  # 4.25 mm
    0.00350,  # 6.50 mm
    0.00475,  # 8.75 mm
    0.00600,  # 11.00 mm
    0.00925,  # 13.25 mm
    0.01250,  # 15.50 mm
    0.01775,  # 17.75 mm
    0.02000,  # 20.00 mm
)

WOODEN_BAR_NAMES = (
    "wooden_bar_level_1",
    "wooden_bar_level_2",
    "wooden_bar_level_3",
    "wooden_bar_level_4",
    "wooden_bar_level_5",
    "wooden_bar_level_6",
    "wooden_bar_level_7",
    "wooden_bar_level_8",
    "wooden_bar_level_9",
)

# A separate collisionless bar is used during stride training.  Keeping it
# separate preserves the physical colliders required by obstacle training.
STRIDE_WOODEN_BAR_NAME = "wooden_bar_stride"
ALL_WOODEN_BAR_NAMES = WOODEN_BAR_NAMES + (STRIDE_WOODEN_BAR_NAME,)

ANKLE_JOINT_NAMES = [
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]

YAW_ROLL_JOINT_NAMES = [
    ".*_leg_yaw_joint",
    ".*_leg_roll_joint",
]


def _make_wooden_bar_cfg(prim_name: str, height: float) -> RigidObjectCfg:
    """Create one fixed-height bar variant for the obstacle curriculum."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=(WOODEN_BAR_WIDTH, WOODEN_BAR_LENGTH, height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.05,
                angular_damping=0.05,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=0.6,
                dynamic_friction=0.5,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.02, 0.02),
                roughness=0.7,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -2.0)),
    )


def _make_stride_training_bar_cfg(prim_name: str, height: float) -> RigidObjectCfg:
    """Create a collisionless visual reference for stride training."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=(WOODEN_BAR_WIDTH, WOODEN_BAR_LENGTH, height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.02, 0.02),
                roughness=0.7,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -2.0)),
    )


##
# Scene definition
##

@configclass
class HumanoidRobotPolicySceneCfg(InteractiveSceneCfg):
    """Scene configuration for rough-terrain walking with forward and turning commands."""

    # Randomly rough ground used to improve locomotion robustness.
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",

        # Generate small, smooth height variations instead of using an infinite plane.
        terrain_type="generator",
        terrain_generator=SMALL_RANDOM_ROUGH_TERRAIN_CFG,

        collision_group=-1,

        # Ground friction remains fixed.
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        ),

        debug_vis=False,
    )

    # Robot.
    robot: ArticulationCfg = HUMANOID_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # Simulated IMU attached to base_link.
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.0,
        debug_vis=False,

        # A stationary physical accelerometer normally reads approximately
        # +9.81 m/s^2 upward. Keep this consistent with the real IMU pipeline.
        gravity_bias=(0.0, 0.0, 9.81),

        # Replace these with the real IMU mounting pose relative to base_link.
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),  # quaternion: w, x, y, z
        ),
    )

    # Contact sensor used for foot stepping rewards only.
    # Do not use this for fall detection when self-collision is enabled,
    # because self-collision also produces contact forces.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
    )

    # Pre-create fixed collider heights. Runtime scale changes are unsupported
    # during tensorized training, so the curriculum activates one variant.
    wooden_bar_level_1 = _make_wooden_bar_cfg(
        "WoodenBarLevel1", WOODEN_BAR_HEIGHTS[0]
    )
    wooden_bar_level_2 = _make_wooden_bar_cfg(
        "WoodenBarLevel2", WOODEN_BAR_HEIGHTS[1]
    )
    wooden_bar_level_3 = _make_wooden_bar_cfg(
        "WoodenBarLevel3", WOODEN_BAR_HEIGHTS[2]
    )
    wooden_bar_level_4 = _make_wooden_bar_cfg(
        "WoodenBarLevel4", WOODEN_BAR_HEIGHTS[3]
    )
    wooden_bar_level_5 = _make_wooden_bar_cfg(
        "WoodenBarLevel5", WOODEN_BAR_HEIGHTS[4]
    )
    wooden_bar_level_6 = _make_wooden_bar_cfg(
        "WoodenBarLevel6", WOODEN_BAR_HEIGHTS[5]
    )
    wooden_bar_level_7 = _make_wooden_bar_cfg(
        "WoodenBarLevel7", WOODEN_BAR_HEIGHTS[6]
    )
    wooden_bar_level_8 = _make_wooden_bar_cfg(
        "WoodenBarLevel8", WOODEN_BAR_HEIGHTS[7]
    )
    wooden_bar_level_9 = _make_wooden_bar_cfg(
        "WoodenBarLevel9", WOODEN_BAR_HEIGHTS[8]
    )

    # Kinematic and collisionless: it stays at the requested ground pose while
    # the robot passes through it during stride training.
    wooden_bar_stride = _make_stride_training_bar_cfg(
        "WoodenBarStride", WOODEN_BAR_HEIGHTS[0]
    )

    # Light.
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            color=(0.9, 0.9, 0.9),
        ),
    )


##
# MDP: Commands
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.ObstacleAwareVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),

        # Standing commands are deliberately common enough to train reliable
        # stop-and-stand behavior. The command term suppresses them while a bar
        # is active.
        rel_standing_envs=0.05,

        # Use direct yaw-rate commands for turning. ``heading_command=False`` means
        # angular velocity is sampled from ``ang_vel_z`` instead of deriving it
        # from an absolute heading target. Therefore, ``rel_heading_envs`` is zero.
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,

        debug_vis=True,
        ranges=mdp.ObstacleAwareVelocityCommandCfg.Ranges(
            # Fixed forward speed with sampled yaw-rate commands.
            #start fresh
#            lin_vel_x=(0.2, 0.2),

            lin_vel_x=(0.4, 0.4),
            ang_vel_z=(0.0, 0.0),
            heading=(-math.pi, math.pi),
        ),
    )


##
# MDP: Actions
##

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        preserve_order=True,
        scale=0.25,
        use_default_offset=True,
    )


##
# MDP: Observations
##

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations used by the policy network."""

        # Three-axis acceleration measured in the simulated IMU frame.
        base_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc,
            params={"asset_cfg": SceneEntityCfg("imu")},

            # Initial estimate; later tune this using recordings from the real IMU.
            noise=Unoise(n_min=-0.3, n_max=0.3),

            # Convert typical acceleration magnitudes to roughly network-sized values.
            # For example, 9.81 m/s^2 becomes approximately 0.981.
            scale=0.1,
        )

        # Recommended: obtain all IMU-related observations from the same
        # simulated sensor frame.
        base_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )

        projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # Commanded walking velocity.
        velocity_commands = ObsTerm(
            func=mdp.forward_yaw_velocity_commands,
            params={"command_name": "base_velocity"},
        )

        # Signed distance to the active bar along the crossing direction. The
        # underlying value is 0.50 m when no bar is present or after both foot
        # centers have crossed it.
        wooden_bar_distance = ObsTerm(
            func=mdp.wooden_bar_distance,
            params={
                "bar_names": ALL_WOODEN_BAR_NAMES,
                "feet_cfg": SceneEntityCfg(
                    "robot",
                    body_names=FOOT_BODY_NAMES,
                    preserve_order=True,
                ),
                "noise_range": (-0.005, 0.005),
            },
        )

        # Joint state.
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=LEG_JOINT_NAMES,
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=LEG_JOINT_NAMES,
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )

        # Previous action.
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            """Post initialization."""
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# MDP: Events
##

@configclass
class EventCfg:
    """Configuration for events.

    Events handle startup/reset randomization.
    This first version is conservative for debugging.
    """

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            # Keep default standing pose at first.
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_wooden_bar = EventTerm(
        func=mdp.reset_wooden_bar,
        mode="reset",
        params={
            "bar_names": ALL_WOODEN_BAR_NAMES,
            "hidden_depth": 2.0,
            "stride_training_spawn_probability": 1.00,
            "obstacle_training_spawn_probability": 0.70,
            "stride_training_spawn_delay_s": 5.0,
            "obstacle_training_spawn_delay_range_s": (3.0, 6.0),
        },
    )

    # Poll once per control step. The reset event stores an episode-relative
    # appearance time: 5 s for stride training and 3-6 s for obstacle training.
    spawn_wooden_bar = EventTerm(
        func=mdp.spawn_wooden_bar,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        is_global_time=False,
        params={
            "bar_names": WOODEN_BAR_NAMES,
            "stride_bar_name": STRIDE_WOODEN_BAR_NAME,
            "bar_heights": WOODEN_BAR_HEIGHTS,
            "robot_name": "robot",
            "stride_training_distance_range": (0.20, 0.20),
            "obstacle_training_distance_range": (0.50, 0.50),
            "drop_clearance": 0.01,
            "command_name": "base_velocity",
        },
    )

    # Randomize the physics material of the two feet.
    randomize_foot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=FOOT_BODY_NAMES,
            ),

            # One random material is assigned to each environment.
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.5, 1.1),

            # Keep feet non-bouncy.
            "restitution_range": (0.0, 0.1),

            # Discretize the random range into material buckets.
            "num_buckets": 64,

            # Prevent physically inconsistent combinations such as
            # dynamic friction being greater than static friction.
            "make_consistent": True,
        },
    )

    # Randomize actuator response independently across simulated robots.
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
                preserve_order=True,
            ),

            # Multipliers applied to the nominal values in humanoid_robot.py.
            "stiffness_distribution_params": (0.90, 1.10),
            "damping_distribution_params": (0.80, 1.20),

            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
                preserve_order=True,
            ),
            "friction_distribution_params": (0.0, 0.05),
            "operation": "add",
            "distribution": "uniform",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the bipedal humanoid walking MDP.

    This is G1-like, but modified to fight two early local optima:
    1. standing still
    2. shuffling/sliding feet without real stepping
    """

    # -------------------------------------------------------------------------
    # Main task rewards
    # -------------------------------------------------------------------------

    # Stronger and sharper than your current version.
    # Your old std=0.5 was too forgiving, so standing still could still get reward.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp_relative,
        weight=4.0,
        params={
            "command_name": "base_velocity",
            "std": 0.20,
            "moving_command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # Track the sampled yaw-rate commands so the policy learns turning together
    # with forward locomotion.
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=3,
        params={
            "command_name": "base_velocity",
            "std": 0.3,
        },
    )

    # -------------------------------------------------------------------------
    # Anti-shuffling / stepping terms
    # -------------------------------------------------------------------------

    # G1-like foot timing reward using only l_ankle_roll_link and r_ankle_roll_link.
    # This assumes the USD collision filters prevent foot self-collision from
    # corrupting foot contact timing. If that still happens later, replace this
    # with a custom ground-filtered reward.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FOOT_BODY_NAMES,
            ),
            # 0.4 is G1-like. For your smaller/custom robot, start slightly lower.
            "threshold": 0.25,
        },
    )

    # G1-like foot slide penalty using only l_ankle_roll_link and r_ankle_roll_link.
    # This assumes the USD collision filters prevent foot self-collision from
    # corrupting foot contact timing. If that still happens later, replace this
    # with a custom ground-filtered reward.
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.35,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FOOT_BODY_NAMES,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=FOOT_BODY_NAMES,
            ),
        },
    )

    # -------------------------------------------------------------------------
    # Contact / termination terms
    # -------------------------------------------------------------------------

    # Strong penalty for early termination, G1-style.
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-200.0,
    )

    # Extra penalty applied only when the wooden bar moves.
    wooden_bar_moved_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=50.0,
        params={
            "term_keys": "wooden_bar_moved",
        },
    )

    # Give a matching extra penalty when the obstacle-crossing deadline expires.
    wooden_bar_deadline_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-200.0,
        params={
            "term_keys": "wooden_bar_deadline",
        },
    )

    # Disabled: fall is detected by root orientation and root height, not contact forces.
    illegal_non_foot_contact = None

    # -------------------------------------------------------------------------
    # Stability terms
    # -------------------------------------------------------------------------

    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-3,
    )

    # Penalize sudden sideways base acceleration.
    base_acc_y_l2 = RewTerm(
        func=mdp.base_acceleration_l2,
        weight=-0.005,
        params={
            "axis": "y",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[BASE_BODY_NAME],
            ),
        },
    )

    # Penalize bouncing and sudden vertical base acceleration.
    base_acc_z_l2 = RewTerm(
        func=mdp.base_acceleration_l2,
        weight=-0.005,
        params={
            "axis": "z",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[BASE_BODY_NAME],
            ),
        },
    )

    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.2,
    )

    # -------------------------------------------------------------------------
    # Joint / action penalties
    # -------------------------------------------------------------------------

    # Make these weaker at first.
    # If they are too strong, the easiest solution is "do not move".
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
            )
        },
    )

    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.0e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
            )
        },
    )

    # Slightly weaker than G1 at first. Increase later when walking works.
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.002,
    )

    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=ANKLE_JOINT_NAMES,
            )
        },
    )

    # Do not penalize leg pitch/knee pitch too much, because those are needed
    # for stepping. Only softly discourage sideways/yaw flailing.
    joint_deviation_yaw_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.03,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=YAW_ROLL_JOINT_NAMES,
            )
        },
    )

    #both feet airborn penalty
    both_feet_airborne = RewTerm(
        func=mdp.both_feet_airborne,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FOOT_BODY_NAMES,
            ),
        },
    )

    # # Sparse task reward for fully clearing the bar within its lateral span.
    # wooden_bar_crossing = RewTerm(
    #     func=mdp.wooden_bar_crossing_reward,
    #     weight=100.0,
    #     params={
    #         "feet_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=FOOT_BODY_NAMES,
    #             preserve_order=True,
    #         ),
    #     },
    # )
    wooden_bar_crossing = None

    # Stage two only: encourage the swing foot to reach 3 cm above the support
    # foot while a nearby wooden bar is visible.
    feet_clearance = RewTerm(
        func=mdp.feet_clearance_reward,
        weight=1.5,
        params={
            "target_height": 0.03,
            "command_name": "base_velocity",
            "bar_names": ALL_WOODEN_BAR_NAMES,
            "activation_distance": 0.20,
            "full_weight_distance": 0.10,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=FOOT_BODY_NAMES,
                preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FOOT_BODY_NAMES,
                preserve_order=True,
            ),
        },
    )

    # Stage two only: strongly reward touchdown strides longer than one 14 cm
    # foot, with full reward at 20 cm. The MDP term supplies the phase/bar mask.
    feet_stride_length = RewTerm(
        func=mdp.feet_stride_length_reward,
        weight=50.0,
        params={
            "foot_length": 0.14,
            "target_stride_length": 0.20,
            "command_name": "base_velocity",
            "bar_names": ALL_WOODEN_BAR_NAMES,
            "activation_distance": 0.20,
            "full_weight_distance": 0.15,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=FOOT_BODY_NAMES,
                preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FOOT_BODY_NAMES,
                preserve_order=True,
            ),
        },
    )


##
# MDP: Terminations
##

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(
        func=mdp.curriculum_time_out,
        time_out=True,
        params={
            "normal_training_length_s": 20.0,
            "stride_training_length_s": 10.0,
            "obstacle_training_length_s": 20.0,
        },
    )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "limit_angle": MAX_BASE_TILT,
        },
    )

    low_base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_height": MIN_BASE_HEIGHT,
        },
    )

    wooden_bar_moved = DoneTerm(
        func=mdp.wooden_bar_moved,
        params={
            "bar_names": ALL_WOODEN_BAR_NAMES,
            "feet_cfg": SceneEntityCfg(
                "robot",
                body_names=FOOT_BODY_NAMES,
                preserve_order=True,
            ),
            "translation_tolerance": 0.005,
            "rotation_tolerance": math.radians(5.0),
            "settling_time_s": 0.20,
        },
    )

    wooden_bar_deadline = DoneTerm(
        func=mdp.wooden_bar_deadline,
        params={
            "feet_cfg": SceneEntityCfg(
                "robot",
                body_names=FOOT_BODY_NAMES,
                preserve_order=True,
            ),
            "time_limit_s": 10.0,
        },
    )


##
# MDP: Curriculum
##

@configclass
class CurriculumCfg:
    """Learn normal walking, long strides near a bar, then obstacle crossing."""
# start fresh
    # fixed_forward_speed = CurrTerm(
    #     func=mdp.fixed_forward_speed_curriculum,
    #     params={
    #         "command_name": "base_velocity",
    #         "initial_speed": 0.2,
    #         "final_speed": 0.40,
    #         "start_step": 500,
    #         # 24,000 control steps / 24 rollout steps
    #         # is approximately 1,000 PPO iterations.
    #         "end_step": 2_500,
    #     },
    # )

    wooden_bar = CurrTerm(
        func=mdp.wooden_bar_curriculum,
        params={
            "bar_heights": WOODEN_BAR_HEIGHTS,
            # Stage one: normal walking before step 6,000.
            # Stage two: stride training from step 6,000 through step 40,000.
            "stride_training_start_step": 4_000,
            # Stage three: current nine-height obstacle curriculum.
            "obstacle_training_start_step": 60_001,
            "end_step": 180_000,
        },
    )


    stride_reward_weights = CurrTerm(
        func=mdp.stride_reward_weight_curriculum,
        params={
            # Must match wooden_bar.stride_training_start_step.
            "stride_training_start_step": 4_000,

            # Reach the current weights here and remain constant afterward.
            "decay_end_step": 40_000,

            # Start at 3x the current values.
            "initial_clearance_weight": 5.0,
            "initial_stride_weight": 200.0,

            # Current values from RewardsCfg.
            "final_clearance_weight": 1.5,
            "final_stride_weight": 50.0,
        },
    )

    # termination_penalty_after_bar = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={
    #         "term_name": "termination_penalty",
    #         "weight": -200.0,
    #         # Use the same step at which wooden-bar training begins.
    #         "num_steps": 5_000,
    #     },
    # )


    # wooden_bar = CurrTerm(
    #     func=mdp.wooden_bar_curriculum,
    #     params={
    #         "bar_heights": WOODEN_BAR_HEIGHTS,
    #         # Stage one: normal walking before step 6,000.
    #         # Stage two: 50% nearby-bar stride training from 6,000 to 12,000.
    #         "stride_training_start_step": 0,
    #         # Stage three: current nine-height obstacle curriculum.
    #         "obstacle_training_start_step": 40_000,
    #         "end_step": 180_000,
    #     },
    # )





##
# Environment configuration
##

@configclass
class HumanoidRobotPolicyEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for rough-terrain forward-velocity and yaw-rate tracking."""

    # Resume the global curriculum clock alongside model_650.
    # One PPO iteration collects 24 environment/control steps (see
    # HumanoidRobotRoughPPORunnerCfg.num_steps_per_env), therefore:
    #
    #     model_650 -> 650 * 24 = 15,600 curriculum steps
    #
    # Set this to 0 when training a new policy from scratch.  For another
    # model_N checkpoint, set it to N * num_steps_per_env.
    curriculum_start_step: int = 0


    # Scene settings.
    scene: HumanoidRobotPolicySceneCfg = HumanoidRobotPolicySceneCfg(
        num_envs=512,
        env_spacing=2.5,
    )

    # Basic settings.
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings.
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""

        # General settings.
        self.decimation = 4
        # Maximum buffer length; curriculum_time_out ends earlier phases at
        # 8 s (normal) or 10 s (stride), and obstacle episodes at 20 s.
        self.episode_length_s = 20.0

        # Simulation settings.
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # Sensor update periods.
        self.scene.contact_forces.update_period = self.sim.dt

        # Viewer.
        self.viewer.eye = (4.0, 4.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.6)

        # Update the simulated IMU at every physics step: 0.005 s = 200 Hz.
        self.scene.imu.update_period = self.sim.dt


##
# Play / visualization configuration
##

@configclass
class HumanoidRobotPolicyEnvCfg_PLAY(HumanoidRobotPolicyEnvCfg):
    """Keyboard-controlled configuration for policy playback."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10.0
        self.terminations.time_out.params.update(
            {
                "normal_training_length_s": 10.0,
                "stride_training_length_s": 10.0,
                "obstacle_training_length_s": 10.0,
            }
        )

        # Keyboard controls the base_velocity command.
        self.commands.base_velocity.class_type = (
            mdp.KeyboardVelocityCommand
        )

        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.ranges.heading = None

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0

        # Make the final obstacle curriculum visible immediately during playback.
        self.curriculum.wooden_bar.params["stride_training_start_step"] = 0
        # self.curriculum.wooden_bar.params["obstacle_training_start_step"] = 0
        self.curriculum.wooden_bar.params["end_step"] = 0
        self.events.reset_wooden_bar.params["obstacle_training_spawn_probability"] = 1.0

        self.events.reset_wooden_bar.params["obstacle_training_spawn_delay_range_s"] = (0.0, 0.0)

        self.observations.policy.enable_corruption = False
        self.observations.policy.wooden_bar_distance.params["noise_range"] = (0.0, 0.0)


        # Camera follows the robot in environment 0.
        self.viewer.origin_type = "asset_root"
        self.viewer.env_index = 0
        self.viewer.asset_name = "robot"

        # Camera position and target relative to the robot root.
        self.viewer.eye = (2.0, 2.0, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        #  # Playback should immediately use the final competition speed.
        # self.curriculum.fixed_forward_speed.params["start_step"] = 0
        # self.curriculum.fixed_forward_speed.params["end_step"] = 0
