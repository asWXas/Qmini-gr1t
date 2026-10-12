# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg,ImuCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import Qmini.tasks.manager_based.gr1t.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

from Qmini.asset.gr1t import GR1T2_CFG, JOINT_NAMES,LEG_JOINT_NAMES,WAIST_JOINT_NAMES,ARM_JOINT_NAMES


# ==============================================================================
# 通用传感器子配置（复用避免重复定义）
# ==============================================================================

FOOT_RAYCASTER_CFG = RayCasterCfg(
    prim_path=MISSING,  # 实例化时动态替换为具体 foot prim
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.2, 0.2)),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)


# ==============================================================================
# 场景主配置
# ==============================================================================

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # --------------------------------------------------------------------------
    # 地形与环境 (Ground & Lighting)
    # --------------------------------------------------------------------------
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/"
                "TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    sky_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=(
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/"
                "PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
            ),
        ),
    )

    # --------------------------------------------------------------------------
    # 机器人资产 (Robot)
    # --------------------------------------------------------------------------
    robot: ArticulationCfg = GR1T2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # --------------------------------------------------------------------------
    # 传感器组 (Sensors)
    # --------------------------------------------------------------------------
    # IMU 惯性测量单元
    # imu: ImuCfg = ImuCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/link_imu",
    #     debug_vis=False,
    # )

    # 躯干高度扫描
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # 足端高度扫描（复用通用模板）
    height_scanner_left: RayCasterCfg = FOOT_RAYCASTER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot/l_foot_roll"
    )
    height_scanner_right: RayCasterCfg = FOOT_RAYCASTER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot/r_foot_roll"
    )

    # 接触力检测
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.FourModeVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        # Four-mode sampling is x-only, y-only, yaw-only, or mixed;
        # standing is intentionally disabled so it is not a fifth mode.
        rel_standing_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.FourModeVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 5.0), lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-2.0, 2.0),
        ),
    )
    # base_velocity = mdp.UniformVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.02,
    #     rel_heading_envs=1.0,
    #     heading_command=True,
    #     heading_control_stiffness=0.5,
    #     debug_vis=True,
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
    #     ),
    # )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        scale={
            # =========================
            # Legs
            # =========================
            "[lr]_hip_roll": 0.15,
            "[lr]_hip_yaw": 0.20,
            "[lr]_hip_pitch": 0.35,
            "[lr]_knee_pitch": 0.45,
            "[lr]_ankle_pitch": 0.25,
            "[lr]_ankle_roll": 0.25,

            # =========================
            # Waist
            # =========================
            "joint_waist_yaw": 0.20,
            "joint_waist_pitch": 0.15,
            "joint_waist_roll": 0.15,

            # =========================
            # Arms
            # =========================
            "[lr]_shoulder_pitch": 0.35,
            "[lr]_shoulder_roll": 0.30,
            "[lr]_shoulder_yaw": 0.30,
            "[lr]_elbow_pitch": 0.40,
            "[lr]_wrist_yaw": 0.20,
        },
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations used by the actor."""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.25
        )

        base_projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        torso_ang_vel = ObsTerm(
            func=mdp.body_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["link_waist_roll"])},
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.25
        )

        torso_projected_gravity = ObsTerm(
            func=mdp.body_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["link_waist_roll"])},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=2.0
        )
        
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.02, n_max=0.02),
            # 安全裁剪: 只在数值发散时生效, 切断 "物理尖峰 -> 观测爆炸 ->
            # 动作爆炸 -> value function 被污染" 的正反馈链路。
            clip=(-10.0, 10.0),
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale = 0.25,
            # 裁剪后相当于 ±40 rad/s, 正常行走远达不到, 只拦发散。
            clip=(-10.0, 10.0),
        )

        gait_phase = ObsTerm(
            func=mdp.gait_phase,
            params={"period": 0.5},  # 步态周期 (秒)，可在此自由调节
        )

        gait_phase_h = ObsTerm(
            func=mdp.gait_phase,
            params={"period": 0.25},  # 步态周期 (秒)，可在此自由调节
        )
        
        gait_phase_l = ObsTerm(
            func=mdp.gait_phase,
            params={"period": 0.125},  # 步态周期 (秒)，可在此自由调节
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True
            
    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)

        base_projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )

        torso_ang_vel = ObsTerm(
            func=mdp.body_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["link_waist_roll"])},
            scale=0.25
        )

        torso_projected_gravity = ObsTerm(
            func=mdp.body_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["link_waist_roll"])},
        )

        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=2.0
        )
        
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            scale=0.25,
        )
        
        gait_phase = ObsTerm(
            func=mdp.gait_phase,
            params={"period": 0.5},  # 步态周期 (秒)，可在此自由调节
        )

        gait_phase_h = ObsTerm(
            func=mdp.gait_phase,
            params={"period": 0.25},  # 步态周期 (秒)，可在此自由调节
        )
        
        gait_phase_l = ObsTerm(
            func=mdp.gait_phase,
            params={"period": 0.125},  # 步态周期 (秒)，可在此自由调节
        )
        
        actions = ObsTerm(func=mdp.last_action)

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        air_time = ObsTerm(
            func=mdp.air_time,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", 
                    body_names=["l_foot_roll", "r_foot_roll"],
                    preserve_order=True
                )
            },
        )
        feet_height = ObsTerm(
            func=mdp.feet_height,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["l_foot_roll", "r_foot_roll"],
                    preserve_order=True
                ),
                "sensor_names": ["height_scanner_left", "height_scanner_right"],
            },
        )

        contact_forces = ObsTerm(
            func=mdp.contact_force_violation,
            params={
                "threshold": 1.0,
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", 
                    body_names=["l_foot_roll", "r_foot_roll"],
                    preserve_order=True
                ),
            },
            scale=0.05
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    body_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(?!l_foot_roll$|r_foot_roll$).*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.3),
            "num_buckets": 64,
        },
    )

    feet_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["l_foot_roll", "r_foot_roll"], preserve_order=True),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.4, 0.6),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "mass_distribution_params": (1.0 / 1.25, 1.25),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "force_range": (0.0, 20.0),
            "torque_range": (0.0, 3.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.0, 0.0),
                "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,  # type: ignore
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for rough-terrain velocity tracking."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0.0,
        params={
            "std": math.sqrt(0.25),
            "command_name": "base_velocity",
        },
    )

    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.0,
        params={
            "std": math.sqrt(0.25),
            "command_name": "base_velocity",
        },
    )

    base_upright = RewTerm(
        func=mdp.base_mothion_up_exp,
        weight=0.0,
        params={
            "std": math.sqrt(0.25),
        },
    )

    torso_upright = RewTerm(
        func=mdp.base_mothion_up_exp,
        weight=0.0,
        params={
            "std": math.sqrt(0.25),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["link_waist_roll"],
            ),
        },
    )
    
    
    base_motion_penalty = RewTerm(
        func=mdp.base_motion_penalty,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot",body_names=["base"]),
            "lin_vel_b": 0.8,
            "ang_vel_b": 0.2,
        }

    )
    
    torso_motion_penalty = RewTerm(
        func=mdp.body_motion_penalty,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["link_waist_roll"],
            ),
            "lin_vel_b": 0.8,
            "ang_vel_b": 0.2,
        }
    )

    base_lateral_sway = RewTerm(
        func=mdp.base_lateral_sway_penalty,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "lin_vel_weight": 1.0,
            "roll_rate_weight": 0.5,
            "roll_angle_weight": 0.5,
            "max_speed": 0.10,
        },
    )
    

    base_height = RewTerm(
        func=mdp.base_height_penalty,
        weight=-0.0,
        params={
            "target_height": 0.94,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.0,
        params={
            "command_name": "base_velocity",
            "legs_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
                preserve_order=True,
            ),
            "waist_cfg": SceneEntityCfg(
                "robot",
                joint_names=WAIST_JOINT_NAMES,
                preserve_order=True,
            ),
            "arms_cfg": SceneEntityCfg(
                "robot",
                joint_names=ARM_JOINT_NAMES,
                preserve_order=True,
            ),
            "legs_weight": 1.0,
            "waist_weight": 0.5,
            "arms_weight": 0.2,
            "stand_speed": 0.05,
            "move_speed": 0.1,
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.3,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "l_foot_roll",
                    "r_foot_roll",
                ],
                preserve_order=True,
            ),
        },
    )

    stance_knee_flexion = RewTerm(
        func=mdp.stance_knee_flexion_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["l_knee_pitch", "r_knee_pitch"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["l_foot_roll", "r_foot_roll"], preserve_order=True
            ),
            "command_name": "base_velocity",
            "max_stance_flexion": 0.25,
            "max_fast_flexion": 0.45,
            "fast_speed": 1.5,
            "contact_ramp_s": (0.02, 0.08),
        },
    )

    swing_knee_flexion: RewTerm = RewTerm(
        func=mdp.swing_knee_flexion_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["l_knee_pitch", "r_knee_pitch"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["l_foot_roll", "r_foot_roll"], preserve_order=True
            ),
            "command_name": "base_velocity",
        },
    )

    leg_pitch_coordination = RewTerm(
        func=mdp.leg_pitch_coordination_penalty,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "l_hip_pitch", "l_knee_pitch", "l_ankle_pitch",
                    "r_hip_pitch", "r_knee_pitch", "r_ankle_pitch",
                ],
                preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["l_foot_roll", "r_foot_roll"],
                preserve_order=True,
            ),
            "command_name": "base_velocity",
            "coordination_weight": 0.2,
            "participation_weight": 1.0,
            "swing_activity_weight": 2.0,
            "knee_activity_weight": 2.0,
            "min_hip_pos_ratio": 0.20,
            "min_knee_pos_ratio": 0.40,
            "min_ankle_pos_ratio": 0.10,
            "min_swing_activity": 0.25,
            "min_knee_activity": 0.15,
        },
    )

    leg_joint_symmetry = RewTerm(
        func=mdp.leg_joint_symmetry_penalty,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "l_hip_pitch", "l_knee_pitch", "l_ankle_pitch",
                    "r_hip_pitch", "r_knee_pitch", "r_ankle_pitch",
                ],
                preserve_order=True,
            ),
            "command_name": "base_velocity",
            "speed_threshold": 0.1,
        },
    )

    diagonal_arm_follow_hip_penalty = RewTerm(
        func=mdp.diagonal_arm_follow_hip_penalty,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "l_hip_pitch",
                    "r_hip_pitch",
                    "l_shoulder_pitch",
                    "r_shoulder_pitch",
                ],
                preserve_order=True,
            ),
            "command_name": "base_velocity",
            "min_scale": 0.30,
            "max_scale": 0.65,
            "min_speed": 0.10,
            "max_speed": 1.00,
            "command_threshold": 0.10,
            "mirror_sign": 1.0,
        },
    )

    foot_clearance = RewTerm(
        func=mdp.foot_clearance_scan_reward,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "l_foot_roll",
                    "r_foot_roll",
                ],
                preserve_order=True,
            ),
            "left_scanner_name": "height_scanner_left",
            "right_scanner_name": "height_scanner_right",
            "target_height": 0.08,
            "std": 0.08,
            "tanh_mult": 2.0,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["l_foot_roll", "r_foot_roll"], preserve_order=True
            ),
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "l_foot_roll",
                    "r_foot_roll",
                ],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "l_foot_roll",
                    "r_foot_roll",
                ],
                preserve_order=True,
            ),
        },
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble_penalty,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["l_foot_roll", "r_foot_roll"],
                preserve_order=True,
            ),
            "threshold": 4.0,
        },
    )

    flight_phase = RewTerm(
        func=mdp.flight_phase_penalty,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["l_foot_roll", "r_foot_roll"],
                preserve_order=True,
            ),
            "command_name": "base_velocity",
            "speed_threshold": 0.1,
        },
    )

    feet_step_symmetry = RewTerm(
        func=mdp.feet_step_symmetry_penalty,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["l_foot_roll", "r_foot_roll"],
                preserve_order=True,
            ),
            "command_name": "base_velocity",
            "speed_threshold": 0.1,
        },
    )


    arm_posture = RewTerm(
        func=mdp.arm_posture_l1_penalty,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "l_shoulder_roll",
                    "l_shoulder_yaw",
                    "l_elbow_pitch",
                    "l_wrist_yaw",
                    "r_shoulder_roll",
                    "r_shoulder_yaw",
                    "r_elbow_pitch",
                    "r_wrist_yaw",
                ],
                preserve_order=True,
            ),
            "command_name": "base_velocity",
            "min_speed": 0.0,
            "max_speed": 1.0,
            "min_elbow_offset": -0.35,
            "max_elbow_offset": -1.40,
        },
    )

    waist_posture = RewTerm(
        func=mdp.waist_l1_penalty,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "joint_waist_yaw",
                    "joint_waist_pitch",
                    "joint_waist_roll",
                ],
                preserve_order=True,
            ),
            "hip_cfg": SceneEntityCfg(
                "robot", joint_names=["l_hip_pitch", "r_hip_pitch"], preserve_order=True
            ),
            "command_name": "base_velocity",
            "yaw_gain": 0.20,
            "max_yaw": 0.15,
            "pitch_free": 0.08,
            "roll_free": 0.08,
            "forward_threshold": 0.10,
            "forward_full": 0.30,
        },
    )

    hip_roll_yaw_posture = RewTerm(
        func=mdp.hip_roll_yaw_l1_penalty,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "l_hip_roll",
                    "l_hip_yaw",
                    "r_hip_roll",
                    "r_hip_yaw",
                ],
                preserve_order=True,
            ),
            "command_name": "base_velocity",
        },
    )

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.0,
    )

    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=JOINT_NAMES,
                preserve_order=True,
            ),
        },
    )

    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=JOINT_NAMES,
                preserve_order=True,
            ),
        },
    )

    joint_mechanical_power = RewTerm(
        func=mdp.joint_mechanical_power_l1,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=JOINT_NAMES,
                preserve_order=True,
            ),
        },
    )

    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=JOINT_NAMES,
                preserve_order=True,
            ),
        },
    )

    joint_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=JOINT_NAMES,
                preserve_order=True,
            ),
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=r"^(?!l_foot_roll$|r_foot_roll$).*$",
            ),
        },
    )

    alive = RewTerm(
        func=mdp.is_alive,
        weight=0.0,
    )

    terminated = RewTerm(
        func=mdp.is_terminated,
        weight=0.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(60.0)},
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]),
            "threshold": 1.0,
        },
    )



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel) # type: ignore


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
