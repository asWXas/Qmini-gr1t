# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
from pathlib import Path

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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import Qmini.tasks.manager_based.umv.mdp as mdp
##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

from Qmini.asset.umv import UMV_CFG


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
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
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = UMV_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the environment."""

    linerides = mdp.LineRidesCommandCfg(
        asset_name="robot",
        guideline_files={
            "01": str(Path(__file__).with_name("mini_hop.json")),
            "10": str(Path(__file__).with_name("backflip.json")),
            "11": str(Path(__file__).with_name("three_point_turn.json")),
        },
        # Stage 1 samples only 01; the curriculum unlocks 10 and 11.
        skill_sampling_weights=(1.0, 1.0, 1.0),
        drive_only_fraction=0.1,
        # Initial training envelope, not a calibrated hardware turning limit.
        # Set None to recover fully independent vx/wz sampling.
        drive_max_curvature=1.5,
        base_body_name="base_link",
        stunt_switch_time_range=(2.0, 5.0),
        lin_vel_resampling_time_range=(2.0, 15.0),
        ang_vel_resampling_time_range=(3.0, 8.0),
        rho_resampling_time_range=(3.0, 10.0),
        ranges=mdp.LineRidesCommandCfg.Ranges(
            lin_vel_x=(-1.0, 2.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.5, 1.5),
            rho_upper=(0.06, 1.0),    # q_h (head)
            rho_lower=(-0.47, 0.52),   # q_n (neck)
        ),
        waypoint_margin=0.15,
        start_waypoint_index=1,
        debug_vis=True,
        debug_num_envs=10,
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    body_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["q_s", "q_n", "q_h"],
        scale=0.25,
        use_default_offset=True,
        clip={
            ".*": (-100.0, 100.0),
        },
    )
    drive_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["q_w"],
        scale=5.0,
        use_default_offset=False,
        clip={
            ".*": (-100.0, 100.0),
        },
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Deployment-facing actor observations; no privileged base linear velocity."""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.25
        )
        base_projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        linerides_command = ObsTerm(
            func=mdp.policy_command,
            params={
                "command_name": "linerides",
            },
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_sincos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "q_h",
                        "q_n",
                        "q_s",
                        "q_w",
                    ],
                    preserve_order=True
                ),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "q_h",
                        "q_n",
                        "q_s",
                        "q_w",
                    ],
                    preserve_order=True
                ),
            },
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale = 0.1,
            clip=(-100.0, 100.0),
        )
        actions = ObsTerm(func=mdp.last_action,clip=(-100.0, 100.0))

        def __post_init__(self):
            self.history_length = 3
            self.enable_corruption = True
            self.concatenate_terms = True
            
    
    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel,clip=(-10.0, 10.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2,clip=(-10.0, 10.0))

        base_projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        velocity_commands = ObsTerm(
            func=mdp.critic_command,
            params={"command_name": "linerides"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_sincos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "q_h",
                        "q_n",
                        "q_s",
                        "q_w",
                    ],
                    preserve_order=True
                ),
            },
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "q_h",
                        "q_n",
                        "q_s",
                        "q_w",
                    ],
                    preserve_order=True
                ),
            },
            scale = 0.1,
            clip=(-100.0, 100.0),
        )
        actions = ObsTerm(func=mdp.last_action,clip=(-100.0, 100.0))
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()



@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.3),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.9, 1.25),
            "operation": "scale",
        },
    )
    add_head_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="head"),
            "mass_distribution_params": (0.9, 1.25),
            "operation": "scale",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )
    head_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="head"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.02, 0.02)},
        },
    )

    # # reset
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    # # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task model = 0
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.lin_vel_x_exp, weight=3.0, params={"command_name": "linerides", "std": math.sqrt(0.2)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.ang_vel_z_exp, weight=3.0, params={"command_name": "linerides", "std": math.sqrt(0.2)}
    )
    # Broad, non-negative rewards complement the precise exponential kernel.
    # drive_lin_vel_soft = RewTerm(
    #     func=mdp.drive_velocity_tracking_soft,
    #     weight=1.0,
    #     params={"command_name": "linerides", "component": "lin_vel_x", "beta": 0.5, "max_error": 5.0},
    # )
    # drive_ang_vel_soft = RewTerm(
    #     func=mdp.drive_velocity_tracking_soft,
    #     weight=1.0,
    #     params={"command_name": "linerides", "component": "ang_vel_z", "beta": 0.5, "max_error": 5.0},
    # )
    
    track_rho_exp = RewTerm(
        func=mdp.track_rho_exp,
        weight=2.0,
        params={
            "std": math.sqrt(0.2),
            "command_name": "linerides",
            "asset_cfg": SceneEntityCfg("robot", joint_names=["q_h", "q_n"], preserve_order=True),
        },
    )
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2_m, weight=-0.01)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2_m, weight=-1.0)

    # -- task model = 1
    
    track_waypoint_exp = RewTerm(
        func=mdp.track_waypoint_exp,
        weight=0.75,
        params={"std": 0.15, "command_name": "linerides"},
    )

    track_waypoint_line = RewTerm(
        func=mdp.track_waypoint_line,
        weight=3.0,  # progress in m/step; RewardManager also multiplies by step_dt
        params={"command_name": "linerides"},
    )
    
    target_update_bonus = RewTerm(
        func=mdp.target_update_bonus,
        weight=15.0,
        params={
            "command_name": "linerides",
            "atol": 1e-5,
        },
    )
    
    stunt_orientation = RewTerm(
        func=mdp.stunt_orientation_error_p,
        weight=-0.01,
        params={
            "command_name": "linerides",
            "asset_name": "robot",
            "base_body_name": "base_link",
        },
    )

    joint_position_error = RewTerm(
        func=mdp.stunt_joint_position_error,
        weight=-0.001,
        params={
            "command_name": "linerides",
            "asset_name": "robot",
            "joint_names": ["q_n", "q_h"],
        },
    )
    
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-2.5)
        
        
    
    # track_waypoint_bonus = RewTerm(
    #     func=mdp.track_waypoint_bonus,          # 必须传"类"本身,不能传实例
    #     weight=5.0,
    #     params={"margin": 0.16, "command_name": "linerides"},
    # )
        
        
        
    #################################
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_U, weight=-0.02)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2_m, weight=-0.01)
    


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link","head"]), "threshold": 1.0},
    )
    # stunt_yaw_deviation = DoneTerm(
    #     func=mdp.stunt_yaw_deviation,
    #     params={
    #         "command_name": "linerides",
    #         "asset_name": "robot",
    #         "base_body_name": "base_link",
    #         "max_yaw_deg": 90.0,
    #     },
    # )

    track_line_position_deviation = DoneTerm(
        func=mdp.stunt_position_deviation,
        params={
            "command_name": "linerides",
            "max_distance": 1.0,
        },
    )

    target_stalled = DoneTerm(
        func=mdp.target_not_update_death,
        params={
            "command_name": "linerides",
            "max_no_update_steps": 300,
            "atol": 1e-5,
        },
        time_out=False,
    )

@configclass
class CurriculumCfg:
    """Unlock LineRides skills after the current skill reaches its final waypoint reliably."""

    linerides_skills = CurrTerm(
        func=mdp.LineRidesSkillCurriculum,
        params={
            "command_name": "linerides",
            "start_stage": 0,
            "min_attempts": 256,
            "success_thresholds": (0.6, 0.5, 0.4),
            "consecutive_windows": 2,
        },
    )


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
    # curriculum: CurriculumCfg = CurriculumCfg()

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
