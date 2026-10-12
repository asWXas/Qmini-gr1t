# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from .env.umv_env_cfg import LocomotionVelocityRoughEnvCfg


# =============================================================================
# Flat Environment
# =============================================================================

@configclass
class UMVFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Flat-terrain configuration for the UMV."""

    def __post_init__(self):
        super().__post_init__()
        # 平地: 取消地形生成器与地形难度课程
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None


# =============================================================================
# Play / Evaluation Environment
# =============================================================================


import Qmini.tasks.manager_based.umv.mdp as mdp

@configclass
class CommandsCfg:
    
    linerides = mdp.LineRidesCommand_Play_Cfg(
        # Robot / trajectory
        asset_name="robot",
        guideline_file="/home/wx/WX/Qmini/source/Qmini/Qmini/tasks/manager_based/umv/env/mini_hop.json",
        base_body_name="base_link",
        # Gamepad
        gamepad_index=0,
        gamepad_env_id=0,
        gamepad_required=True,
        gamepad_deadzone=0.08,
        # Vibration
        enable_rumble=True,
        rumble_strength=0.75,
        rumble_duration_ms=180,
        # MODE 0 joystick command ranges
        ranges=mdp.LineRidesCommand_Play_Cfg.Ranges(
            lin_vel_x=(-1.0, 2.0),  # 左摇杆 前/后 (-1 ~ +2 m/s)
            lin_vel_y=(0.0, 0.0),   # 无 lateral velocity
            ang_vel_z=(-1.5, 1.5),  # 右摇杆 左/右控制 yaw
            rho_upper=(0.0, 0.0),
            rho_lower=(0.6, 0.6),
        ),
        # Trajectory tracking
        waypoint_margin=0.30,
        start_waypoint_index=1,
        debug_num_envs=10,
    )

@configclass
class UMVFlatEnvCfg_PLAY(UMVFlatEnvCfg):
    """Flat-terrain configuration for visualization and evaluation."""

    # commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Evaluation samples all skills without advancing the training curriculum.
        self.curriculum.linerides_skills = None
        self.commands.linerides.skill_sampling_weights = (1.0, 1.0, 1.0)
        # 评估时不注入观测噪声
        self.observations.policy.enable_corruption = False
