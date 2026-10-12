import math

import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils.configclass import configclass

from .env.gr1t_env_cfg import LocomotionVelocityRoughEnvCfg

# =============================================================================
# Flat Environment
# =============================================================================

# Mostly flat terrain with low-amplitude height variation.  The plane patch is
# deliberately the dominant mode so the policy can retain its learned gait;
# the rough patches add only mild foot-placement and balance perturbations.
LIGHT_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.30),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.40,
            noise_range=(0.01, 0.04),
            noise_step=0.01,
            border_width=0.25,
        ),
        "gentle_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.30,
            slope_range=(0.1, 0.30),
            platform_width=3.0,
            border_width=0.25,
        ),
    },
)

@configclass
class QminiFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Mostly-flat, mildly rough terrain configuration for GR1T2 locomotion."""

    def __post_init__(self):
        super().__post_init__()
        # Straight legs put the GR1T2 sole about 0.953 m below the base.
        # Start just above the floor so reset does not begin with bent knees.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.96)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = LIGHT_ROUGH_TERRAINS_CFG
        self.curriculum.terrain_levels = None
        self.scene.terrain.terrain_generator.curriculum = False

        # ==================================================================
        # Commands
        # ==================================================================

        # Four motion modes plus an explicit (0, 0, 0) command.
        # Probabilities must be non-negative and sum to 1.0.
        self.commands.base_velocity.prob_x_only = 0.225
        self.commands.base_velocity.prob_y_only = 0.09
        self.commands.base_velocity.prob_yaw_only = 0.135
        self.commands.base_velocity.prob_mixed = 0.45
        self.commands.base_velocity.prob_zero = 0.10

        # ----------------------------------------------------------
        # Velocity tracking
        # ----------------------------------------------------------
        self.rewards.track_lin_vel_xy.weight = 2.0
        self.rewards.track_lin_vel_xy.params["std"] = math.sqrt(0.25)
        self.rewards.track_lin_vel_xy.params["command_name"] = "base_velocity"

        self.rewards.track_ang_vel_z.weight = 1.5
        self.rewards.track_ang_vel_z.params["std"] = math.sqrt(0.25)
        self.rewards.track_ang_vel_z.params["command_name"] = "base_velocity"

        # ----------------------------------------------------------
        # Base / torso posture
        # ----------------------------------------------------------
        self.rewards.base_upright.weight = 0.25
        self.rewards.base_upright.params["std"] = math.sqrt(0.25)

        self.rewards.torso_upright.weight = 1.0
        self.rewards.torso_upright.params["std"] = math.sqrt(0.25)
        
        self.rewards.base_motion_penalty.weight = -0.10
        self.rewards.torso_motion_penalty.weight = -0.15

        # 抑制左右摇摆 / "鸭子步": 横向速度 + roll 角速度 + 侧倾角
        self.rewards.base_lateral_sway.weight = -0.1
        self.rewards.base_lateral_sway.params["lin_vel_weight"] = 1.0
        self.rewards.base_lateral_sway.params["roll_rate_weight"] = 0.5
        self.rewards.base_lateral_sway.params["roll_angle_weight"] = 0.5
        self.rewards.base_lateral_sway.params["max_speed"] = 0.10

        self.rewards.base_height.weight = -8.0
        self.rewards.base_height.params["target_height"] = 0.95

        # ----------------------------------------------------------
        # Standstill
        # ----------------------------------------------------------
        self.rewards.stand_still.weight = -0.5
        self.rewards.stand_still.params["legs_weight"] = 1.0
        self.rewards.stand_still.params["waist_weight"] = 0.5
        self.rewards.stand_still.params["arms_weight"] = 0.2
        self.rewards.stand_still.params["stand_speed"] = 0.05
        self.rewards.stand_still.params["move_speed"] = 0.10

        # ----------------------------------------------------------
        # Gait
        # ----------------------------------------------------------
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.15
        self.rewards.feet_air_time.params["command_name"] = "base_velocity"

        # Single support: encourage an extended knee, with about 7 deg of slack.
        # Touchdown/double support: allow cushioning; swing stays unrestricted.
        self.rewards.stance_knee_flexion.weight = -4.0
        self.rewards.stance_knee_flexion.params["max_stance_flexion"] = 0.12
        self.rewards.stance_knee_flexion.params["walk_speed"] = 1.0
        self.rewards.stance_knee_flexion.params["fast_speed"] = 2.5
        self.rewards.stance_knee_flexion.params["max_fast_flexion"] = 0.45
        self.rewards.stance_knee_flexion.params["max_transition_flexion"] = 0.30
        self.rewards.stance_knee_flexion.params["contact_ramp_s"] = (0.04, 0.12)

        # Bend during mid-swing; release before the estimated landing time.
        self.rewards.swing_knee_flexion.weight = -1.0
        self.rewards.swing_knee_flexion.params["min_flexion"] = 0.35
        self.rewards.swing_knee_flexion.params["min_fast_flexion"] = 0.55
        self.rewards.swing_knee_flexion.params["speed_range"] = (1.0, 2.5)
        self.rewards.swing_knee_flexion.params["swing_duration_range"] = (0.20, 0.50)
        self.rewards.swing_knee_flexion.params["initial_swing_duration"] = 0.30

        # Prefer ground support at walking speeds; permit flight when running.
        self.rewards.flight_phase.weight = -0.5
        self.rewards.flight_phase.params["running_speed_range"] = (1.0, 2.0)
        self.rewards.flight_phase.params["air_time_tolerance"] = 0.02

        self.rewards.foot_clearance.weight = 1.0
        self.rewards.foot_clearance.params["target_height"] = 0.10
        self.rewards.foot_clearance.params["std"] = 0.08
        self.rewards.foot_clearance.params["tanh_mult"] = 2.0

        self.rewards.feet_slide.weight = -0.5
        self.rewards.feet_stumble.weight = -0.04
        self.rewards.feet_stumble.params["threshold"] = 4.0


        # ----------------------------------------------------------
        # Arm <-> opposite hip coordination
        # ----------------------------------------------------------
        self.rewards.diagonal_arm_follow_hip_penalty.weight = -1.0#-0.60
        self.rewards.diagonal_arm_follow_hip_penalty.params["min_scale"] = 0.40
        self.rewards.diagonal_arm_follow_hip_penalty.params["max_scale"] = 1.0
        self.rewards.diagonal_arm_follow_hip_penalty.params["min_speed"] = 0.10
        self.rewards.diagonal_arm_follow_hip_penalty.params["max_speed"] = 2.00
        self.rewards.diagonal_arm_follow_hip_penalty.params["command_threshold"] = 0.10
        # URDF: positive shoulder_pitch and hip_pitch both swing toward body -X.
        self.rewards.diagonal_arm_follow_hip_penalty.params["mirror_sign"] = 1.0
        self.rewards.diagonal_arm_follow_hip_penalty.params["remove_hip_mean"] = True
        self.rewards.diagonal_arm_follow_hip_penalty.params["position_tolerance"] = 0.05

        # ----------------------------------------------------------
        # Arm posture
        # ----------------------------------------------------------
        self.rewards.arm_posture.weight = -1.0#-0.04
        self.rewards.arm_posture.params["min_speed"] = 0.1
        self.rewards.arm_posture.params["max_speed"] = 2.0
        self.rewards.arm_posture.params["min_elbow_offset"] = -0.10
        self.rewards.arm_posture.params["max_elbow_offset"] = -0.6

        # ----------------------------------------------------------
        # Waist posture & Hip roll/yaw posture
        # ----------------------------------------------------------
        self.rewards.waist_posture.weight = -0.50
        self.rewards.waist_posture.params["command_name"] = "base_velocity"

        self.rewards.hip_roll_yaw_posture.weight = -0.10
        self.rewards.hip_roll_yaw_posture.params["command_name"] = "base_velocity"

        # ----------------------------------------------------------
        # Smoothness / energy
        # ----------------------------------------------------------
        self.rewards.action_rate_l2.weight = -0.015
        self.rewards.joint_acc.weight = -2.5e-7
        self.rewards.joint_torques.weight = -1.0e-7
        # Absolute mechanical power (W); RewardManager applies the control dt.
        self.rewards.joint_mechanical_power.weight = -2.0e-4

        # ----------------------------------------------------------
        # Limits & Contacts & Episode
        # ----------------------------------------------------------
        self.rewards.joint_limits.weight = -1.0
        self.rewards.joint_torque_limits.weight = 0.0

        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["threshold"] = 1.0

        # self.rewards.alive.weight = 0.25
        self.rewards.terminated.weight = -2.0

        # Isaac Lab skips computing zero-weight terms, but still registers and
        # displays them. None keeps them out of RewardManager and its logs.
        for name, term in list(vars(self.rewards).items()):
            if isinstance(term, RewardTermCfg) and term.weight == 0.0:
                setattr(self.rewards, name, None)


# =============================================================================
# Play / Evaluation Environment
# =============================================================================

@configclass
class QminiFlatEurekaEnvCfg(QminiFlatEnvCfg):
    """Eureka uses the existing critic group as its reward state."""


@configclass
class QminiFlatEnvCfg_PLAY(QminiFlatEnvCfg):
    """Flat-terrain configuration for visualization and evaluation."""

    def __post_init__(self):
        """Post initialization."""

        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.events.base_external_force_torque = None
        self.events.push_robot = None
