"""Grouped-reward copy of the GR1T flat-terrain task."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg
from isaaclab.utils.configclass import configclass

from Qmini.tasks.manager_based.gr1t.env.gr1t_env_cfg import RewardsCfg as FlatRewardsCfg
from Qmini.tasks.manager_based.gr1t.flat_env_cfg import QminiFlatEnvCfg

@configclass
class TaskRewardsCfg:
    """Command tracking and the minimum stability required to walk."""

    track_lin_vel_xy: RewardTermCfg | None = None
    track_ang_vel_z: RewardTermCfg | None = None
    alive: RewardTermCfg | None = None
    base_upright: RewardTermCfg | None = None
    torso_upright: RewardTermCfg | None = None
    base_height: RewardTermCfg | None = None
    stand_still: RewardTermCfg | None = None


@configclass
class CoordinationRewardsCfg:
    """Leg, arm, and posture coordination rewards from the flat task."""

    base_motion_penalty: RewardTermCfg | None = None
    torso_motion_penalty: RewardTermCfg | None = None
    base_lateral_sway: RewardTermCfg | None = None
    feet_air_time: RewardTermCfg | None = None
    stance_knee_flexion: RewardTermCfg | None = None
    swing_knee_flexion: RewardTermCfg | None = None
    leg_pitch_coordination: RewardTermCfg | None = None
    leg_joint_symmetry: RewardTermCfg | None = None
    diagonal_arm_follow_hip_penalty: RewardTermCfg | None = None
    foot_clearance: RewardTermCfg | None = None
    feet_step_symmetry: RewardTermCfg | None = None
    arm_posture: RewardTermCfg | None = None
    waist_posture: RewardTermCfg | None = None
    hip_roll_yaw_posture: RewardTermCfg | None = None


@configclass
class ConstraintRewardsCfg:
    """Contact, smoothness, energy, actuator, and termination constraints."""

    feet_slide: RewardTermCfg | None = None
    feet_stumble: RewardTermCfg | None = None
    flight_phase: RewardTermCfg | None = None
    action_rate_l2: RewardTermCfg | None = None
    joint_acc: RewardTermCfg | None = None
    joint_torques: RewardTermCfg | None = None
    joint_mechanical_power: RewardTermCfg | None = None
    joint_limits: RewardTermCfg | None = None
    joint_torque_limits: RewardTermCfg | None = None
    undesired_contacts: RewardTermCfg | None = None
    terminated: RewardTermCfg | None = None


@configclass
class GR1TMultiCriticRewardsCfg:
    """Semantic GR1T reward groups in critic-head order."""

    task: TaskRewardsCfg = TaskRewardsCfg()
    coordination: CoordinationRewardsCfg = CoordinationRewardsCfg()
    constraint: ConstraintRewardsCfg = ConstraintRewardsCfg()

    @classmethod
    def from_flat(cls, flat: FlatRewardsCfg) -> GR1TMultiCriticRewardsCfg:
        """Move the already tuned flat reward terms into semantic groups."""
        grouped = cls()
        for group_cfg in grouped.__dict__.values():
            for term_name in group_cfg.__dict__:
                if hasattr(flat, term_name):
                    setattr(group_cfg, term_name, getattr(flat, term_name))

        return grouped


@configclass
class GR1TMultiCriticFlatEnvCfg(QminiFlatEnvCfg):
    """GR1T flat task with unchanged reward terms reorganized into three groups."""

    def __post_init__(self) -> None:
        # The source GR1T flat config tunes a flat RewardsCfg. Let it finish first,
        # then reorganize those exact term objects without changing functions,
        # parameters, or RewTerm weights.
        self.rewards = FlatRewardsCfg()
        super().__post_init__()
        self.rewards = GR1TMultiCriticRewardsCfg.from_flat(self.rewards)


@configclass
class GR1TMultiCriticFlatEnvCfg_PLAY(GR1TMultiCriticFlatEnvCfg):
    """Reduced parallelism and disturbances for policy visualization."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.events.base_external_force_torque = None
        self.events.push_robot = None


__all__ = [
    "CoordinationRewardsCfg",
    "ConstraintRewardsCfg",
    "GR1TMultiCriticFlatEnvCfg",
    "GR1TMultiCriticFlatEnvCfg_PLAY",
    "GR1TMultiCriticRewardsCfg",
    "TaskRewardsCfg",
]
