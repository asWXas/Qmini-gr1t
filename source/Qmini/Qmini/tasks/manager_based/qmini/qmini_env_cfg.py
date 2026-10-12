# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal grouped-reward cart-pole environment for Multi-Critic PPO."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

# isort: off
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG
# isort: on


@configclass
class QminiSceneCfg(InteractiveSceneCfg):
    """A dependency-free cart-pole scene suitable for checking the algorithm wiring."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class ActionsCfg:
    """Cart force action."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["slider_to_cart"],
        scale=100.0,
    )


@configclass
class ObservationsCfg:
    """Shared actor/critic observations for the base template."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Episode reset randomization."""

    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )


@configclass
class RewardsCfg:
    """Three semantic groups; their names/order must match ``critic_groups``."""

    @configclass
    class TaskCfg:
        # Primary objective: survive while keeping the pole upright.
        alive = RewTerm(func=mdp.is_alive, weight=1.0)
        pole_pos = RewTerm(
            func=mdp.joint_pos_target_l2,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
                "target": 0.0,
            },
        )

    @configclass
    class AestheticCfg:
        # Smooth, low-speed motion shaping.
        cart_vel = RewTerm(
            func=mdp.joint_vel_l1,
            weight=-0.01,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
        )
        pole_vel = RewTerm(
            func=mdp.joint_vel_l1,
            weight=-0.005,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
        )

    @configclass
    class ConstraintCfg:
        # Constraint/failure signal remains separate from the task objective.
        terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    task: TaskCfg = TaskCfg()
    aesthetic: AestheticCfg = AestheticCfg()
    constraint: ConstraintCfg = ConstraintCfg()


@configclass
class TerminationsCfg:
    """Time and cart travel limits."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "bounds": (-3.0, 3.0),
        },
    )


@configclass
class QminiEnvCfg(ManagerBasedRLEnvCfg):
    """Multi-Critic cart-pole training configuration."""

    scene: QminiSceneCfg = QminiSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation


__all__ = ["QminiEnvCfg"]
