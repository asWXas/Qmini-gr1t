"""Isaac Lab configuration objects for multi-critic PPO."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg


@configclass
class RslRlMultiCriticPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO configuration whose critic groups define both count and ordering."""

    class_name: str = "Qmini.manager.multi_critic.algo.multi_critic_ppo:MultiCriticPPO"

    critic_groups: dict[str, dict[str, float]] = MISSING
    """Ordered mapping from reward group name to a mapping containing ``weight``."""

    normalize_critic_weights: bool = False
    """Normalize non-negative actor advantage weights to sum to one."""

    rnd_group: str | None = None
    """Group receiving RND intrinsic reward. Required when ``rnd_cfg`` is enabled."""


__all__ = ["RslRlMultiCriticPpoAlgorithmCfg"]
