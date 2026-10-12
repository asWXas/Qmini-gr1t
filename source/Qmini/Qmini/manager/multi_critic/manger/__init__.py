"""Grouped reward manager and environment integration.

The ``manger`` package name is retained for compatibility with the existing project path.
"""

from .reward_manager import MultiCriticRewardManager
from .rl_env import MultiCriticManagerBasedRLEnv

__all__ = ["MultiCriticManagerBasedRLEnv", "MultiCriticRewardManager"]
