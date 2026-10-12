"""Manager-based environment that exports grouped rewards to RSL-RL."""

from __future__ import annotations

from typing import Any

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.managers import CommandManager, CurriculumManager, TerminationManager

from .reward_manager import MultiCriticRewardManager


class MultiCriticManagerBasedRLEnv(ManagerBasedRLEnv):
    """Use :class:`MultiCriticRewardManager` and expose its output through ``extras``."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs: Any) -> None:
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def load_managers(self) -> None:
        # This follows ManagerBasedRLEnv.load_managers; only the reward manager type changes.
        self.command_manager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        ManagerBasedEnv.load_managers(self)

        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        self.reward_manager = MultiCriticRewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        self._configure_gym_env_spaces()
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def step(self, action: torch.Tensor):
        result = super().step(action)
        extras = result[-1]
        # process_env_step clones the tensor before the next environment step mutates this buffer.
        extras["group_rewards"] = self.reward_manager.group_rewards
        extras["reward_group_names"] = self.reward_manager.reward_group_names
        return result


__all__ = ["MultiCriticManagerBasedRLEnv"]
