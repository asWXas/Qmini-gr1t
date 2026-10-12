# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal cart-pole task demonstrating the Multi-Critic PPO integration."""

import gymnasium as gym

from . import agents


_ENTRY_POINT = "Qmini.manager.multi_critic.manger.rl_env:MultiCriticManagerBasedRLEnv"


def _register_task(task_id: str) -> None:
    gym.register(
        id=task_id,
        entry_point=_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.qmini_env_cfg:QminiEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
            # Scalar reward compatibility is retained, so the same environment can
            # still be used by SKRL. Multi-critic learning itself is RSL-RL only.
            "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        },
    )


# Keep the original template id and provide an explicit Multi-Critic alias.
_register_task("Template-Qmini-v0")
_register_task("Template-Qmini-MultiCritic-v0")


__all__ = []
