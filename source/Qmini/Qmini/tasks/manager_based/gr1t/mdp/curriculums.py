# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.curriculums import modify_env_param
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainImporter

# --------------------------------------------------------------------------
class modify_term_cfg_Q(modify_env_param):
    """修改 manager term cfg，并返回当前实际值用于 Curriculum 日志。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        # commands.xxx -> command_manager.cfg.xxx
        self._address = self._address.replace("s.", "_manager.cfg.", 1)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        address: str,
        modify_fn: callable,
        modify_params: dict | None = None,
    ):
        # 先执行原来的修改逻辑
        super().__call__(
            env=env,
            env_ids=env_ids,
            address=address,
            modify_fn=modify_fn,
            modify_params=modify_params,
        )

        # 重新读取修改后的真实值
        current_value = self._get_fn()

        # 这个返回值就是 Curriculum/stand_to_walk
        return current_value
# --------------------------------------------------------------------------

def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.linalg.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.linalg.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


#########################################

def override_rel_standing_envs(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data: float,
    min_value: float = 0.15,
    start_step: int = 200,
):
    """survival_rate 越高，rel_standing_envs 越低。前 start_step 步跳过更新。"""
    # 无需更新的环境或还在 warmup 阶段，直接跳过
    if len(env_ids) == 0 or env.common_step_counter < start_step:
        return modify_term_cfg_Q.NO_CHANGE

    term_idx = env.termination_manager.active_terms.index("time_out")
    survival_rate = env.termination_manager._last_episode_dones[:, term_idx].float().mean().item()
    # print(f"Curriculum: survival_rate={survival_rate:.3f}")
    target_value = max(min_value, 1-survival_rate)

    # 只允许下降 (单调递减)
    if target_value < data:
        return target_value

    return modify_term_cfg_Q.NO_CHANGE


