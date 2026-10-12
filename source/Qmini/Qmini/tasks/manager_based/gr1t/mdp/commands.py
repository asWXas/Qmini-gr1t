"""Four-mode velocity commands for GR1T locomotion."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class FourModeVelocityCommand(UniformVelocityCommand):
    """Sample four motion modes plus an explicit zero command."""

    cfg: FourModeVelocityCommandCfg

    def __init__(self, cfg: FourModeVelocityCommandCfg, env: ManagerBasedEnv):
        probabilities = (
            cfg.prob_x_only,
            cfg.prob_y_only,
            cfg.prob_yaw_only,
            cfg.prob_mixed,
            cfg.prob_zero,
        )
        if any(not math.isfinite(probability) or probability < 0.0 for probability in probabilities):
            raise ValueError("Four velocity-mode probabilities must be finite and non-negative.")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6):
            raise ValueError("Four velocity-mode probabilities must sum to 1.0.")
        if cfg.heading_command or cfg.rel_standing_envs != 0.0:
            raise ValueError("Four-mode sampling requires heading_command=False and rel_standing_envs=0.0.")

        super().__init__(cfg, env)
        self._mode_probabilities = torch.tensor(probabilities, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return
        self.is_standing_env[env_ids] = False

        # Parent samples each component from its configured range. Mask the
        # unused components after drawing one mode independently per env.
        modes = torch.multinomial(self._mode_probabilities, len(env_ids), replacement=True)
        commands = self.vel_command_b[env_ids]
        commands[modes == 0, 1:] = 0.0  # x only
        commands[modes == 1, 0] = 0.0  # y only
        commands[modes == 1, 2] = 0.0
        commands[modes == 2, :2] = 0.0  # yaw only
        commands[modes == 4, :] = 0.0  # explicit zero command
        # Mode 3 keeps all three sampled components.
        self.vel_command_b[env_ids] = commands


@configclass
class FourModeVelocityCommandCfg(UniformVelocityCommandCfg):
    """Probabilities for x-only, y-only, yaw-only, and mixed modes."""

    class_type: type = FourModeVelocityCommand

    prob_x_only: float = 0.25
    prob_y_only: float = 0.25
    prob_yaw_only: float = 0.25
    prob_mixed: float = 0.25
    prob_zero: float = 0.0
