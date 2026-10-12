"""Performance-gated sampling curriculum for the three LineRides skills."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class LineRidesSkillCurriculum(ManagerTermBase):
    """Promote 01 -> 01+10 -> 01+10+11 using completed guidelines.

    A skill attempt succeeds only after its final waypoint is reached. Outcomes
    are collected at environment reset, before the command term clears them.
    Earlier skills retain nonzero sampling weight at every subsequent stage.
    """

    _weights = (
        (1.0, 0.0, 0.0),
        (0.7, 0.3, 0.0),
        (0.45, 0.45, 0.1),
        (1.0, 1.0, 1.0),
    )
    _target_modes = (1, 2, 3)

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.command = env.command_manager.get_term(cfg.params["command_name"])
        if self.command.enabled_skill_modes.tolist() != [1, 2, 3]:
            raise ValueError("LineRides curriculum requires skills 01, 10, and 11.")
        self.stage = cfg.params.get("start_stage", 0)
        if type(self.stage) is not int or not 0 <= self.stage < len(self._weights):
            raise ValueError("start_stage must be an integer in [0, 3].")
        self._apply_weights()
        self._attempts = torch.zeros(4, dtype=torch.long, device=env.device)
        self._successes = torch.zeros_like(self._attempts)
        self._window_attempts = 0
        self._window_successes = 0
        self._passed_windows = 0
        self._last_window_rate = 0.0

    def _apply_weights(self):
        weights = torch.tensor(self._weights[self.stage], device=self.device)
        self.command.skill_sampling_weights.copy_(weights)

    def reset(self, env_ids=None):
        """Keep global curriculum statistics across individual environment resets."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        command_name: str = "linerides",
        start_stage: int = 0,
        min_attempts: int = 256,
        success_thresholds: tuple[float, float, float] = (0.6, 0.5, 0.4),
        consecutive_windows: int = 2,
    ) -> dict[str, float]:
        """Measure current skill on disjoint windows of completed attempts."""
        if type(min_attempts) is not int or type(consecutive_windows) is not int:
            raise ValueError("min_attempts and consecutive_windows must be integers.")
        if min_attempts < 1 or consecutive_windows < 1:
            raise ValueError("min_attempts and consecutive_windows must be positive.")
        if len(success_thresholds) != 3 or any(not 0.0 <= value <= 1.0 for value in success_thresholds):
            raise ValueError("success_thresholds must contain three probabilities in [0, 1].")

        self._attempts += self.command.episode_skill_attempts[env_ids].sum(dim=0)
        self._successes += self.command.episode_skill_successes[env_ids].sum(dim=0)

        if self.stage < len(self._target_modes):
            mode = self._target_modes[self.stage]
            self._window_attempts += int(self.command.episode_skill_attempts[env_ids, mode].sum().item())
            self._window_successes += int(self.command.episode_skill_successes[env_ids, mode].sum().item())
            if self._window_attempts >= min_attempts:
                rate = self._window_successes / self._window_attempts
                self._last_window_rate = rate
                self._passed_windows = self._passed_windows + 1 if rate >= success_thresholds[self.stage] else 0
                self._window_attempts = 0
                self._window_successes = 0
                if self._passed_windows >= consecutive_windows:
                    self.stage += 1
                    self._apply_weights()
                    self._attempts.zero_()
                    self._successes.zero_()
                    self._passed_windows = 0

        state = {
            "stage": float(self.stage + 1),
            "target_mode": float(self._target_modes[self.stage]) if self.stage < 3 else 0.0,
            "window_attempts": float(self._window_attempts),
            "last_window_completion": self._last_window_rate,
            "passed_windows": float(self._passed_windows),
        }
        for mode in self._target_modes:
            attempts = int(self._attempts[mode].item())
            state[f"attempts_{mode:02b}"] = float(attempts)
            state[f"completion_{mode:02b}"] = float(self._successes[mode].item() / attempts) if attempts else 0.0
            state[f"weight_{mode:02b}"] = float(self.command.skill_sampling_weights[mode - 1].item())
        return state
