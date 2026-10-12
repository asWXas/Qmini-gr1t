"""GR1T evaluation metric for the reusable Eureka Manager environment."""

from __future__ import annotations

import torch
from Qmini.manager.eureka import EurekaManagerBasedRLEnv


class GR1TEurekaEnv(EurekaManagerBasedRLEnv):
    """Score velocity tracking over a full episode, penalizing early falls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in ("velocity_commands", "base_lin_vel", "base_ang_vel"):
            if name not in self.eureka_slices:
                raise ValueError(f"GR1T success metric requires observation {name}.")
            scale = self.eureka_scales[name]
            if self.eureka_clips[name] is not None or bool(torch.any(torch.as_tensor(scale) == 0)):
                raise ValueError(f"GR1T success metric requires unclipped, nonzero-scaled {name}.")
        self._tracking_sum = torch.zeros(self.num_envs, device=self.device)

    def _update_eureka_metrics(self) -> None:
        command = self.eureka_state["velocity_commands"] / self.eureka_scales["velocity_commands"]
        lin_vel = self.eureka_state["base_lin_vel"] / self.eureka_scales["base_lin_vel"]
        ang_vel = self.eureka_state["base_ang_vel"] / self.eureka_scales["base_ang_vel"]
        lin_error = (command[:, :2] - lin_vel[:, :2]).square().sum(dim=1)
        yaw_error = (command[:, 2] - ang_vel[:, 2]).square()
        self._tracking_sum += torch.exp(-lin_error / 0.25) * torch.exp(-yaw_error / 0.25)

    def compute_eureka_success(self, env_ids: torch.Tensor) -> torch.Tensor:
        return (self._tracking_sum[env_ids] / self.max_episode_length).mean()

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # Official Eureka collects success before invoking this original reset.
        super()._reset_idx(env_ids)
        if hasattr(self, "_tracking_sum"):
            self._tracking_sum[env_ids] = 0
