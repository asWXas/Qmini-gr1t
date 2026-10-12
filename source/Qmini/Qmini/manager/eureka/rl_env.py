"""Reusable Eureka interfaces layered on Isaac Lab's ManagerBasedRLEnv."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


class EurekaManagerBasedRLEnv(ManagerBasedRLEnv):
    """Inherit simulation, managers and resets; expose Eureka's reward hooks.

    Override eureka_observation_group to select another noise-free, history-free
    concatenated vector group. Tasks implement _update_eureka_metrics and
    compute_eureka_success; the base class assumes no robot or command names.
    """

    eureka_observation_group = "critic"

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self._configure_eureka_observations()
        # Save the original bound method BEFORE installing the hook.
        self._manager_reward_compute = self.reward_manager.compute
        self.reward_manager.compute = self._compute_eureka_reward

    def _compute_eureka_reward(self, dt: float) -> torch.Tensor:
        """Called by inherited step before resets and command updates."""
        self.eureka_obs = self.observation_manager.compute_group(
            self.eureka_observation_group, update_history=False
        )
        self.eureka_state = {name: self.eureka_obs[:, bounds] for name, bounds in self.eureka_slices.items()}
        self._update_eureka_metrics()
        return self._get_rewards()

    def _get_rewards(self) -> torch.Tensor:
        """Original Manager reward, saved as oracle by official Eureka.

        Generated functions return per-step rewards: multiply rates by step_dt
        exactly once. This original Manager computation already applies dt.
        """
        return self._manager_reward_compute(dt=self.step_dt)

    def _get_observations(self) -> dict:
        """State contract consumed by Eureka's prompt.

        Use self.eureka_state in _get_rewards_eureka(self): named slices of
        self.eureka_obs, computed BEFORE rewards and resets. Values retain the
        selected group's scaling and clipping. The appended schema documents
        actual fields, slices, functions and scales. Divide by scale to recover
        pre-scale values; clipping cannot be undone, relative values stay relative.
        Only use fields in the schema. Never modify state in-place or call
        _get_observations from reward code. Return (total, terms_dict), each
        tensor shaped [self.num_envs]. Multiply reward rates and logged terms
        by self.step_dt exactly once.
        """
        return self.observation_manager.compute(update_history=False)

    def _update_eureka_metrics(self) -> None:
        """Task extension point: accumulate evaluation metrics before reset."""

    def compute_eureka_success(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Override in a task subclass, or use a different TASKS_CFG expression."""
        raise NotImplementedError("Define a task-specific Eureka success metric.")

    def _configure_eureka_observations(self) -> None:
        """Derive state slices and prompt metadata from the selected group."""
        manager = self.observation_manager
        name = self.eureka_observation_group
        if name not in manager.active_terms:
            raise ValueError(f"Missing Eureka observation group: {name}")
        group = getattr(self.cfg.observations, name)
        if not group.concatenate_terms or group.enable_corruption:
            raise ValueError(f"Eureka group {name} must be concatenated and noise-free.")
        self.eureka_slices = {}
        self.eureka_scales = {}
        self.eureka_clips = {}
        self._eureka_schema = []
        offset = 0
        # Resolved term configs are currently internal to Isaac Lab.
        for term_name, dim, term in zip(
            manager.active_terms[name], manager.group_obs_term_dim[name], manager._group_obs_term_cfgs[name]
        ):
            if len(dim) != 1 or term.history_length or term.modifiers:
                raise ValueError(f"Eureka term {term_name} must be a history-free vector without modifiers.")
            stop = offset + dim[0]
            self.eureka_slices[term_name] = slice(offset, stop)
            scale = term.scale if term.scale is not None else 1.0
            self.eureka_scales[term_name] = scale
            self.eureka_clips[term_name] = term.clip
            func_name = getattr(term.func, "__name__", type(term.func).__name__)
            self._eureka_schema.append(
                f"{term_name}: self.eureka_obs[:, {offset}:{stop}], scale={scale}, clip={term.clip}, "
                f"function={term.func.__module__}.{func_name}, params={term.params}; "
                f"meaning={(term.func.__doc__ or '').strip().split(chr(10))[0]}"
            )
            offset = stop
        if manager.group_obs_dim[name] != (offset,):
            raise ValueError(f"Eureka group {name} must be a flat concatenated vector.")

    def get_eureka_state_schema(self) -> str:
        """Describe the actual selected observation layout for the LLM."""
        return f"State group: {self.eureka_observation_group} (values are scaled)\n" + "\n".join(self._eureka_schema)
