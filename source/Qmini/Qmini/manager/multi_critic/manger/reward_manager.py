"""Reward manager supporting semantic groups without changing scalar rewards."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from prettytable import PrettyTable

from isaaclab.managers import ManagerTermBase, RewardManager, RewardTermCfg


class MultiCriticRewardManager(RewardManager):
    """Parse either legacy flat rewards or two-level grouped reward configs.

    A flat reward config remains valid and is exposed as one ``"default"`` group.
    Grouped configs expose their top-level field names in declaration order.
    """

    DEFAULT_GROUP_NAME = "default"

    def __init__(self, cfg: object, env) -> None:
        # These lists are filled by _prepare_terms(), which RewardManager.__init__ calls.
        self._reward_group_names: list[str] = []
        self._term_group_indices: list[int] = []
        super().__init__(cfg, env)

        if not self._reward_group_names:
            raise ValueError("Reward configuration must define at least one reward group.")
        self._group_reward_buf = torch.zeros(
            (self.num_envs, self.num_reward_groups), dtype=torch.float, device=self.device
        )
        self._group_episode_sums = torch.zeros_like(self._group_reward_buf)

    @property
    def group_rewards(self) -> torch.Tensor:
        """Current weighted reward sums, shaped ``[num_envs, num_reward_groups]``."""
        return self._group_reward_buf

    @property
    def reward_group_names(self) -> list[str]:
        """Reward group names in the same order as the last tensor dimension."""
        return list(self._reward_group_names)

    @property
    def num_reward_groups(self) -> int:
        return len(self._reward_group_names)

    def __str__(self) -> str:
        msg = (
            f"<MultiCriticRewardManager> contains {len(self._term_names)} active terms "
            f"in {self.num_reward_groups} groups.\n"
        )
        table = PrettyTable()
        table.title = "Active Grouped Reward Terms"
        table.field_names = ["Index", "Group", "Name", "Weight"]
        table.align["Group"] = "l"
        table.align["Name"] = "l"
        table.align["Weight"] = "r"
        for index, (name, term_cfg, group_index) in enumerate(
            zip(self._term_names, self._term_cfgs, self._term_group_indices)
        ):
            table.add_row([index, self._reward_group_names[group_index], name, term_cfg.weight])
        return msg + table.get_string() + "\n"

    def compute(self, dt: float) -> torch.Tensor:
        """Compute scalar total and K grouped rewards from each term exactly once."""
        self._reward_buf.zero_()
        self._group_reward_buf.zero_()

        for term_idx, (name, term_cfg, group_idx) in enumerate(
            zip(self._term_names, self._term_cfgs, self._term_group_indices)
        ):
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue

            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            self._reward_buf += value
            self._group_reward_buf[:, group_idx] += value
            self._episode_sums[name] += value
            self._group_episode_sums[:, group_idx] += value
            self._step_reward[:, term_idx] = value / dt

        return self._reward_buf

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Reset terms and publish per-term plus per-group episode metrics."""
        selected_ids = slice(None) if env_ids is None else env_ids
        extras = super().reset(env_ids)
        for group_idx, group_name in enumerate(self._reward_group_names):
            episodic_sum_avg = torch.mean(self._group_episode_sums[selected_ids, group_idx])
            extras[f"Reward/{group_name}"] = episodic_sum_avg / self._env.max_episode_length_s
            self._group_episode_sums[selected_ids, group_idx] = 0.0
        return extras

    def _prepare_terms(self) -> None:
        top_level_items = list(self._cfg_items(self.cfg))
        direct_terms = [(name, value) for name, value in top_level_items if isinstance(value, RewardTermCfg)]
        nested_groups = [
            (name, value)
            for name, value in top_level_items
            if value is not None and not isinstance(value, RewardTermCfg)
        ]

        if direct_terms and nested_groups:
            raise TypeError(
                "Reward configuration cannot mix top-level RewardTermCfg entries with nested reward groups. "
                "Move all terms into groups or keep the legacy config completely flat."
            )

        if direct_terms or not nested_groups:
            self._reward_group_names.append(self.DEFAULT_GROUP_NAME)
            for term_name, term_cfg in direct_terms:
                self._register_term(self.DEFAULT_GROUP_NAME, 0, term_name, term_cfg)
            return

        for group_name, group_cfg in nested_groups:
            if not isinstance(group_name, str) or not group_name:
                raise ValueError(f"Reward group names must be non-empty strings, received {group_name!r}.")
            if group_name in self._reward_group_names:
                raise ValueError(f"Duplicate reward group name: {group_name!r}.")
            group_idx = len(self._reward_group_names)
            self._reward_group_names.append(group_name)
            for term_name, term_cfg in self._cfg_items(group_cfg):
                if term_cfg is None:
                    continue
                if not isinstance(term_cfg, RewardTermCfg):
                    raise TypeError(
                        f"Configuration for reward term '{group_name}.{term_name}' must be RewardTermCfg, "
                        f"received {type(term_cfg)}. Only two reward config levels are supported."
                    )
                self._register_term(group_name, group_idx, term_name, term_cfg)

    @staticmethod
    def _cfg_items(cfg: object) -> Iterable[tuple[str, object]]:
        if isinstance(cfg, dict):
            return cfg.items()
        if not hasattr(cfg, "__dict__"):
            raise TypeError(f"Reward configuration must be a dict or config object, received {type(cfg)}.")
        return cfg.__dict__.items()

    def _register_term(
        self,
        group_name: str,
        group_idx: int,
        term_name: str,
        term_cfg: RewardTermCfg,
    ) -> None:
        if not isinstance(term_cfg.weight, (float, int)):
            raise TypeError(
                f"Weight for reward term '{group_name}.{term_name}' must be float or int, "
                f"received {type(term_cfg.weight)}."
            )
        qualified_name = f"{group_name}/{term_name}"
        self._resolve_common_term_cfg(qualified_name, term_cfg, min_argc=1)
        self._term_names.append(qualified_name)
        self._term_cfgs.append(term_cfg)
        self._term_group_indices.append(group_idx)
        if isinstance(term_cfg.func, ManagerTermBase):
            self._class_term_cfgs.append(term_cfg)


__all__ = ["MultiCriticRewardManager"]
