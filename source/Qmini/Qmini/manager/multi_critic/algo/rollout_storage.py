"""Rollout storage for PPO with one value head per reward group."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict


class MultiCriticRolloutStorage(RolloutStorage):
    """Keep scalar environment rewards and vector-valued critic targets side by side."""

    class Transition(RolloutStorage.Transition):
        """A transition augmented with the independently grouped rewards."""

        def __init__(self) -> None:
            super().__init__()
            self.group_rewards: torch.Tensor | None = None
            """Reward for every critic group, with shape ``[num_envs, num_reward_groups]``."""

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        reward_groups: Sequence[str],
        device: str = "cpu",
    ) -> None:
        self.reward_group_names = list(reward_groups)
        if not self.reward_group_names:
            raise ValueError("reward_groups must contain at least one group name.")
        if len(set(self.reward_group_names)) != len(self.reward_group_names):
            raise ValueError(f"reward_groups contains duplicate names: {self.reward_group_names}.")
        if any(not isinstance(name, str) or not name for name in self.reward_group_names):
            raise ValueError(f"Every reward group name must be a non-empty string: {self.reward_group_names}.")

        self.num_reward_groups = len(self.reward_group_names)
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)

        self.group_rewards = torch.zeros(
            num_transitions_per_env,
            num_envs,
            self.num_reward_groups,
            device=self.device,
        )
        if training_type == "rl":
            target_shape = (num_transitions_per_env, num_envs, self.num_reward_groups)
            self.values = torch.zeros(target_shape, device=self.device)
            self.returns = torch.zeros(target_shape, device=self.device)
            self.advantages = torch.zeros(target_shape, device=self.device)

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        """Validate and store a scalar reward plus the K group rewards."""
        group_rewards = getattr(transition, "group_rewards", None)
        if group_rewards is None:
            raise ValueError("Multi-critic transitions must provide transition.group_rewards.")
        expected = (self.num_envs, self.num_reward_groups)
        if tuple(group_rewards.shape) != expected:
            raise ValueError(
                f"transition.group_rewards has shape {tuple(group_rewards.shape)}; expected {expected} "
                f"for groups {self.reward_group_names}."
            )
        if self.training_type == "rl":
            values = transition.values
            if values is None or tuple(values.shape) != expected:
                received = None if values is None else tuple(values.shape)
                raise ValueError(f"transition.values has shape {received}; expected {expected}.")

        step = self.step
        super().add_transition(transition)
        self.group_rewards[step].copy_(group_rewards)


__all__ = ["MultiCriticRolloutStorage"]
