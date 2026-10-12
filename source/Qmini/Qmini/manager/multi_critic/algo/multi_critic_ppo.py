"""PPO with a shared critic backbone, K value outputs, and one actor."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict

from .rollout_storage import MultiCriticRolloutStorage


class MultiCriticPPO(PPO):
    """PPO whose actor uses a weighted mixture of independently computed GAE heads."""

    def __init__(
        self,
        *args,
        critic_groups: Mapping[str, Mapping[str, float]],
        normalize_critic_weights: bool = False,
        rnd_group: str | None = None,
        **kwargs,
    ) -> None:
        group_names, weights = self._parse_critic_groups(critic_groups)
        super().__init__(*args, **kwargs)

        self.critic_groups = dict(critic_groups)
        self.critic_group_names = group_names
        self.num_critics = len(group_names)
        self.critic_weights = torch.tensor(weights, dtype=torch.float, device=self.device)
        if normalize_critic_weights:
            self.critic_weights /= self.critic_weights.sum()

        if not isinstance(self.storage, MultiCriticRolloutStorage):
            raise TypeError(
                f"MultiCriticPPO requires MultiCriticRolloutStorage, received {type(self.storage).__name__}."
            )
        self._validate_group_names(self.storage.reward_group_names, "rollout storage")
        self.transition = MultiCriticRolloutStorage.Transition()

        self.rnd_group = rnd_group
        self.rnd_group_index: int | None = None
        if rnd_group is not None and rnd_group not in self.critic_group_names:
            raise ValueError(f"Unknown rnd_group {rnd_group!r}; expected one of {self.critic_group_names}.")
        if self.rnd is not None:
            if rnd_group is None:
                raise ValueError(
                    "RND is enabled for MultiCriticPPO, so algorithm.rnd_group must name exactly one "
                    f"critic group. Available groups: {self.critic_group_names}."
                )
            self.rnd_group_index = self.critic_group_names.index(rnd_group)

    @staticmethod
    def _parse_critic_groups(
        critic_groups: Mapping[str, Mapping[str, float]],
    ) -> tuple[list[str], list[float]]:
        if not critic_groups:
            raise ValueError("critic_groups must contain at least one group.")
        group_names = list(critic_groups.keys())
        if len(set(group_names)) != len(group_names):
            raise ValueError(f"critic_groups contains duplicate names: {group_names}.")

        weights: list[float] = []
        for name, group_cfg in critic_groups.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"Critic group names must be non-empty strings, received {name!r}.")
            if not isinstance(group_cfg, Mapping) or "weight" not in group_cfg:
                raise ValueError(f"critic_groups[{name!r}] must be a mapping containing 'weight'.")
            try:
                weight = float(group_cfg["weight"])
            except (TypeError, ValueError) as exc:
                raise TypeError(f"critic_groups[{name!r}]['weight'] must be numeric.") from exc
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"critic_groups[{name!r}]['weight'] must be finite and >= 0, received {weight}.")
            weights.append(weight)
        if not any(weight > 0.0 for weight in weights):
            raise ValueError("At least one critic group weight must be greater than zero.")
        return group_names, weights

    def _validate_group_names(self, received: Sequence[str], source: str) -> None:
        received_names = list(received)
        if received_names != self.critic_group_names:
            raise ValueError(
                f"Reward group names/order from {source} do not match critic_groups.\n"
                f"expected: {self.critic_group_names}\nreceived: {received_names}"
            )

    def _validate_last_dim(self, tensor: torch.Tensor, name: str) -> None:
        if tensor.ndim == 0 or tensor.shape[-1] != self.num_critics:
            raise ValueError(
                f"{name} must have last dimension {self.num_critics} for groups "
                f"{self.critic_group_names}, received shape {tuple(tensor.shape)}."
            )

    def _normalize_multi_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        self._validate_last_dim(advantages, "advantages")
        reduction_dims = tuple(range(advantages.ndim - 1))
        mean = advantages.mean(dim=reduction_dims, keepdim=True)
        std = advantages.std(dim=reduction_dims, keepdim=True, correction=0)
        return (advantages - mean) / (std + 1e-8)

    def mix_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        """Return one actor advantage while preserving all K critic targets."""
        self._validate_last_dim(advantages, "advantages")
        return (advantages * self.critic_weights).sum(dim=-1)

    def act(self, obs: TensorDict) -> torch.Tensor:
        actions = super().act(obs)
        self._validate_last_dim(self.transition.values, "critic values")  # type: ignore[arg-type]
        return actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, Any],
    ) -> None:
        """Store scalar rewards and obtain true group rewards from environment extras."""
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        if "group_rewards" not in extras:
            raise KeyError(
                "MultiCriticPPO expected extras['group_rewards']; use MultiCriticManagerBasedRLEnv "
                "or an equivalent environment adapter."
            )
        if "reward_group_names" not in extras:
            raise KeyError("MultiCriticPPO expected extras['reward_group_names'] for order validation.")
        self._validate_group_names(extras["reward_group_names"], "environment extras")

        group_rewards = extras["group_rewards"].to(self.device).clone()
        expected_shape = (self.storage.num_envs, self.num_critics)
        if tuple(group_rewards.shape) != expected_shape:
            raise ValueError(
                f"group_rewards has shape {tuple(group_rewards.shape)}; expected {expected_shape} "
                f"for groups {self.critic_group_names}."
            )
        if rewards.ndim not in (1, 2) or rewards.shape[0] != self.storage.num_envs:
            raise ValueError(
                f"Scalar environment rewards must have shape [N] or [N, 1], received {tuple(rewards.shape)}."
            )
        if rewards.ndim == 2 and rewards.shape[1] != 1:
            raise ValueError(f"Scalar environment rewards received non-scalar shape {tuple(rewards.shape)}.")

        self.transition.rewards = rewards.clone()
        self.transition.group_rewards = group_rewards
        self.transition.dones = dones

        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            intrinsic = self.intrinsic_rewards
            if intrinsic.ndim == 2 and intrinsic.shape[-1] == 1:
                intrinsic = intrinsic.squeeze(-1)
            if tuple(intrinsic.shape) != (self.storage.num_envs,):
                raise ValueError(
                    f"RND intrinsic reward must have shape [N] or [N, 1], received "
                    f"{tuple(self.intrinsic_rewards.shape)}."
                )
            self.transition.group_rewards[:, self.rnd_group_index] += intrinsic  # type: ignore[index]

        if "time_outs" in extras:
            time_outs = extras["time_outs"].to(self.device).reshape(-1, 1)
            self.transition.group_rewards += self.gamma * self.transition.values * time_outs  # type: ignore[operator]

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute vectorized, independent GAE for all K critic heads."""
        st = self.storage
        last_values = self.critic(obs).detach()
        self._validate_last_dim(last_values, "last critic values")
        self._validate_last_dim(st.group_rewards, "stored group_rewards")
        self._validate_last_dim(st.values, "stored values")

        advantage = torch.zeros_like(last_values)
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.group_rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]

        st.advantages = st.returns - st.values
        self._validate_last_dim(st.returns, "returns")
        self._validate_last_dim(st.advantages, "advantages")
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = self._normalize_multi_advantages(st.advantages)

    def update(self) -> dict[str, float]:
        """Apply standard PPO using mixed actor advantages and K-head value losses."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_value_loss_per_head = torch.zeros(self.num_critics, device=self.device)
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None
        mean_group_rewards = self.storage.group_rewards.mean(dim=(0, 1)).detach().cpu()

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]
            self._validate_last_dim(batch.values, "batch values")
            self._validate_last_dim(batch.returns, "batch returns")
            self._validate_last_dim(batch.advantages, "batch advantages")

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = self._normalize_multi_advantages(batch.advantages)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"], obs=batch.observations, actions=batch.actions
                )
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            self._validate_last_dim(values, "critic batch output")
            distribution_params = tuple(
                parameter[:original_batch_size] for parameter in self.actor.output_distribution_params
            )
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif 0.0 < kl_mean < self.desired_kl / 2.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            mixed_advantages = self.mix_advantages(batch.advantages)
            ratio = torch.exp(actions_log_prob - batch.old_actions_log_prob.squeeze(-1))
            surrogate = -mixed_advantages * ratio
            surrogate_clipped = -mixed_advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                selected_value_losses = torch.max(value_losses, value_losses_clipped)
            else:
                selected_value_losses = (batch.returns - values).pow(2)
            value_loss = selected_value_losses.mean()
            head_reduction_dims = tuple(range(selected_value_losses.ndim - 1))
            mean_value_loss_per_head += selected_value_losses.mean(dim=head_reduction_dims).detach()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )
                mean_actions = self.actor(batch.observations.detach().clone())
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                symmetry_loss = nn.MSELoss()(
                    mean_actions[original_batch_size:],
                    actions_mean_symm.detach()[original_batch_size:],
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                rnd_loss = nn.MSELoss()(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict: dict[str, float] = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
        }
        if mean_rnd_loss is not None:
            loss_dict["rnd"] = mean_rnd_loss / num_updates
        if mean_symmetry_loss is not None:
            loss_dict["symmetry"] = mean_symmetry_loss / num_updates
        for index, group_name in enumerate(self.critic_group_names):
            loss_dict[f"Reward/{group_name}"] = float(mean_group_rewards[index])
            loss_dict[f"ValueLoss/{group_name}"] = float(mean_value_loss_per_head[index] / num_updates)

        self.storage.clear()
        return loss_dict

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> MultiCriticPPO:
        """Build K from critic_groups and validate the environment's exact group ordering."""
        algorithm_cfg = cfg["algorithm"]
        critic_groups = algorithm_cfg.get("critic_groups")
        group_names, _ = MultiCriticPPO._parse_critic_groups(critic_groups)

        reward_manager = getattr(getattr(env, "unwrapped", env), "reward_manager", None)
        environment_group_names = getattr(reward_manager, "reward_group_names", None)
        if environment_group_names is None:
            raise ValueError(
                "MultiCriticPPO requires an environment with reward_manager.reward_group_names. "
                "Use MultiCriticManagerBasedRLEnv or provide an equivalent adapter."
            )
        if list(environment_group_names) != group_names:
            raise ValueError(
                "Reward group names/order from RewardManager do not match critic_groups.\n"
                f"expected: {group_names}\nreceived: {list(environment_group_names)}"
            )

        algorithm_class = resolve_callable(algorithm_cfg.pop("class_name"))
        actor_class = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class = resolve_callable(cfg["critic"].pop("class_name"))

        default_sets = ["actor", "critic"]
        if algorithm_cfg.get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(algorithm_cfg, obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns
        critic = critic_class(obs, cfg["obs_groups"], "critic", len(group_names), **cfg["critic"]).to(device)
        print(f"Multi-Critic Model ({len(group_names)} heads: {group_names}): {critic}")

        storage = MultiCriticRolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            reward_groups=group_names,
            device=device,
        )
        algorithm = algorithm_class(
            actor,
            critic,
            storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        return algorithm


__all__ = ["MultiCriticPPO"]
