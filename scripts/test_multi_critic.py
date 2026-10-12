"""CPU-only shape and optimization tests for the RSL-RL multi-critic path."""

from __future__ import annotations

import importlib.util
import sys
import types
from copy import deepcopy
from pathlib import Path

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

ALGO_DIR = Path(__file__).resolve().parents[1] / "source/Qmini/Qmini/manager/multi_critic/algo"


def _load_algorithm_modules():
    # Avoid importing Qmini.__init__: it registers simulator tasks and therefore needs a launched Isaac Sim app.
    for package_name in (
        "Qmini",
        "Qmini.manager",
        "Qmini.manager.multi_critic",
        "Qmini.manager.multi_critic.algo",
    ):
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = []
            sys.modules[package_name] = package

    def load(module_name: str, filename: str):
        spec = importlib.util.spec_from_file_location(module_name, ALGO_DIR / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    storage_module = load("Qmini.manager.multi_critic.algo.rollout_storage", "rollout_storage.py")
    ppo_module = load("Qmini.manager.multi_critic.algo.multi_critic_ppo", "multi_critic_ppo.py")
    return storage_module.MultiCriticRolloutStorage, ppo_module.MultiCriticPPO


def _load_reward_manager():
    """Load the grouped manager against the small part of Isaac's manager contract it extends."""

    class ManagerTermBase:
        pass

    class RewardTermCfg:
        def __init__(self, func, weight, params=None):
            self.func = func
            self.weight = weight
            self.params = params or {}

    class RewardManager:
        def __init__(self, cfg, env):
            self.cfg = deepcopy(cfg)
            self._env = env
            self._term_names = []
            self._term_cfgs = []
            self._class_term_cfgs = []
            self._prepare_terms()
            self._episode_sums = {name: torch.zeros(self.num_envs, device=self.device) for name in self._term_names}
            self._reward_buf = torch.zeros(self.num_envs, device=self.device)
            self._step_reward = torch.zeros(self.num_envs, len(self._term_names), device=self.device)

        @property
        def num_envs(self):
            return self._env.num_envs

        @property
        def device(self):
            return self._env.device

        def _resolve_common_term_cfg(self, *_args, **_kwargs):
            pass

        def reset(self, env_ids=None):
            env_ids = slice(None) if env_ids is None else env_ids
            result = {}
            for name, values in self._episode_sums.items():
                result[f"Episode_Reward/{name}"] = values[env_ids].mean()
                values[env_ids] = 0.0
            return result

    isaaclab = types.ModuleType("isaaclab")
    managers = types.ModuleType("isaaclab.managers")
    managers.ManagerTermBase = ManagerTermBase
    managers.RewardManager = RewardManager
    managers.RewardTermCfg = RewardTermCfg
    isaaclab.managers = managers
    sys.modules["isaaclab"] = isaaclab
    sys.modules["isaaclab.managers"] = managers

    package_name = "Qmini.manager.multi_critic.manger"
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    module_name = f"{package_name}.reward_manager"
    path = ALGO_DIR.parent / "manger/reward_manager.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.MultiCriticRewardManager, RewardTermCfg


def _make_algorithm(num_envs=4, rollout_length=5):
    storage_class, algorithm_class = _load_algorithm_modules()
    names = ["task", "aesthetic", "constraint"]
    obs = TensorDict({"policy": torch.randn(num_envs, 6)}, batch_size=[num_envs])
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MLPModel(
        obs,
        obs_groups,
        "actor",
        2,
        hidden_dims=[8],
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    critic = MLPModel(obs, obs_groups, "critic", len(names), hidden_dims=[8])
    storage = storage_class("rl", num_envs, rollout_length, obs, [2], reward_groups=names, device="cpu")
    algorithm = algorithm_class(
        actor,
        critic,
        storage,
        critic_groups={
            "task": {"weight": 0.60},
            "aesthetic": {"weight": 0.15},
            "constraint": {"weight": 0.25},
        },
        num_learning_epochs=1,
        num_mini_batches=1,
        device="cpu",
    )
    return algorithm, obs, names


def test_storage_critic_and_mixed_advantage_shapes():
    algorithm, obs, names = _make_algorithm()
    storage = algorithm.storage
    assert algorithm.critic(obs).shape == (4, 3)
    assert storage.group_rewards.shape == (5, 4, 3)
    assert storage.values.shape == (5, 4, 3)
    assert storage.returns.shape == (5, 4, 3)
    assert storage.advantages.shape == (5, 4, 3)
    assert storage.reward_group_names == names
    assert algorithm.mix_advantages(torch.randn(20, 3)).shape == (20,)


def test_nested_reward_manager_preserves_scalar_reward_and_group_order():
    manager_class, reward_term_cfg = _load_reward_manager()

    class Env:
        num_envs = 4
        device = "cpu"
        max_episode_length_s = 2.0

    def constant_reward(env, value):
        return torch.full((env.num_envs,), value)

    config = {
        "task": {
            "tracking": reward_term_cfg(constant_reward, 2.0, {"value": 1.0}),
            "bonus": reward_term_cfg(constant_reward, 0.5, {"value": 2.0}),
        },
        "aesthetic": {},
        "constraint": {
            "penalty": reward_term_cfg(constant_reward, -1.0, {"value": 3.0}),
        },
    }
    manager = manager_class(config, Env())
    total = manager.compute(dt=0.1)

    assert manager.reward_group_names == ["task", "aesthetic", "constraint"]
    assert manager.num_reward_groups == 3
    assert manager.group_rewards.shape == (4, 3)
    torch.testing.assert_close(total, manager.group_rewards.sum(dim=-1))
    torch.testing.assert_close(manager.group_rewards[0], torch.tensor([0.3, 0.0, -0.3]))
    logs = manager.reset()
    assert all(f"Reward/{name}" in logs for name in manager.reward_group_names)


def test_rollout_gae_and_update_keep_heads_independent():
    algorithm, final_obs, names = _make_algorithm()
    for _ in range(algorithm.storage.num_transitions_per_env):
        obs = TensorDict({"policy": torch.randn(4, 6)}, batch_size=[4])
        algorithm.act(obs)
        group_rewards = torch.randn(4, 3)
        algorithm.process_env_step(
            obs,
            group_rewards.sum(dim=-1),
            torch.zeros(4, dtype=torch.long),
            {"group_rewards": group_rewards, "reward_group_names": names},
        )

    algorithm.compute_returns(final_obs)
    assert algorithm.storage.returns.shape == (5, 4, 3)
    assert algorithm.storage.advantages.shape == (5, 4, 3)
    torch.testing.assert_close(algorithm.storage.advantages.mean(dim=(0, 1)), torch.zeros(3), atol=1e-6, rtol=0)

    losses = algorithm.update()
    assert all(f"Reward/{name}" in losses for name in names)
    assert all(f"ValueLoss/{name}" in losses for name in names)
    assert all(f"CriticWeight/{name}" not in losses for name in names)


def test_group_order_and_weight_validation_are_strict():
    algorithm, obs, names = _make_algorithm(num_envs=2, rollout_length=1)
    algorithm.act(obs)
    try:
        algorithm.process_env_step(
            obs,
            torch.zeros(2),
            torch.zeros(2, dtype=torch.long),
            {
                "group_rewards": torch.zeros(2, 3),
                "reward_group_names": list(reversed(names)),
            },
        )
    except ValueError as exc:
        assert "expected:" in str(exc) and "received:" in str(exc)
    else:
        raise AssertionError("Mismatched group ordering was not rejected.")

    _, algorithm_class = _load_algorithm_modules()
    try:
        algorithm_class._parse_critic_groups({"task": {"weight": -1.0}})
    except ValueError:
        pass
    else:
        raise AssertionError("Negative critic weight was not rejected.")
