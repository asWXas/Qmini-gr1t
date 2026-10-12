# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg

from Qmini.manager.multi_critic.cfg import RslRlMultiCriticPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL runner for the three-head Multi-Critic cart-pole template."""

    class_name = "OnPolicyRunner"
    num_steps_per_env = 16
    max_iterations = 3000
    save_interval = 100
    experiment_name = "qmini_multi_critic_template"
    empirical_normalization = False
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlMultiCriticPpoAlgorithmCfg(
        # This insertion order must exactly match RewardsCfg in qmini_env_cfg.py.
        critic_groups={
            "task": {"weight": 0.60},
            "aesthetic": {"weight": 0.15},
            "constraint": {"weight": 0.25},
        },
        normalize_critic_weights=False,
        rnd_group=None,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        rnd_cfg=None,
        symmetry_cfg=None,
    )
