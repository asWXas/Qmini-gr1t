import gymnasium as gym

from . import agents

# =============================================================================
# Train
# =============================================================================

gym.register(
    id="Qmini-Flat-TCN",
    entry_point="Qmini.manager.rl_env:CustomManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QminiFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_tcn_cfg:TCNRunnerCfg",
    },
)

gym.register(
    id="Qmini-Flat-Play-TCN",
    entry_point="Qmini.manager.rl_env:CustomManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QminiFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_tcn_cfg:TCNRunnerCfg",
    },
)



# python scripts/rsl_rl/train.py --task=Qmini-Flat-v0 --num_envs=1 --visualizer kit
# =============================================================================
# Play / Evaluation
# =============================================================================

gym.register(
    id="Qmini-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QminiFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Qmini-Flat-Eureka-v0",
    entry_point=f"{__name__}.env.eureka_env:GR1TEurekaEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QminiFlatEurekaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Qmini-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QminiFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
