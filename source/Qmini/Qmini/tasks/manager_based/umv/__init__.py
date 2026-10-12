import gymnasium as gym

from . import agents


# =============================================================================
# UMV - Flat terrain
# =============================================================================
#
# python scripts/rsl_rl/train.py --task=UMV-Flat-v0 --num_envs=1 --visualizer kit
#
# =============================================================================


# ---------------------------------- Train ------------------------------------

gym.register(
    id="UMV-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UMVFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


# ------------------------------ Play / Evaluation ----------------------------

gym.register(
    id="UMV-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UMVFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
