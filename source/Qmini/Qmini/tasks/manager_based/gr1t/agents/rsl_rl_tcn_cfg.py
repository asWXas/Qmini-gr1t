from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlMLPModelCfg
)
@configclass
class TCNCfg:
    """Configuration for the temporal convolution network."""
    history_length: int = 6           # Number of observation history frames
    channels: list[int] = [128, 128]  # Output channels of each TCN block
    kernel_size: int = 3              # Temporal convolution kernel size
    dilations: list[int] = [1, 2]     # Dilation of each TCN block


@configclass
class RslRlTCNModelCfg(RslRlMLPModelCfg):
    """Configuration for the TCN + MLP model."""

    class_name = "Qmini.tasks.manager_based.gr1t.algo.tcn_models:TCN"
    tcn_cfg: TCNCfg = TCNCfg()  # Temporal convolution configuration
    

@configclass
class TCNRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 150
    experiment_name = "flat_tcn"
    actor = RslRlTCNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,

        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
        ),

        tcn_cfg=TCNCfg(
            history_length=5,
            channels=[64],
            kernel_size=5,
            dilations=[1],
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
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
        symmetry_cfg=None,
    )
