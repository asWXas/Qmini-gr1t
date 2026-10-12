from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.utils import resolve_nn_activation


class _OnnxTCNModel(nn.Module):
    """ONNX export wrapper for the history-latent TCN model.

    Input is the same flattened observation used by RSL-RL:
        [B, sum(H * D_group)]

    For every observation group we first restore its own time dimension,
    concatenate groups along the feature dimension, then split into:
        past    = frames [0 : H-1]
        current = frame  [H-1]

    Only ``past`` is encoded by the TCN. The MLP receives:
        concat(current_frame, history_latent)
    """

    is_recurrent: bool = False

    def __init__(self, model: TCN, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.history_length = model.history_length
        self.frame_obs_dim = model.frame_obs_dim
        self.input_size = model.obs_dim

        # Needed to reconstruct multiple observation groups exactly as in training.
        self.group_frame_dims = list(model.group_frame_dims)
        self.group_flat_dims = list(model.group_flat_dims)

        self.tcn_obs_normalizer = copy.deepcopy(model.tcn_obs_normalizer)
        self.tcn = copy.deepcopy(model.tcn)
        self.mlp = copy.deepcopy(model.mlp)

        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else nn.Identity()
        )

    def _build_history_tensor_from_flat(self, x: torch.Tensor) -> torch.Tensor:
        """[B, sum(H*Dg)] -> [B, H, sum(Dg)]."""
        history_parts: list[torch.Tensor] = []
        offset = 0

        for flat_dim, frame_dim in zip(self.group_flat_dims, self.group_frame_dims):
            group_x = x[:, offset : offset + flat_dim]
            group_history = group_x.reshape(x.shape[0], self.history_length, frame_dim)
            history_parts.append(group_history)
            offset += flat_dim

        return torch.cat(history_parts, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, sum(H*Dg)] -> [B, H, D]
        x = self._build_history_tensor_from_flat(x)
        x = self.tcn_obs_normalizer(x)

        # Past frames only: [B, H-1, D]
        history = x[:, :-1, :]

        # Current frame remains uncompressed: [B, D]
        current = x[:, -1, :]

        # [B, H-1, D] -> [B, D, H-1] -> TCN -> [B, C, H-1]
        history_features = self.tcn(history.transpose(1, 2))

        # Last causal output summarizes all usable past frames: [B, C]
        history_latent = history_features[:, :, -1]

        # [B, D] + [B, C] -> [B, D+C] -> MLP -> actions
        latent = torch.cat((current, history_latent), dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class _TorchTCNModel(nn.Module):
    """TorchScript export wrapper matching the training-time TCN data flow."""

    is_recurrent: bool = False

    def __init__(self, model: TCN) -> None:
        super().__init__()
        self.history_length = model.history_length
        self.frame_obs_dim = model.frame_obs_dim

        self.group_frame_dims = list(model.group_frame_dims)
        self.group_flat_dims = list(model.group_flat_dims)

        self.tcn_obs_normalizer = copy.deepcopy(model.tcn_obs_normalizer)
        self.tcn = copy.deepcopy(model.tcn)
        self.mlp = copy.deepcopy(model.mlp)

        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else nn.Identity()
        )

    def _build_history_tensor_from_flat(self, x: torch.Tensor) -> torch.Tensor:
        """[B, sum(H*Dg)] -> [B, H, sum(Dg)]."""
        history_parts: list[torch.Tensor] = []
        offset = 0

        for flat_dim, frame_dim in zip(self.group_flat_dims, self.group_frame_dims):
            group_x = x[:, offset : offset + flat_dim]
            group_history = group_x.reshape(x.shape[0], self.history_length, frame_dim)
            history_parts.append(group_history)
            offset += flat_dim

        return torch.cat(history_parts, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, sum(H*Dg)] -> [B, H, D]
        x = self._build_history_tensor_from_flat(x)
        x = self.tcn_obs_normalizer(x)

        # Past H-1 frames -> TCN history latent.
        history = x[:, :-1, :]
        history_features = self.tcn(history.transpose(1, 2))
        history_latent = history_features[:, :, -1]

        # Current frame bypasses the TCN.
        current = x[:, -1, :]

        # Current state + temporal history summary -> MLP.
        latent = torch.cat((current, history_latent), dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        pass


class CausalConv1d(nn.Module):
    """Causal 1-D convolution using only the present/past of its input sequence."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        if kernel_size <= 0:
            raise ValueError("kernel_size must be > 0.")
        if dilation <= 0:
            raise ValueError("dilation must be > 0.")

        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Residual temporal convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        activation: str,
    ) -> None:
        super().__init__()

        self.conv = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        activation_module = resolve_nn_activation(activation)
        if activation_module is None:
            raise ValueError(f"Unsupported activation: {activation}")

        self.activation: nn.Module = activation_module

        if in_channels == out_channels:
            self.residual: nn.Module = nn.Identity()
        else:
            self.residual = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = y + self.residual(x)
        return self.activation(y)


class TCN(MLPModel):
    """History-latent TCN + MLP Actor/Critic.

    Observation semantics
    ---------------------
    Every observation group must contain a whole-frame history stack:

        group_g: [B, H * D_g]

    Every group is restored independently and groups are then concatenated on
    the feature axis:

        [B, H, D_1] + ... + [B, H, D_n]
            -> [B, H, D]

    The latest frame is *not* compressed by the TCN:

        past frames [t-H+1, ..., t-1] -> TCN -> history_latent [B, C]
        current frame [t]                         -> current       [B, D]

        concat(current, history_latent) -> [B, D+C] -> MLP -> output

    This keeps current proprioception directly accessible to the policy while
    the TCN is used only as a temporal encoder for the previous H-1 frames.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        tcn_cfg: dict[str, Any] | None = None,
    ) -> None:
        if tcn_cfg is None:
            tcn_cfg = {}

        # ------------------------------------------------------------------
        # 1. TCN configuration.
        #
        # These attributes must exist before super().__init__(), because
        # MLPModel constructs its MLP using self._get_latent_dim().
        # ------------------------------------------------------------------
        self.history_length = int(tcn_cfg.get("history_length", 5))
        self.tcn_channels = [int(v) for v in tcn_cfg.get("channels", [64])]
        self.kernel_size = int(tcn_cfg.get("kernel_size", 4))
        self.dilations = [int(v) for v in tcn_cfg.get("dilations", [1])]

        if self.history_length < 2:
            raise ValueError(
                "history_length must be >= 2 because the current frame bypasses "
                "the TCN and at least one past frame is required."
            )
        if len(self.tcn_channels) == 0:
            raise ValueError("tcn_cfg['channels'] cannot be empty.")
        if len(self.tcn_channels) != len(self.dilations):
            raise ValueError(
                "tcn_cfg['channels'] and tcn_cfg['dilations'] must have the same length."
            )
        if any(v <= 0 for v in self.tcn_channels):
            raise ValueError("All TCN channels must be > 0.")
        if any(v <= 0 for v in self.dilations):
            raise ValueError("All TCN dilations must be > 0.")

        self.tcn_output_dim = self.tcn_channels[-1]

        # ------------------------------------------------------------------
        # 2. Resolve observation groups and compute per-frame dimensions
        # BEFORE calling MLPModel.__init__().
        #
        # This is required because _get_latent_dim() now returns D + C.
        # ------------------------------------------------------------------
        if obs_set not in obs_groups:
            raise KeyError(
                f"Observation set '{obs_set}' not found in obs_groups. "
                f"Available sets: {list(obs_groups.keys())}"
            )

        selected_groups = list(obs_groups[obs_set])
        if len(selected_groups) == 0:
            raise ValueError(f"Observation set '{obs_set}' contains no observation groups.")

        self.history_obs_groups = selected_groups
        self.group_frame_dims: list[int] = []
        self.group_flat_dims: list[int] = []

        for group in self.history_obs_groups:
            if group not in obs.keys():
                raise KeyError(
                    f"Observation group '{group}' from obs_groups['{obs_set}'] "
                    "does not exist in the provided TensorDict."
                )

            group_dim = int(obs[group].shape[-1])

            if group_dim % self.history_length != 0:
                raise ValueError(
                    f"Observation group '{group}' has dim={group_dim}, which cannot "
                    f"be divided by history_length={self.history_length}."
                )

            frame_dim = group_dim // self.history_length
            self.group_flat_dims.append(group_dim)
            self.group_frame_dims.append(frame_dim)

        self.frame_obs_dim = sum(self.group_frame_dims)

        # ------------------------------------------------------------------
        # 3. Parent MLPModel.
        #
        # Parent responsibilities:
        #   - observation-group bookkeeping
        #   - action distribution
        #   - MLP head
        #
        # The parent normalizer is disabled because normalization is handled
        # frame-wise by tcn_obs_normalizer.
        # ------------------------------------------------------------------
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=False,
            distribution_cfg=distribution_cfg,
        )

        # Validate that our pre-super observation calculation agrees with RSL-RL.
        expected_obs_dim = self.frame_obs_dim * self.history_length
        if expected_obs_dim != self.obs_dim:
            raise ValueError(
                "TCN observation dimension mismatch: "
                f"expected={expected_obs_dim}, MLPModel obs_dim={self.obs_dim}."
            )

        # ------------------------------------------------------------------
        # 4. Shared per-frame observation normalization.
        # ------------------------------------------------------------------
        self.tcn_obs_normalization = obs_normalization

        if obs_normalization:
            self.tcn_obs_normalizer: nn.Module = EmpiricalNormalization(self.frame_obs_dim)
        else:
            self.tcn_obs_normalizer = nn.Identity()

        # ------------------------------------------------------------------
        # 5. TCN temporal encoder.
        #
        # Only H-1 past frames are fed into this network.
        # Example recommended setting for H=5:
        #   channels=[64], kernel_size=4, dilations=[1]
        # ------------------------------------------------------------------
        layers: list[nn.Module] = []
        in_channels = self.frame_obs_dim

        for out_channels, dilation in zip(self.tcn_channels, self.dilations):
            layers.append(
                TCNBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    activation=activation,
                )
            )
            in_channels = out_channels

        self.tcn = nn.Sequential(*layers)
        self._init_tcn_weights()

    def _get_latent_dim(self) -> int:
        """MLP input = current-frame features + TCN history latent."""
        return self.frame_obs_dim + self.tcn_output_dim

    def _build_history_tensor(self, obs: TensorDict) -> torch.Tensor:
        """Restore whole-frame history as [B, H, D].

        Each group is reshaped independently before concatenation. This avoids
        mixing time indices when multiple flattened history groups are used.
        """
        history_parts: list[torch.Tensor] = []

        for group, frame_dim, flat_dim in zip(
            self.history_obs_groups,
            self.group_frame_dims,
            self.group_flat_dims,
        ):
            group_obs = obs[group]

            if group_obs.ndim != 2:
                raise ValueError(
                    "TCN expects 2-D flattened observation groups, but "
                    f"'{group}' has shape {tuple(group_obs.shape)}."
                )

            if group_obs.shape[-1] != flat_dim:
                raise ValueError(
                    f"Observation group '{group}' changed dimension: "
                    f"got={group_obs.shape[-1]}, expected={flat_dim}."
                )

            # [B, H * Dg] -> [B, H, Dg]
            group_history = group_obs.reshape(
                group_obs.shape[0],
                self.history_length,
                frame_dim,
            )
            history_parts.append(group_history)

        if len(history_parts) == 0:
            raise RuntimeError("TCN received no observation groups.")

        # [B, H, D1 + D2 + ...]
        return torch.cat(history_parts, dim=-1)

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        # Full whole-frame stack: [B, H, D]
        x = self._build_history_tensor(obs)

        # Shared feature normalization for every frame.
        # "Current raw obs" here means uncompressed current features; if
        # obs_normalization=True they are normalized just like past frames.
        x = self.tcn_obs_normalizer(x)

        # --------------------------------------------------------------
        # Past H-1 frames -> TCN temporal latent.
        # --------------------------------------------------------------
        history = x[:, :-1, :]  # [B, H-1, D]
        history_features = self.tcn(history.transpose(1, 2))  # [B, C, H-1]
        history_latent = history_features[:, :, -1]  # [B, C]

        # --------------------------------------------------------------
        # Current frame bypasses TCN.
        # --------------------------------------------------------------
        current = x[:, -1, :]  # [B, D]

        # --------------------------------------------------------------
        # MLP latent = current state + temporal history summary.
        # --------------------------------------------------------------
        return torch.cat((current, history_latent), dim=-1)  # [B, D+C]

    def update_normalization(self, obs: TensorDict) -> None:
        if not self.tcn_obs_normalization:
            return

        # [B, H, D] -> [B*H, D]
        # The same physical feature shares normalization statistics across time.
        x = self._build_history_tensor(obs).reshape(-1, self.frame_obs_dim)

        if isinstance(self.tcn_obs_normalizer, EmpiricalNormalization):
            self.tcn_obs_normalizer.update(x)

    def _init_tcn_weights(self) -> None:
        for module in self.tcn.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxTCNModel(self, verbose)

    def as_jit(self) -> nn.Module:
        return _TorchTCNModel(self)


# -----------------------------------------------------------------------------
# Recommended TCN configuration for the current design
# -----------------------------------------------------------------------------
#
# history_length = 5 means:
#   past    : t-4, t-3, t-2, t-1  -> TCN
#   current : t                    -> direct MLP input
#
# A single kernel_size=4 layer sees all four past frames at its last output.
#
# tcn_cfg = {
#     "history_length": 5,
#     "channels": [64],
#     "kernel_size": 4,
#     "dilations": [1],
# }
#
# If frame_obs_dim = 47:
#   history input : [B, 47, 4]
#   TCN latent    : [B, 64]
#   current frame : [B, 47]
#   MLP input     : [B, 111]