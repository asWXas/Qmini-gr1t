"""Left-right reflection for GR1T2 PPO observations and actions.

Isaac Lab concatenates the history of each observation term before the next
term.  The transform therefore uses the observation manager's term dimensions
instead of treating the policy vector as a stack of complete frames.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from Qmini.asset.gr1t import JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_JOINT_INDEX = [
    JOINT_NAMES.index(("r_" if name.startswith("l_") else "l_") + name[2:])
    if name.startswith(("l_", "r_"))
    else JOINT_NAMES.index(name)
    for name in JOINT_NAMES
]
_JOINT_SIGN = [-1.0 if name.endswith(("_roll", "_yaw")) else 1.0 for name in JOINT_NAMES]

_VECTOR_SIGNS = {
    "base_lin_vel": (1, -1, 1),
    "base_ang_vel": (-1, 1, -1),
    "torso_ang_vel": (-1, 1, -1),
    "base_projected_gravity": (1, -1, 1),
    "torso_projected_gravity": (1, -1, 1),
    "velocity_commands": (1, -1, -1),
}
_FEET_TERMS = {"air_time", "feet_height", "contact_forces"}
_PHASE_TERMS = {"gait_phase", "gait_phase_h", "gait_phase_l"}


def _mirror_joints(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] != len(JOINT_NAMES):
        raise ValueError(f"Expected {len(JOINT_NAMES)} GR1T2 joints, got {values.shape[-1]}")
    indices = torch.tensor(_JOINT_INDEX, device=values.device)
    signs = values.new_tensor(_JOINT_SIGN)
    return values.index_select(-1, indices) * signs


def _mirror_height_scan(env: ManagerBasedRLEnv, values: torch.Tensor) -> torch.Tensor:
    pattern = env.unwrapped.cfg.scene.height_scanner.pattern_cfg
    nx = math.floor(pattern.size[0] / pattern.resolution + 0.5) + 1
    ny = math.floor(pattern.size[1] / pattern.resolution + 0.5) + 1
    if values.shape[-1] != nx * ny:
        raise ValueError(f"Height scan has {values.shape[-1]} rays; expected {nx * ny}")
    if pattern.ordering == "xy":
        return values.reshape(-1, ny, nx).flip(1).reshape_as(values)
    if pattern.ordering == "yx":
        return values.reshape(-1, nx, ny).flip(2).reshape_as(values)
    raise ValueError(f"Unsupported height scan ordering: {pattern.ordering}")


def _mirror_term(env: ManagerBasedRLEnv, name: str, values: torch.Tensor) -> torch.Tensor:
    if name in _VECTOR_SIGNS:
        signs = _VECTOR_SIGNS[name]
        if values.shape[-1] != len(signs):
            raise ValueError(f"Observation {name} must have {len(signs)} values per frame")
        return values * values.new_tensor(signs)
    if name in {"joint_pos", "joint_vel", "actions"}:
        return _mirror_joints(values)
    if name in _FEET_TERMS:
        if values.shape[-1] != 2:
            raise ValueError(f"Observation {name} must contain left and right feet")
        return values.flip(-1)
    if name == "height_scan":
        return _mirror_height_scan(env, values)
    if name in _PHASE_TERMS:
        if values.shape[-1] != 2:
            raise ValueError(f"Observation {name} must contain sine and cosine")
        return values.clone()
    raise ValueError(f"No left-right reflection defined for observation {name}")


def _mirror_group(env: ManagerBasedRLEnv, group_name: str, values: torch.Tensor) -> torch.Tensor:
    manager = env.unwrapped.observation_manager
    names = manager.active_terms[group_name]
    shapes = manager.group_obs_term_dim[group_name]
    mirrored = values.clone()
    offset = 0
    for name, shape in zip(names, shapes, strict=True):
        width = math.prod(shape)
        base_width = (
            len(JOINT_NAMES)
            if name in {"joint_pos", "joint_vel", "actions"}
            else (
                len(_VECTOR_SIGNS[name])
                if name in _VECTOR_SIGNS
                else (2 if name in _FEET_TERMS | _PHASE_TERMS else width)
            )
        )
        if width % base_width:
            raise ValueError(f"Observation {group_name}/{name} has incompatible width {width}")
        term = values[:, offset : offset + width].reshape(-1, base_width)
        mirrored[:, offset : offset + width] = _mirror_term(env, name, term).reshape(-1, width)
        offset += width
    if offset != values.shape[-1]:
        raise ValueError(f"Observation {group_name} has {values.shape[-1]} values, expected {offset}")
    return mirrored


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Return original and reflected samples in that order for RSL-RL PPO."""
    obs_aug = None
    if obs is not None:
        obs_aug = obs.repeat(2)
        batch_size = obs.batch_size[0]
        for group_name in obs.keys():
            obs_aug[group_name][batch_size:] = _mirror_group(env, group_name, obs[group_name])

    actions_aug = None
    if actions is not None:
        actions_aug = torch.cat((actions, _mirror_joints(actions)), dim=0)
    return obs_aug, actions_aug
