from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import ContactSensor


def body_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Angular velocity of selected bodies expressed in their local frames. Returns: [num_envs, 3 * num_bodies]."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    body_ang_vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids]
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    
    body_ang_vel_b = math_utils.quat_apply_inverse(body_quat_w, body_ang_vel_w)
    return body_ang_vel_b.reshape(env.num_envs, -1)


def body_projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity direction projected into selected body frames. Returns: [num_envs, 3 * num_bodies]."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    gravity_dir_w = asset.data.GRAVITY_VEC_W.unsqueeze(1)
    
    # IMPORTANT: quat_apply_inverse does not automatically expand the vector across multiple bodies.
    gravity_dir_w = gravity_dir_w.expand(-1, body_quat_w.shape[1], -1)
    projected_gravity = math_utils.quat_apply_inverse(body_quat_w, gravity_dir_w)
    
    return projected_gravity.reshape(env.num_envs, -1)


def air_time(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Current air time of selected feet. Returns: [num_envs, num_feet]."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_air_time = contact_sensor.data.current_air_time

    if current_air_time is None:
        raise RuntimeError(
            "Contact sensor air-time tracking is disabled. "
            "Set track_air_time=True in ContactSensorCfg."
        )

    air_time_val = current_air_time[:, sensor_cfg.body_ids]
    return air_time_val.reshape(env.num_envs, -1)

def contact_force_violation(
    env: ManagerBasedEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """输出接触力超过 threshold 的幅值。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    force_norm = torch.linalg.norm(net_contact_forces, dim=-1)
    max_force = torch.max(force_norm, dim=1).values

    return (max_force - threshold).clamp_min(0.0)

def feet_height(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, sensor_names: list[str]) -> torch.Tensor:
    """Foot clearance relative to local terrain. Returns: [num_envs, num_feet]."""
    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    terrain_heights = []

    for foot_id, sensor_name in enumerate(sensor_names):
        scanner = env.scene.sensors[sensor_name]
        ray_hit_z = scanner.data.ray_hits_w[..., 2]

        # Ray caster may return inf for missed rays.
        valid = torch.isfinite(ray_hit_z)
        ray_hit_z = torch.where(valid, ray_hit_z, torch.full_like(ray_hit_z, torch.nan))
        
        terrain_z = torch.nanmedian(ray_hit_z, dim=1).values

        # If all rays missed, make clearance 0 instead of NaN.
        terrain_z = torch.where(torch.isfinite(terrain_z), terrain_z, feet_z[:, foot_id])
        terrain_heights.append(terrain_z)

    terrain_z = torch.stack(terrain_heights, dim=-1)
    feet_clearance = feet_z - terrain_z

    # Numerical penetration -> 0
    feet_clearance = torch.clamp(feet_clearance, min=0.0)
    
    return feet_clearance.reshape(env.num_envs, -1)
