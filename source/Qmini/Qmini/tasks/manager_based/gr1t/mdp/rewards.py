# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GR1T2 人形机器人行走奖励函数集合。

设计目标: 让机器人以 **协调、稳定、接近人类** 的步态行走。

内容组织:
    0. 通用工具函数       (_smoothstep / _command_speed / _command_norm1 ...)
    1. 躯干与基座稳定性   (直立 / 高度 / 速度阻尼 / 侧向摇摆)
    2. 步态与足部         (腾空时间 / 抬脚高度 / 打滑 / 绊脚 / 对称性 / 腾空相)
    3. 关节协调           (腿三关节协调 / 手臂对侧摆臂)
    4. 姿态与能耗         (站立姿态 / 腰 / 髋 / 手臂姿态)
    5. 观测用             (步态相位)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor, RayCaster


# =============================================================================
# 0. 通用工具函数
# =============================================================================


def _smoothstep(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """把 value 从 [lower, upper] 平滑映射到 [0, 1] (三次 smoothstep)。"""
    x = torch.clamp((value - lower) / max(upper - lower, 1e-6), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _command_speed(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """指令水平速度大小 |vx, vy|, shape [num_envs]。"""
    command = env.command_manager.get_command(command_name)
    return torch.linalg.norm(command[:, :2], dim=1)


def _command_norm1(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """指令 L1 范数 |vx| + |vy| + |wz|, shape [num_envs]。"""
    command = env.command_manager.get_command(command_name)
    return torch.linalg.norm(command[:, :3], ord=1, dim=1)


def _projected_gravity_body(asset: Articulation, body_ids) -> torch.Tensor:
    """指定刚体坐标系下的重力方向, shape [num_envs, num_bodies, 3]。"""
    quat_w = asset.data.body_quat_w[:, body_ids]
    gravity_w = asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, quat_w.shape[1], -1)
    return quat_apply_inverse(quat_w, gravity_w)


def command_smooth_scale(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    max_speed: float = 0.15,
) -> torch.Tensor:
    """指令平滑开关: 移动速度在 0 ~ max_speed 之间由 0 平滑过渡到 1。"""
    return _smoothstep(_command_speed(env, command_name), 0.0, max_speed)


# =============================================================================
# 1. 躯干与基座稳定性
# =============================================================================



def base_mothion_up_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """躯干直立奖励 (指数核) + 倒立惩罚。

    - 直立: 投影重力 xy 分量 ≈ 0 -> 奖励接近 1
    - 倒立: 投影重力 z > 0 -> 额外惩罚

    未指定 body_names 时使用 root(基座)的投影重力;
    指定 body_names 时使用对应刚体坐标系下的投影重力 (例如躯干 link_waist_roll)。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if asset_cfg.body_names is not None:
        # 使用目标刚体自身坐标系下的投影重力 (修复原先 torso 与 base 完全等效的问题)
        gravity_b = _projected_gravity_body(asset, asset_cfg.body_ids).mean(dim=1)
    else:
        gravity_b = asset.data.projected_gravity_b

    # 直立奖励: gx, gy 接近 0
    tilt_error = torch.sum(gravity_b[:, :2].square(), dim=1)
    upright_reward = torch.exp(-tilt_error / (std**2))

    # 倒立惩罚: gz > 0
    inverted_penalty = 2.0 * torch.clamp(gravity_b[:, 2], min=0.0)

    return upright_reward - inverted_penalty


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    legs_cfg: SceneEntityCfg,
    waist_cfg: SceneEntityCfg,
    arms_cfg: SceneEntityCfg,
    legs_weight: float = 0.8,
    waist_weight: float = 0.6,
    arms_weight: float = 0.8,
    
    stand_speed: float = 0.05,
    move_speed: float = 0.15,
) -> torch.Tensor:
    """站立姿态 L1 惩罚 (随指令大小平滑淡出)。

    指令接近 0 (站立) 时惩罚机器人偏离默认站姿; 指令增大 (开始行走) 时平滑关闭,
    从而与行走奖励不冲突。
    """
    asset: Articulation = env.scene[legs_cfg.name]

    # 指令 L1 范数: |vx| + |vy| + |wz|
    command_norm = _command_norm1(env, command_name)

    # 站立 -> 行走 平滑过渡 (<= stand_speed: 1, >= move_speed: 0)
    blend = 1.0 - _smoothstep(command_norm, stand_speed, move_speed)

    # Joint positions
    pos = asset.data.joint_pos
    default_pos = asset.data.default_joint_pos

    # L1 posture penalties
    legs_penalty = torch.linalg.norm(pos[:, legs_cfg.joint_ids] - default_pos[:, legs_cfg.joint_ids], ord=1, dim=1)
    waist_penalty = torch.linalg.norm(pos[:, waist_cfg.joint_ids] - default_pos[:, waist_cfg.joint_ids], ord=1, dim=1)
    arms_penalty = torch.linalg.norm(pos[:, arms_cfg.joint_ids] - default_pos[:, arms_cfg.joint_ids], ord=1, dim=1)

    # Group weighting
    posture_penalty = (legs_weight * legs_penalty) + (waist_weight * waist_penalty) + (arms_weight * arms_penalty)

    return blend * posture_penalty

def base_height_penalty(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """惩罚基座高度偏离目标高度 (L2 平方核)。

    注意:
        - 平地: target_height 为世界系目标高度。
        - 崎岖地形: 用射线传感器读数修正目标高度。

    重要:
        RayCaster 未命中时返回 inf/NaN。若不清洗, 高度误差会变成 inf,
        该奖励会污染 value function 并直接把训练炸掉。这里只对有限值求均值。
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # 清洗 inf / NaN 射线命中点, 只用有限值求地形高度
        ray_hits_z = sensor.data.ray_hits_w[..., 2]
        finite = torch.isfinite(ray_hits_z)
        safe_z = torch.where(finite, ray_hits_z, torch.zeros_like(ray_hits_z))
        mean_z = safe_z.sum(dim=1) / finite.sum(dim=1).clamp(min=1)
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + mean_z
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)


# =============================================================================
# 2. 步态与足部
# =============================================================================




def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding (linear velocity) when in contact with the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


def feet_stumble_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 4.0,
) -> torch.Tensor:
    """惩罚脚部水平冲击力 (绊脚 / 蹭地 / 拖脚)。

    当脚底水平力显著大于垂直力时判定为绊脚, 促使机器人真正抬脚跨步而非贴地拖行。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]

    force_xy = torch.linalg.norm(forces[..., :2], dim=-1)
    force_z = torch.abs(forces[..., 2])

    return torch.sum(torch.relu(force_xy - threshold * force_z), dim=1)


def flight_phase_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    speed_threshold: float = 0.1,
    running_speed_range: tuple[float, float] | None = None,
    air_time_tolerance: float = 0.02,
) -> torch.Tensor:
    """Discourage walking flight, with contact-noise grace and optional running relief."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    both_in_air = torch.all(contact_time <= 0.0, dim=1).float()
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    flight_gate = _smoothstep(air_time.min(dim=1).values, air_time_tolerance, air_time_tolerance + 0.02)
    speed = _command_speed(env, command_name)
    gate = _smoothstep(speed, speed_threshold, speed_threshold + 0.2)
    if running_speed_range is not None:
        gate = gate * (1.0 - _smoothstep(speed, *running_speed_range))
    return both_in_air * flight_gate * gate


def feet_step_symmetry_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    speed_threshold: float = 0.1,
) -> torch.Tensor:
    """惩罚左右步态不对称。

    在任一只脚触地 (first contact) 的瞬间, 比较左右两侧上一步的腾空时长;
    差值越大说明左右步幅/步频越不对称, 惩罚越大, 从而鼓励对称的自然步态。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids] > 0.0

    asymmetry = torch.abs(last_air_time[:, 0] - last_air_time[:, 1])
    gate = (_command_speed(env, command_name) > speed_threshold).float()
    return asymmetry * torch.any(first_contact, dim=1).float() * gate


def stance_knee_flexion_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    max_stance_flexion: float = 0.25,
    max_fast_flexion: float = 0.45,
    fast_speed: float = 1.5,
    contact_ramp_s: tuple[float, float] = (0.02, 0.08),
    walk_speed: float = 0.3,
    max_transition_flexion: float | None = None,
) -> torch.Tensor:
    """Discourage a crouched support leg while allowing knee flexion in swing.

    GR1T2 knee_pitch is zero when straight and increases when flexed. The
    support penalty rises after touchdown; faster motion allows more knee bend.
    Optional double-support relief leaves room for landing and push-off while
    encouraging extension in single support. This is a contact-based heuristic,
    not a prescribed gait clock. Swing knees remain unconstrained by this term.
    Joint and contact body order must both be left, right.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    knee_angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    speed = _command_speed(env, command_name)
    fast_blend = _smoothstep(speed, walk_speed, fast_speed)
    allowed_flexion = max_stance_flexion + fast_blend * (max_fast_flexion - max_stance_flexion)
    support_weight = _smoothstep(contact_time, *contact_ramp_s)
    allowed_flexion = allowed_flexion.unsqueeze(1)
    if max_transition_flexion is not None:
        other_support = torch.flip(support_weight, dims=[1])
        moving = _smoothstep(speed, 0.05, 0.20).unsqueeze(1)
        relief = torch.relu(max_transition_flexion - allowed_flexion)
        allowed_flexion = allowed_flexion + relief * other_support * moving
    excess_flexion = torch.relu(knee_angle - allowed_flexion)
    # Do not trade crouching for knee hyperextension (negative GR1T2 knee angle).
    hyperextension = torch.relu(-knee_angle)
    return torch.mean(support_weight * (excess_flexion.square() + hyperextension.square()), dim=1)


def swing_knee_flexion_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    min_flexion: float = 0.35,
    min_fast_flexion: float = 0.55,
    speed_range: tuple[float, float] = (1.0, 2.5),
    swing_duration_range: tuple[float, float] = (0.20, 0.50),
    initial_swing_duration: float = 0.30,
) -> torch.Tensor:
    """Discourage a rigid swing knee, without forcing a full joint trajectory.

    Both entity selections must use left/right order. The last completed swing
    estimates each foot's timing (bounded for robustness); this is not a measured
    phase or touchdown prediction. A soft minimum knee bend is active only in
    the estimated middle of swing and fades out to allow extension before landing.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    knee = asset.data.joint_pos[:, asset_cfg.joint_ids]
    air_time = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    last_air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    contact_time = sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    duration = torch.where(last_air_time > 0.0, last_air_time, initial_swing_duration)
    duration = duration.clamp(*swing_duration_range)
    phase = air_time / duration
    mid_swing = _smoothstep(phase, 0.10, 0.30) * (1.0 - _smoothstep(phase, 0.60, 0.90))
    mid_swing = mid_swing * ((air_time > 0.0) & (contact_time <= 0.0))

    speed = _command_speed(env, command_name)
    min_angle = min_flexion + _smoothstep(speed, *speed_range) * (min_fast_flexion - min_flexion)
    deficit = torch.relu(min_angle.unsqueeze(1) - knee)
    return (mid_swing * deficit.square()).mean(dim=1) * _smoothstep(speed, 0.10, 0.30)


def leg_pitch_coordination_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    coordination_weight: float = 0.2,
    participation_weight: float = 1.0,
    swing_activity_weight: float = 2.0,
    knee_activity_weight: float = 2.0,
    min_hip_pos_ratio: float = 0.20,
    min_knee_pos_ratio: float = 0.40,
    min_ankle_pos_ratio: float = 0.10,
    min_swing_activity: float = 0.25,
    min_knee_activity: float = 0.15,
) -> torch.Tensor:
    """摆动腿 Hip-Knee-Ankle 协调与参与度惩罚 (返回正值 penalty，配合 RewTerm weight < 0 使用)。"""
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    eps = 1e-6

    # 1. 关节相对默认姿态的位置偏移
    q = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    l_hip, l_knee, l_ankle = q[:, 0], q[:, 1], q[:, 2]
    r_hip, r_knee, r_ankle = q[:, 3], q[:, 4], q[:, 5]

    # 2. 摆动相检测 (contact_time <= 0 为摆动相)
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    left_swing, right_swing = contact_time[:, 0] <= 0.0, contact_time[:, 1] <= 0.0

    # 3. 串联协调性 (仅对摆动腿软约束: hip + knee + ankle ≈ 0)
    l_coord = torch.where(left_swing, (l_hip + l_knee + l_ankle).square(), 0.0)
    r_coord = torch.where(right_swing, (r_hip + r_knee + r_ankle).square(), 0.0)
    coordination_penalty = 0.5 * (l_coord + r_coord)

    # 4. 各关节活动比例约束 (防止某关节完全不动)
    l_total = l_hip.abs() + l_knee.abs() + l_ankle.abs()
    r_total = r_hip.abs() + r_knee.abs() + r_ankle.abs()

    l_part = (
        torch.relu(min_hip_pos_ratio - l_hip.abs() / (l_total + eps)).square()
        + torch.relu(min_knee_pos_ratio - l_knee.abs() / (l_total + eps)).square()
        + torch.relu(min_ankle_pos_ratio - l_ankle.abs() / (l_total + eps)).square()
    )
    r_part = (
        torch.relu(min_hip_pos_ratio - r_hip.abs() / (r_total + eps)).square()
        + torch.relu(min_knee_pos_ratio - r_knee.abs() / (r_total + eps)).square()
        + torch.relu(min_ankle_pos_ratio - r_ankle.abs() / (r_total + eps)).square()
    )
    participation_penalty = 0.5 * (torch.where(left_swing, l_part, 0.0) + torch.where(right_swing, r_part, 0.0))

    # 5. 摆动腿总活动量约束 (防止摆动腿整体僵直锁死)
    l_activity = torch.where(left_swing, torch.relu(min_swing_activity - l_total).square(), 0.0)
    r_activity = torch.where(right_swing, torch.relu(min_swing_activity - r_total).square(), 0.0)
    swing_activity_penalty = 0.5 * (l_activity + r_activity)

    # 6. 膝关节屈曲约束 (防止摆动时膝关节不弯曲直腿拖地)
    l_knee_act = torch.where(left_swing, torch.relu(min_knee_activity - l_knee.abs()).square(), 0.0)
    r_knee_act = torch.where(right_swing, torch.relu(min_knee_activity - r_knee.abs()).square(), 0.0)
    knee_activity_penalty = 0.5 * (l_knee_act + r_knee_act)

    # 7. 综合加权惩罚并附加指令平滑门控
    penalty = (
        coordination_weight * coordination_penalty
        + participation_weight * participation_penalty
        + swing_activity_weight * swing_activity_penalty
        + knee_activity_weight * knee_activity_penalty
    )
    return penalty * command_smooth_scale(env, command_name)


def leg_joint_symmetry_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    speed_threshold: float = 0.1,
) -> torch.Tensor:
    """惩罚左右腿关节活动量不对称 (防止"一条腿拖着不动" / 瘸腿步态)。

    对称的交替步态中, 任一时刻左右腿相对默认姿态的关节偏角大小应基本相等
    (左腿前摆 a 时右腿后摆约 -a)。因此用两侧关节偏角绝对值之和的差作为惩罚,
    对"一条腿几乎不动"的解给出持续、密集的负反馈。

    asset_cfg.joint_ids 顺序需为:
        [l_hip_pitch, l_knee_pitch, l_ankle_pitch, r_hip_pitch, r_knee_pitch, r_ankle_pitch]
    """
    asset: Articulation = env.scene[asset_cfg.name]

    q = torch.abs(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    left_activity = q[:, 0] + q[:, 1] + q[:, 2]
    right_activity = q[:, 3] + q[:, 4] + q[:, 5]

    penalty = torch.square(left_activity - right_activity)
    gate = (_command_speed(env, command_name) > speed_threshold).float()
    return penalty * gate

def diagonal_arm_follow_hip_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    min_scale: float = 0.30,
    max_scale: float = 0.65,
    min_speed: float = 0.10,
    max_speed: float = 1.00,
    command_threshold: float = 0.1,
    mirror_sign: float = 1.0,
    remove_hip_mean: bool = False,
    position_tolerance: float = 0.0,
) -> torch.Tensor:
    """对侧手臂 - 大腿摆动协调惩罚 (人类行走的核心特征)。

    人类行走时手臂与 **对侧** 大腿同相摆动:
        左肩 pitch 应与右髋 pitch 同相, 右肩 pitch 应与左髋 pitch 同相。
    摆臂幅度随前进速度由 min_scale 平滑增大到 max_scale, 避免站立时乱摆臂。
    返回 >= 0 的偏差值 (RewTerm 中 weight 需为负)。

    asset_cfg.joint_ids 顺序需为: [l_hip_pitch, r_hip_pitch, l_shoulder_pitch, r_shoulder_pitch]
    对 GR1T2_new.urdf，左右 hip_pitch 与 shoulder_pitch 的轴均为局部 +Y。
    计入肩关节的安装旋转后，零位附近正角度仍使手和脚沿机体 -X 方向运动，
    因此对侧手/脚同向摆动时 mirror_sign 应为 +1.0。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 1. 相对默认姿态的关节位置
    q = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    l_hip, r_hip, l_shoulder, r_shoulder = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # 2. 指令与随速度平滑变化的摆动幅度
    command = env.command_manager.get_command(command_name)
    forward_speed = torch.abs(command[:, 0])
    scale = min_scale + _smoothstep(forward_speed, min_speed, max_speed) * (max_scale - min_scale)

    # Optionally use only the alternating component, excluding a shared hip
    # posture offset that would otherwise pull both arms forward/backward.
    if remove_hip_mean:
        hip_mean = 0.5 * (l_hip + r_hip)
        l_hip = l_hip - hip_mean
        r_hip = r_hip - hip_mean

    # 3. 对侧耦合目标: 左肩跟随右髋, 右肩跟随左髋
    target_l_shoulder = mirror_sign * scale * r_hip
    target_r_shoulder = mirror_sign * scale * l_hip

    # 4. 位置偏差惩罚
    pos_penalty = 0.5 * (
        torch.relu((l_shoulder - target_l_shoulder).abs() - position_tolerance).square()
        + torch.relu((r_shoulder - target_r_shoulder).abs() - position_tolerance).square()
    )

    # Lateral stepping does not have the same arm/leg swing pattern as walking forward.
    return pos_penalty * _smoothstep(forward_speed, command_threshold, command_threshold + 0.2)


def foot_clearance_scan_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    command_name: str = "base_velocity",
    left_scanner_name: str = "height_scanner_left",
    right_scanner_name: str = "height_scanner_right",
) -> torch.Tensor:
    """Reward moderate clearance of moving swing feet relative to local terrain."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    left_scanner = env.scene.sensors[left_scanner_name]
    right_scanner = env.scene.sensors[right_scanner_name]

    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    left_hits = left_scanner.data.ray_hits_w[..., 2]
    right_hits = right_scanner.data.ray_hits_w[..., 2]
    left_valid = torch.isfinite(left_hits)
    right_valid = torch.isfinite(right_hits)
    left_ground_z = torch.where(left_valid, left_hits, -torch.inf).max(dim=1).values
    right_ground_z = torch.where(right_valid, right_hits, -torch.inf).max(dim=1).values
    ground_z = torch.stack([left_ground_z, right_ground_z], dim=1)
    valid_ground = torch.isfinite(ground_z)
    ground_z = torch.where(valid_ground, ground_z, foot_z)

    clearance = foot_z - ground_z
    clearance_reward = torch.exp(-torch.square(clearance - target_height) / (std**2))
    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    foot_speed = torch.linalg.norm(foot_vel_xy, dim=2)
    velocity_gate = torch.tanh(tanh_mult * foot_speed)
    swing = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] <= 0.0
    reward = clearance_reward * velocity_gate * swing * valid_ground
    return reward.mean(dim=1) * command_smooth_scale(env, command_name)



def waist_l1_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    hip_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    yaw_gain: float = 0.20,
    max_yaw: float = 0.15,
    pitch_free: float = 0.08,
    roll_free: float = 0.08,
    forward_threshold: float = 0.10,
    forward_full: float = 0.30,
) -> torch.Tensor:
    """Keep the torso balanced while allowing a small gait-coupled waist twist.

    Joint order: waist [yaw, pitch, roll], hips [left_pitch, right_pitch].
    A forward right leg (negative right hip pitch in the GR1T2 URDF) calls
    for negative waist yaw, moving the left side of the torso forward.
    """
    asset = env.scene[asset_cfg.name]
    waist = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    hips = asset.data.joint_pos[:, hip_cfg.joint_ids] - asset.data.default_joint_pos[:, hip_cfg.joint_ids]

    target_yaw = torch.clamp(-yaw_gain * (hips[:, 0] - hips[:, 1]), -max_yaw, max_yaw)
    forward_speed = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    forward_gate = _smoothstep(forward_speed, forward_threshold, forward_full)
    yaw_penalty = torch.abs(waist[:, 0] - target_yaw) * forward_gate

    # Small pitch/roll motions can compensate for a moving support leg;
    # penalize only excessive bending or side leaning.
    tilt_penalty = torch.relu(torch.abs(waist[:, 1]) - pitch_free) + torch.relu(
        torch.abs(waist[:, 2]) - roll_free
    )
    return yaw_penalty + tilt_penalty


def hip_roll_yaw_l1_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """髋部 Roll / Yaw 关节偏离默认姿态的 L1 惩罚 (带移动指令平滑门控)。"""
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    
    # 使用 L1 范数 (ord=1) 计算绝对值之和
    penalty = torch.linalg.norm(q, ord=1, dim=1)
    return penalty * command_smooth_scale(env, command_name)


def joint_mechanical_power_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sum absolute mechanical joint power in watts for the selected motors.

    GR1T2 uses explicit DCMotor actuators, so ``applied_torque`` is the torque
    actually sent to the simulator. RewardManager multiplies this rate by dt.
    Absolute power also charges for braking, rather than cancelling it against
    positive work at another joint.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(torque * velocity), dim=1)


def arm_posture_l1_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    min_speed: float = 0.0,
    max_speed: float = 1.0,
    min_elbow_offset: float = 0.10,
    max_elbow_offset: float = 0.50,
) -> torch.Tensor:
    """手臂姿态 L1 惩罚 (肘部随速度平滑前屈，带统一 CMD 平滑开关)。

    asset_cfg.joint_ids 顺序需为:
        [0: l_shoulder_roll, 1: l_shoulder_yaw, 2: l_elbow_pitch, 3: l_wrist_yaw,
         4: r_shoulder_roll, 5: r_shoulder_yaw, 6: r_elbow_pitch, 7: r_wrist_yaw]
    """
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    command = env.command_manager.get_command(command_name)
    speed = torch.linalg.norm(command[:, :2], dim=1)

    # 肘部前屈 target 随速度平滑插值 (smoothstep)
    x = torch.clamp((speed - min_speed) / (max_speed - min_speed), 0.0, 1.0)
    alpha = x * x * (3.0 - 2.0 * x)
    elbow_offset = min_elbow_offset + alpha * (max_elbow_offset - min_elbow_offset)

    # 左右肘关节 (索引 2, 6) 动态增加前屈 offset
    target = q_default.clone()
    target[:, [2, 6]] += elbow_offset.unsqueeze(-1)

    # L1 姿态惩罚 * 统一速度平滑门控
    penalty = torch.linalg.norm(q - target, ord=1, dim=1)
    return penalty * command_smooth_scale(env, command_name)


def base_motion_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    lin_vel_b: float = 0.8,
    ang_vel_b: float = 0.2,
) -> torch.Tensor:
    """Penalize base vertical and roll/pitch velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return lin_vel_b * torch.square(asset.data.root_lin_vel_b[:, 2]) + ang_vel_b * torch.sum(
        torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1
    )

def body_motion_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    lin_vel_b: float = 0.8,
    ang_vel_b: float = 0.2,
) -> torch.Tensor:
    """惩罚指定刚体自身坐标系下的垂直线速度 (z) 和水平角速度 (roll/pitch)。支持单/多 body。"""
    asset = env.scene[asset_cfg.name]

    # [N, B, 3] & [N, B, 4]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
    body_ang_vel_w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids]
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]

    # World frame -> 各自 Body frame (原生支持多维广播)
    body_lin_vel_b = quat_apply_inverse(body_quat_w, body_lin_vel_w)
    body_ang_vel_b = quat_apply_inverse(body_quat_w, body_ang_vel_w)

    # z 轴线速度平方 (L2) + roll/pitch 角速度绝对值之和 (L1)
    lin_penalty = torch.square(body_lin_vel_b[..., 2])
    ang_penalty = torch.linalg.norm(body_ang_vel_b[..., :2], ord=1, dim=-1)

    # 沿 body 维度求和，返回 [num_envs]
    return torch.sum(lin_vel_b * lin_penalty + ang_vel_b * ang_penalty, dim=1)


def base_lateral_sway_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    lin_vel_weight: float = 1.0,
    roll_rate_weight: float = 0.5,
    roll_angle_weight: float = 0.5,
    max_speed: float = 0.10,
) -> torch.Tensor:
    """抑制基座左右摇摆 (横向速度 + roll 角速度 + 侧倾角)。

    人形机器人在行走时常见 "鸭子步" (左右大幅晃动), 该惩罚可让质心轨迹更接近
    人类的窄支撑、直线行走风格, 同时在站立时通过平滑门控自动关闭。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    commanded_y = env.command_manager.get_command(command_name)[:, 1]
    lin_vel_y_error = asset.data.root_lin_vel_b[:, 1] - commanded_y
    roll_rate = asset.data.root_ang_vel_b[:, 0]
    roll_angle = asset.data.projected_gravity_b[:, 1]

    penalty = (
        lin_vel_weight * lin_vel_y_error.square()
        + roll_rate_weight * roll_rate.square()
        + roll_angle_weight * roll_angle.square()
    )
    return penalty * command_smooth_scale(env, command_name, max_speed=max_speed)


# =============================================================================
# 5. 观测用
# =============================================================================


def gait_phase(
    env: ManagerBasedRLEnv,
    period: float = 0.5,
) -> torch.Tensor:
    """周期步态相位观测。返回 [num_envs, 2] 的 [sin(phase), cos(phase)]。"""
    # 每个 env 当前已运行时间及对应的周期相位 [0, 2π)
    time = env.episode_length_buf * env.step_dt
    phase = 2.0 * torch.pi * torch.remainder(time, period) / period

    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)
