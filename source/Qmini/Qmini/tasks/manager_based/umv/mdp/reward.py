import math

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject,Articulation
from isaaclab.managers import ManagerTermBase
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils

def action_rate_l2_U(
    env: ManagerBasedRLEnv,
    max_sum: float = 100.0,
) -> torch.Tensor:
    """Penalize action rate with L2 squared, capped by max_sum."""

    action_rate = torch.sum(
        torch.square(
            env.action_manager.action
            - env.action_manager.prev_action
        ),
        dim=1,
    )

    return torch.clamp(
        action_rate,
        max=max_sum,
    )

def lin_vel_x_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity command along x axis using exponential kernel.

    Active only when mode (cmd[:, 0]) == 0.
    Command layout: [mode, vx, vy, wz, ...]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    # 模式和 x 方向目标速度
    mode = cmd[:, 0]
    target_lin_vel_x = cmd[:, 1]

    # 只计算 x 方向速度误差，不考虑 y
    lin_vel_error = torch.square(
        target_lin_vel_x - asset.data.root_lin_vel_b[:, 0]
    )

    reward = torch.exp(-lin_vel_error / (std**2))

    # 仅 mode == 0 时生效
    return torch.where(mode == 0, reward, torch.zeros_like(reward))


def ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel.
    
    Active only when mode (cmd[:, 0]) == 0.
    Command layout: [mode, vx, vy, wz, ...]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    
    # 提取模式与角速度目标 wz
    mode = cmd[:, 0]
    target_ang_vel_z = cmd[:, 3]
    # print(target_ang_vel_z )
    # 计算误差与指数奖励
    ang_vel_error = torch.square(target_ang_vel_z - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / (std**2))
    # print(target_ang_vel_z)
    # 仅在 mode == 0 时生效，其余情况归零
    return torch.where(mode == 0, reward, 0.0)


def drive_velocity_tracking_soft(
    env: ManagerBasedRLEnv,
    command_name: str,
    component: str,
    beta: float = 0.5,
    max_error: float = 5.0,
    error_scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """DRIVE-only broad velocity tracking reward; use a positive weight.

    Smooth L1 is quadratic near the target and linear for larger errors.
    Return 1 / (1 + smooth_l1(error) / error_scale). Error differences remain
    appreciable outside the narrow exponential tracking kernel. Keeping this
    companion reward non-negative avoids introducing a continuous error
    penalty that a policy could escape through early termination.
    This is reward shaping, not a
    differentiable controller: PPO still learns from sampled returns.

    Component is 'lin_vel_x' (m/s) or 'ang_vel_z' (rad/s). Keep the same
    base-frame velocity convention as lin_vel_x_exp / ang_vel_z_exp.
    The cap bounds extreme errors; beyond max_error the reward is constant.
    Ground-truth velocity is used for training reward only, never actor input.
    """
    if any(not math.isfinite(value) or value <= 0.0 for value in (beta, max_error, error_scale)):
        raise ValueError("beta, max_error and error_scale must be finite and positive.")
    command = env.command_manager.get_command(command_name)
    asset: RigidObject = env.scene[asset_cfg.name]
    if component == "lin_vel_x":
        error = command[:, 1] - asset.data.root_lin_vel_b[:, 0]
    elif component == "ang_vel_z":
        error = command[:, 3] - asset.data.root_ang_vel_b[:, 2]
    else:
        raise ValueError("component must be 'lin_vel_x' or 'ang_vel_z'.")
    absolute_error = error.abs().clamp_max(max_error)
    cost = torch.where(
        absolute_error < beta,
        0.5 * absolute_error.square() / beta,
        absolute_error - 0.5 * beta,
    )
    reward = 1.0 / (1.0 + cost / error_scale)
    return torch.where(command[:, 0] == 0, reward, torch.zeros_like(reward))

def track_rho_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.25,
    command_name: str = "linerides",
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=["q_h", "q_n"],     # 顺序必须与 cmd[:, 4:6] 对齐
        preserve_order=True,            # 关键!见下面第 3 节
    ),
) -> torch.Tensor:
    """DRIVE 模式(mode == 0)下跟踪 rho 关节目标 [q_h, q_n]。

    命令 layout: [mode, vx, vy, wz, rho_upper(q_h), rho_lower(q_n), ...]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    mode = cmd[:, 0]
    target = cmd[:, 4:6]                                    # [q_h_des, q_n_des]
    actual = asset.data.joint_pos[:, asset_cfg.joint_ids]    # [q_h, q_n]

    error = torch.sum(torch.square(target - actual), dim=1)
    reward = torch.exp(-error / (std ** 2))

    # 仅 DRIVE 生效:STUNT 段 cmd[4:6] 恒为 0,不屏蔽会把头/颈往 0 拉
    return torch.where(mode == 0, reward, 0.0)



def lin_vel_z_l2_m(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel only in DRIVE mode (mode == 0)."""
    # extract the used quantities (to enable type-hinting)
    cmd = env.command_manager.get_command("linerides")
    mode = cmd[:, 0]
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # mode == 0 时为 1.0（生效），mode == 1 时为 0.0（屏蔽）
    drive_mask = (mode == 0).float()
    
    return torch.square(asset.data.root_lin_vel_b[:, 2]) * drive_mask

def ang_vel_xy_l2_m(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    cmd = env.command_manager.get_command("linerides")
    mode = cmd[:, 0]
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # mode == 0 时为 1.0（生效），mode == 1 时为 0.0（屏蔽）
    drive_mask = (mode == 0).float()
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)* drive_mask

def flat_orientation_l2_m(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    cmd = env.command_manager.get_command("linerides")
    mode = cmd[:, 0]
    drive_mask = (mode == 0).float()
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) * drive_mask

#----------------------------------------

#point
def track_waypoint_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.20,
    command_name: str = "linerides",
) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    stunt_mask = (cmd[:, 0] != 0).float()
    x_stunt = cmd[:, 6:9]
    p_goal = cmd[:, 9:12]
    distance = torch.linalg.vector_norm(x_stunt - p_goal, dim=-1)
    reward = torch.exp(-torch.square(distance / std))
    return reward * stunt_mask 

class track_waypoint_bonus(ManagerTermBase):
    """每次目标刷新后，进入目标范围时奖励一次。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # 当前目标是否已经奖励过
        self._latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # 用于判断目标是否发生刷新
        self._last_target = torch.zeros(env.num_envs, 3, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
            
        self._latched[env_ids] = False
        self._last_target[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, margin: float = 0.30, command_name: str = "linerides"):
        cmd = env.command_manager.get_command(command_name)
        stunt = cmd[:, 0] != 0
        current = cmd[:, 6:9]
        target = cmd[:, 9:12]

        # ==================================================
        # 1. 判断目标是否有效
        # ==================================================
        valid_target = stunt

        # ==================================================
        # 2. 判断目标是否刷新
        # ==================================================
        target_changed = valid_target & (~torch.isclose(target, self._last_target, atol=1e-5, rtol=0.0).all(dim=-1))
        
        # 目标刷新：重新获得一次奖励资格，并记录新的目标
        self._latched[target_changed] = False
        self._last_target[target_changed] = target[target_changed]

        # ==================================================
        # 3. 无效目标重置状态
        # ==================================================
        self._latched[~stunt] = False
        self._last_target[~stunt] = 0.0

        # ==================================================
        # 4. 当前点和目标点之间的距离
        # ==================================================
        distance = torch.linalg.vector_norm(current - target, dim=-1)

        # ==================================================
        # 5. 奖励条件: 目标有效 + stunt 有效 + 距离 < margin + 当前目标还没有奖励过
        # ==================================================
        hit = valid_target & stunt & (distance < margin) & (~self._latched)

        # ==================================================
        # 6. 一旦奖励过，锁住
        # ==================================================
        self._latched[hit] = True

        return hit.float()

class track_waypoint_line(ManagerTermBase):
    """论文的 guideline tracking reward(progress 形式)。

        r_t = ||x_{t-1}^stunt - p_goal||_2 - ||x_t^stunt - p_goal||_2

    - 单位"米/控制步":靠近为正、远离为负;段内累加 = (起始距离 - 结束距离);
    - mode == 0 时不计分；特技中的原点也是合法目标;
    - 目标点切换 / episode reset 时重新播种 d_{t-1},切换那一帧不计分。
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # d_{t-1} = ||x_{t-1} - p_goal||
        self._last_distance = torch.zeros(env.num_envs, device=env.device)
        self._last_target = torch.zeros(env.num_envs, 3, device=env.device)
        self._last_mode = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        # _reset_idx() 对每一次 reset(超时/终止/env.reset())都会调用
        if env_ids is None:
            env_ids = slice(None)
        self._last_distance[env_ids] = 0.0
        self._last_target[env_ids] = 0.0
        self._last_mode[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, command_name: str = "linerides"):
        cmd = env.command_manager.get_command(command_name)

        target = cmd[:, 9:12]                                              # p_goal(stunt 系)
        distance = torch.linalg.vector_norm(cmd[:, 6:9] - target, dim=-1)  # d_t

        valid = cmd[:, 0] != 0

        # 有效目标发生变化(含首次进入 STUNT)-> 只播种,本帧不计分
        changed = valid & (
            (cmd[:, 0] != self._last_mode)
            | ~torch.isclose(target, self._last_target, atol=1e-6, rtol=0.0).all(dim=-1)
        )

        reward = self._last_distance - distance          # d_{t-1} - d_t
        reward[~valid | changed] = 0.0

        # --- 状态更新(两行的掩码故意的:见下面说明)---
        # 1) d_{t-1}:valid 就推进,包含"目标刚切换"那一帧 => 完成播种
        self._last_distance = torch.where(valid, distance, self._last_distance)
        # 2) 目标快照:valid 写当前目标;无效(DRIVE)时归 0,下次进 STUNT 才会被判为 changed
        self._last_target = torch.where(
            valid.unsqueeze(-1), target, torch.zeros_like(target)
        )
        self._last_mode[:] = cmd[:, 0]

        return reward



def front_wheel_only_penalty(
    env: ManagerBasedRLEnv,
    grace_time: float = 0.1,          # 后轮离地持续超过这么久才算违规(防抖)
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces",
        body_names=["rear_wheel", "front_wheel"],
        preserve_order=True,          # -> body_ids = [rear_wheel_id, front_wheel_id]
    ),
) -> torch.Tensor:
    """惩罚"仅前轮着地"(前轮着地 & 后轮离地)。

    允许:两轮都着地 / 仅后轮着地(抬头) / 两轮都离地(腾空)。
    返回 0/1 指示量,配合**负 weight**使用。
    若要当 termination 用,把最后的 `.float()` 去掉(返回 bool mask)即可。
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    rear_id, front_id = sensor_cfg.body_ids[0], sensor_cfg.body_ids[1]

    rear_airborne = sensor.data.current_air_time[:, rear_id] > grace_time     # 后轮离地 >= grace_time
    front_grounded = sensor.data.current_contact_time[:, front_id] > 0.0      # 前轮着地
    return (front_grounded & rear_airborne).float()


def flat_orientation_l2_p(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    cmd = env.command_manager.get_command("linerides")
    mode = cmd[:, 0]
    drive_mask = (mode != 0).float()
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) * drive_mask

def stunt_orientation_error_p(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_name: str = "robot",
    base_body_name: str = "base_link",
) -> torch.Tensor:
    """Orientation error active only in skill modes 01 and 11.

    Returns:
        - DRIVE (00) and skill 10: error = 0
        - Skills 01 and 11: error = 1 - q_SB.w^2 (alignment with target [1, 0, 0, 0])
    """
    command = env.command_manager.get_command(command_name)
    line_track_mask = (command[:, 0] == 1) | (command[:, 0] == 3)

    error = torch.zeros(command.shape[0], dtype=command.dtype, device=command.device)
    if not line_track_mask.any():
        return error

    robot = env.scene[asset_name]
    base_body_id = robot.body_names.index(base_body_name)

    # Base orientation (q_WB) and frozen stunt-start orientation (q_WS) in world frame
    q_WB = robot.data.body_quat_w[line_track_mask, base_body_id]
    q_WS = command[line_track_mask, 12:16]

    q_WB = q_WB / torch.linalg.vector_norm(q_WB, dim=-1, keepdim=True).clamp_min(1.0e-8)
    q_WS = q_WS / torch.linalg.vector_norm(q_WS, dim=-1, keepdim=True).clamp_min(1.0e-8)

    # Orientation in stunt frame: q_SB = inv(q_WS) * q_WB
    q_SB = math_utils.quat_mul(math_utils.quat_inv(q_WS), q_WB)
    q_SB = q_SB / torch.linalg.vector_norm(q_SB, dim=-1, keepdim=True).clamp_min(1.0e-8)

    # Error metric: 1 - w^2 = sin^2(theta / 2)
    error[line_track_mask] = (1.0 - q_SB[:, 0].square()).clamp(0.0, 1.0)
    return error

def is_terminated_p(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    
    cmd = env.command_manager.get_command("linerides")
    mode = cmd[:, 0]
    drive_mask = (mode != 0).float()
    return env.termination_manager.terminated.float() * drive_mask

def stunt_joint_position_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_name: str = "robot",
    joint_names: list[str] = ["q_n", "q_h"],
) -> torch.Tensor:
    """Joint position deviation penalty in Track-Line / STUNT mode only.

    Returns:
        - DRIVE: 0.0
        - STUNT: sum((q - q_ref)^2) relative to default_joint_pos
    """
    command = env.command_manager.get_command(command_name)
    track_line_mask = command[:, 0] > 0.5

    error = torch.zeros(command.shape[0], dtype=command.dtype, device=command.device)
    if not track_line_mask.any():
        return error

    robot = env.scene[asset_name]
    joint_ids = [robot.joint_names.index(name) for name in joint_names]

    # Current vs default joint positions for selected joints
    q = robot.data.joint_pos[track_line_mask][:, joint_ids]
    q_ref = robot.data.default_joint_pos[track_line_mask][:, joint_ids]

    error[track_line_mask] = (q - q_ref).square().sum(dim=-1)
    return error


class target_update_bonus(ManagerTermBase):
    """Reward actual waypoint hits, not automatic DRIVE->skill entry.

    The training command counts hits including the final waypoint. The
    legacy gamepad command falls back to target changes within STUNT only.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._last_target = torch.zeros(env.num_envs, 3, device=env.device)
        self._last_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._last_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._last_target[env_ids] = 0.0
        self._last_count[env_ids] = 0
        self._last_active[env_ids] = False

    def __call__(self, env: ManagerBasedRLEnv, command_name: str = "linerides", atol: float = 1e-5):
        term = env.command_manager.get_term(command_name)
        if hasattr(term, "reached_waypoint_count"):
            count = term.reached_waypoint_count
            reward = (count - self._last_count).clamp_min(0).float()
            self._last_count[:] = count
            return reward
        cmd = env.command_manager.get_command(command_name)
        target = cmd[:, 9:12]
        mask = cmd[:, 0] > 0.5
        # 当前目标是否有效
        valid_target = mask & self._last_active

        # 当前 target 和上一帧 target 是否不同
        changed = valid_target & (~torch.isclose(target, self._last_target, atol=atol, rtol=0.0).all(dim=-1))

        # 关键：每一帧都刷新 last_target
        self._last_target[:] = target
        self._last_active[:] = mask

        return changed.float()*mask
