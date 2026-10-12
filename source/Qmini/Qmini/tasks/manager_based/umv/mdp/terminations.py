import math
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase
import isaaclab.utils.math as math_utils


def stunt_yaw_deviation(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_name: str = "robot",
    base_body_name: str = "base_link",
    max_yaw_deg: float = 90.0,
) -> torch.Tensor:
    """Terminate when heading deviates too much in stunt-start frame.

    Only active during STUNT / Line Track mode.
    Termination: |yaw_SB| >= max_yaw_deg (computed by rotating local +X into S frame).
    """
    command = env.command_manager.get_command(command_name)
    stunt_mask = command[:, 0] > 0.5

    terminated = torch.zeros(command.shape[0], dtype=torch.bool, device=command.device)
    if not stunt_mask.any():
        return terminated

    robot = env.scene[asset_name]
    base_body_id = robot.body_names.index(base_body_name)

    # Base orientation (q_WB) and frozen stunt-start orientation (q_WS) in world frame
    q_WB = robot.data.body_quat_w[stunt_mask, base_body_id]
    q_WS = command[stunt_mask, 12:16]

    q_WB = q_WB / torch.linalg.vector_norm(q_WB, dim=-1, keepdim=True).clamp_min(1.0e-8)
    q_WS = q_WS / torch.linalg.vector_norm(q_WS, dim=-1, keepdim=True).clamp_min(1.0e-8)

    # Orientation in stunt frame: q_SB = inv(q_WS) * q_WB
    q_SB = math_utils.quat_mul(math_utils.quat_inv(q_WS), q_WB)

    # Rotate vehicle local +X forward vector into stunt frame
    forward_b = torch.zeros((q_SB.shape[0], 3), dtype=q_SB.dtype, device=q_SB.device)
    forward_b[:, 0] = 1.0
    forward_s = math_utils.quat_apply(q_SB, forward_b)

    # Heading error in stunt XY plane
    yaw_error = torch.atan2(forward_s[:, 1], forward_s[:, 0])

    max_yaw_rad = math.radians(max_yaw_deg)
    terminated[stunt_mask] = yaw_error.abs() >= max_yaw_rad
    return terminated


def stunt_position_deviation(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_distance: float = 0.8,
) -> torch.Tensor:
    """Terminate if current stunt position is too far from current target.

    Condition (STUNT mode only):
        ||x_stunt - p_goal|| > max_distance
    """
    command = env.command_manager.get_command(command_name)
    stunt_mask = command[:, 0] > 0.5

    # Euclidean distance between current stunt position (6:9) and target waypoint (9:12)
    distance = torch.linalg.vector_norm(command[:, 6:9] - command[:, 9:12], dim=-1)

    return stunt_mask & (distance > max_distance)


class target_not_update_death(ManagerTermBase):
    """任何特技模式下，目标点长时间不更新则判定 episode 失败。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # 保存上一帧目标点
        self._last_target = torch.zeros(env.num_envs, 3, device=env.device)
        # 连续多少 step 没有更新目标
        self._no_update_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._last_mode = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
            
        self._last_target[env_ids] = 0.0
        self._no_update_steps[env_ids] = 0
        self._last_mode[env_ids] = 0

    def __call__(self, env: ManagerBasedRLEnv, command_name: str = "linerides", max_no_update_steps: int = 100, atol: float = 1e-5):
        cmd = env.command_manager.get_command(command_name)
        mode = cmd[:, 0]
        target = cmd[:, 9:12]  # 当前目标点

        # 比较上一帧目标和当前目标
        target_changed = (mode != self._last_mode) | (
            ~torch.isclose(target, self._last_target, atol=atol, rtol=0.0).all(dim=-1)
        )

        # All nonzero modes are active, including a target at the origin.
        active = mode != 0

        # 目标发生更新 -> 清零; 没有更新 -> +1
        self._no_update_steps[active & target_changed] = 0
        not_updated = active & (~target_changed)
        self._no_update_steps[not_updated] += 1

        # DRIVE 不进行死亡计数
        inactive = ~active
        self._no_update_steps[inactive] = 0

        # 每一帧刷新 last_target
        self._last_target[:] = target
        self._last_mode[:] = mode

        # 连续太久没更新 -> 死亡
        dead = active & (self._no_update_steps >= max_no_update_steps)

        return dead
