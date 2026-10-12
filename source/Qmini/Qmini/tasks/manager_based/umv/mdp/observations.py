import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv,ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg,CommandTerm
import isaaclab.utils.math as math_utils

# def joint_pos_rel_sincos(env: ManagerBasedEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Actuated joint positions with sin/cos encoding for rear wheel.

#     Joint order: [q_h, q_n, q_s, q_w]
#     Returns: [q_h_rel, q_n_rel, q_s_rel, sin(q_w_rel), cos(q_w_rel)]
#     Shape: (num_envs, 5)
#     """
#     asset: Articulation = env.scene[asset_cfg.name]

#     joint_pos_rel = (
#         asset.data.joint_pos[:, asset_cfg.joint_ids]
#         - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
#     )

#     # 截断连续关节角度到 [-pi, pi]，保证数值稳定性
#     wheel_pos = torch.remainder(joint_pos_rel[:, -1:] + torch.pi, 2 * torch.pi) - torch.pi

#     return torch.cat(
#         (
#             joint_pos_rel[:, :-1],
#             torch.sin(wheel_pos),
#             torch.cos(wheel_pos),
#         ),
#         dim=-1,
#     )
    
    
def joint_pos_rel_sincos(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Actuated joint positions with sin/cos encoding for rear wheel.

    Joint order: [q_h, q_n, q_s, q_w]
    Returns: [q_h_rel, q_n_rel, q_s_rel, sin(q_w_rel), cos(q_w_rel)]
    Shape: (num_envs, 5)
    """
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos_rel = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )

    # ---------------------------------------------------------
    # NaN / Inf check
    # ---------------------------------------------------------
    invalid_mask = ~torch.isfinite(joint_pos_rel)

    if invalid_mask.any():
        invalid_env_ids = torch.nonzero(
            invalid_mask.any(dim=1),
            as_tuple=False,
        ).squeeze(-1)

        print(
            "[ERROR] joint_pos_rel contains NaN / Inf!\n"
            f"env_ids: {invalid_env_ids.tolist()}\n"
            f"values:\n{joint_pos_rel[invalid_env_ids]}"
        )

    # Rear wheel continuous joint -> [-pi, pi]
    wheel_pos = (
        torch.remainder(
            joint_pos_rel[:, -1:] + torch.pi,
            2.0 * torch.pi,
        )
        - torch.pi
    )

    obs = torch.cat(
        (
            joint_pos_rel[:, :-1],
            torch.sin(wheel_pos),
            torch.cos(wheel_pos),
        ),
        dim=-1,
    )

    # ---------------------------------------------------------
    # Final observation check
    # ---------------------------------------------------------
    invalid_obs_mask = ~torch.isfinite(obs)

    if invalid_obs_mask.any():
        invalid_env_ids = torch.nonzero(
            invalid_obs_mask.any(dim=1),
            as_tuple=False,
        ).squeeze(-1)

        print(
            "[ERROR] joint_pos_rel_sincos observation contains NaN / Inf!\n"
            f"env_ids: {invalid_env_ids.tolist()}\n"
            f"obs:\n{obs[invalid_env_ids]}"
        )

    return obs
    
    
def policy_command(
    env: ManagerBasedRLEnv,
    command_name: str
) -> torch.Tensor:
    """R13: [mode_hi, mode_lo, vx, vy, wz, q_h, q_n, x_s(3), goal-x_s(3)].

    Encode the stable internal scalar mode ID as 00/01/10/11. This also
    supports the legacy gamepad command (which currently selects 00/01 only).
    """
    command = env.command_manager.get_command(command_name)
    mode = command[:, 0].long()
    bits = torch.stack((mode // 2, mode % 2), dim=-1).to(command.dtype)
    return torch.cat((bits, command[:, 1:9], command[:, 9:12] - command[:, 6:9]), dim=-1)


def critic_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_name: str = "robot",
    base_body_name: str = "base_link",
) -> torch.Tensor:
    """LineRides critic command.

    Returns R17: policy command R13 followed by q_SB (4).
    where:
        - DRIVE: q_SB = [1, 0, 0, 0] (identity quaternion)
        - STUNT: q_SB = inv(q_WS) * q_WB (base orientation in frozen stunt-start frame)
    """
    command = env.command_manager.get_command(command_name)
    robot = env.scene[asset_name]
    base_body_id = robot.body_names.index(base_body_name)

    # Base orientation in world frame and default DRIVE identity quaternion
    q_WB = robot.data.body_quat_w[:, base_body_id]
    q_SB = torch.zeros_like(q_WB)
    q_SB[:, 0] = 1.0

    stunt_mask = command[:, 0] > 0.5
    if stunt_mask.any():
        q_WS_stunt = command[stunt_mask, 12:16]
        q_WB_stunt = q_WB[stunt_mask]

        # q_SB = inv(q_WS) * q_WB
        q_SB_stunt = math_utils.quat_mul(math_utils.quat_inv(q_WS_stunt), q_WB_stunt)
        q_SB_stunt = q_SB_stunt / torch.linalg.vector_norm(q_SB_stunt, dim=-1, keepdim=True).clamp_min(1.0e-8)
        q_SB_stunt = torch.where(q_SB_stunt[:, 0:1] < 0.0, -q_SB_stunt, q_SB_stunt)

        q_SB[stunt_mask] = q_SB_stunt

    return torch.cat((policy_command(env, command_name), q_SB), dim=-1)
