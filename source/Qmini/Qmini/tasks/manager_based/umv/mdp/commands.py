from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class LineRidesCommand(CommandTerm):
    """Composite LineRides command for DRIVE -> skill -> DRIVE.

    Full internal command layout (R16):

        [
            mode,
            vx, vy, wz,
            rho_upper, rho_lower,
            x_stunt_x, x_stunt_y, x_stunt_z,
            p_goal_x, p_goal_y, p_goal_z,
            w    x   y    z
        ]

    where:
        mode = 0 -> DRIVE
        mode = 1, 2, 3 -> skills encoded as 01, 10, 11 for the policy

    DRIVE output:
        [0, vx, 0, wz, upper, lower, 0, 0, 0, 0, 0, 0]

    STUNT output (quaternion omitted in these examples):
        [skill_id, 0, 0, 0, 0, 0, dx, dy, dz, px, py, pz]

    Internal column offsets remain stable for rewards and terminations.
    The policy receives two mode bits, driving commands, current stunt
    displacement and target displacement (13 dimensions).
    """

    DRIVE = 0
    STUNT = 1

    cfg: LineRidesCommandCfg

    def __init__(self, cfg: LineRidesCommandCfg, env: ManagerBasedRLEnv):
        # NOTE:
        # CommandTerm.__init__ may call _set_debug_vis_impl() immediately.
        # Therefore that method must not depend on buffers initialized below.
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # ------------------------------------------------------------------
        # Static 3-D guideline in the frozen yaw-only stunt-start frame.
        # ------------------------------------------------------------------
        self._init_guidelines()
        # Legacy single-guideline attributes refer to the first enabled skill.
        first_mode = int(self.enabled_skill_modes[0].item())
        self.guideline_name = self.guideline_names[first_mode]
        self.guideline_s = self.guidelines[first_mode]
        self.num_waypoints = int(self.guideline_s.shape[0])

        if not (0 <= cfg.start_waypoint_index < self.num_waypoints):
            raise ValueError(
                f"start_waypoint_index={cfg.start_waypoint_index} is invalid "
                f"for {self.num_waypoints} waypoints."
            )

        # Cumulative guideline arc length d_i.
        segment_lengths = torch.linalg.vector_norm(
            self.guideline_s[1:] - self.guideline_s[:-1], dim=-1
        )
        self.guideline_arc_length = torch.cat(
            (
                torch.zeros(1, dtype=torch.float32, device=self.device),
                torch.cumsum(segment_lengths, dim=0),
            ),
            dim=0,
        )

        # ------------------------------------------------------------------
        # Explicit mode state.
        # ------------------------------------------------------------------
        self.mode = torch.full(
            (self.num_envs,),
            self.DRIVE,
            dtype=torch.long,
            device=self.device,
        )

        # Keep a fixed subset of parallel environments in DRIVE for training.
        if not 0.0 <= cfg.drive_only_fraction <= 1.0:
            raise ValueError("drive_only_fraction must be in [0, 1].")
        self.drive_only = torch.arange(self.num_envs, device=self.device) < int(
            self.num_envs * cfg.drive_only_fraction
        )
        self.skill_mode = torch.full_like(self.mode, first_mode)
        # Increment only on actual waypoint hits (including the final one).
        self.reached_waypoint_count = torch.zeros_like(self.mode)
        # Keep episode-level outcomes until CurriculumManager reads them before
        # CommandManager.reset() clears the episode on the same reset step.
        self.episode_skill_attempts = torch.zeros(self.num_envs, 4, dtype=torch.long, device=self.device)
        self.episode_skill_successes = torch.zeros_like(self.episode_skill_attempts)

        # Full R16 output.
        self.command_b = torch.zeros(
            self.num_envs,
            16,
            dtype=torch.float32,
            device=self.device,
        )

        # ------------------------------------------------------------------
        # Driving command.
        # ------------------------------------------------------------------
        self.vel_command_b = torch.zeros(
            self.num_envs,
            3,
            dtype=torch.float32,
            device=self.device,
        )
        # Preserve the independent angular sample while limiting the exposed
        # yaw-rate command to a configurable curvature envelope.
        self.raw_ang_vel_command = torch.zeros(self.num_envs, device=self.device)
        curvature = cfg.drive_max_curvature
        if curvature is not None and (not math.isfinite(curvature) or curvature <= 0.0):
            raise ValueError("drive_max_curvature must be finite and positive, or None.")

        # rho[:, 0] = q_h desired (upper body)
        # rho[:, 1] = q_n desired (lower body)
        self.rho_command = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float32,
            device=self.device,
        )

        # Independent command hold timers.
        self.lin_vel_time_left = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.ang_vel_time_left = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.rho_time_left = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        # Time remaining until DRIVE -> STUNT.
        self.stunt_switch_time_left = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        # ------------------------------------------------------------------
        # Stunt-start frame.
        # ------------------------------------------------------------------
        self.stunt_start_pos_w = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self.stunt_start_yaw_quat_w = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.device
        )
        self.stunt_start_yaw_quat_w[:, 0] = 1.0

        # True after the first DRIVE -> STUNT transition in the current
        # episode.  Used only for debug visualization so the last anchored
        # guideline can remain visible after returning to DRIVE.
        self.guideline_anchor_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # ------------------------------------------------------------------
        # Guideline progression.
        # ------------------------------------------------------------------
        self.current_waypoint_idx = torch.full(
            (self.num_envs,),
            cfg.start_waypoint_index,
            dtype=torch.long,
            device=self.device,
        )

        self.goal_pos_s = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )

        self.completed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Actual base path length d_t accumulated during the current stunt.
        self.traveled_distance = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.previous_base_pos_w = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )

        # ------------------------------------------------------------------
        # Robot IDs.
        # ------------------------------------------------------------------
        # The stunt frame origin is the base body position at DRIVE -> STUNT.
        # The base body's local +X is defined as vehicle forward.  We freeze
        # only its yaw, so stunt +Z is exactly world +Z and the frame remains
        # right-handed (X forward, Y lateral, Z up).
        try:
            self.base_body_id = self.robot.body_names.index(cfg.base_body_name)
        except ValueError as exc:
            raise ValueError(
                f"base_body_name='{cfg.base_body_name}' is not in robot.body_names: "
                f"{self.robot.body_names}"
            ) from exc

        # Rho visualization IDs.
        self.q_h_joint_id = self.robot.joint_names.index("q_h")
        self.q_n_joint_id = self.robot.joint_names.index("q_n")
        self.neck_body_id = self.robot.body_names.index("neck")
        self.head_body_id = self.robot.body_names.index("head")

        self._joint_axis_y = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self._joint_axis_y[:, 1] = 1.0

        # ------------------------------------------------------------------
        # Metrics.
        # ------------------------------------------------------------------
        self.metrics["distance_to_goal"] = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.metrics["traveled_distance"] = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.metrics["stunt_mode"] = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        for name in ("drive_mode", "drive_lin_vel_error", "drive_ang_vel_error", "drive_rho_error"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        for name in ("drive_lin_vel_bias", "drive_ang_vel_bias", "drive_tracking_success"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        self._metric_steps = torch.zeros(self.num_envs, device=self.device)
        self._drive_metric_steps = torch.zeros(self.num_envs, device=self.device)
        self._drive_metric_sums = torch.zeros(self.num_envs, 6, device=self.device)

    # ======================================================================
    # Public API
    # ======================================================================

    @property
    def command(self) -> torch.Tensor:
        """Return full R16 command (scalar mode ID in column 0).

        x_stunt is refreshed on access. This matters in ManagerBasedRLEnv
        because reward/termination are evaluated before CommandManager.compute()
        in each control step.
        """
        if hasattr(self, "command_b"):
            self._refresh_x_stunt_columns()
        return self.command_b

    @property
    def policy_command(self) -> torch.Tensor:
        """R13: [mode_hi, mode_lo, vx, vy, wz, q_h, q_n, x_s(3), goal-x_s(3)]."""
        command = self.command
        bits = torch.stack((self.mode // 2, self.mode % 2), dim=-1).to(command.dtype)
        return torch.cat((bits, command[:, 1:9], command[:, 9:12] - command[:, 6:9]), dim=-1)

    @property
    def x_stunt(self) -> torch.Tensor:
        """Current [dx, dy, dz] in the stunt-start frame.

        Returns zeros in DRIVE mode.
        """
        value = self._compute_x_stunt()
        stunt_mask = (self.mode != self.DRIVE).unsqueeze(-1)
        return torch.where(stunt_mask, value, torch.zeros_like(value))

    @property
    def distance_to_goal(self) -> torch.Tensor:
        """Current ||x_stunt - p_goal||, zero outside STUNT."""
        distance = torch.linalg.vector_norm(
            self.x_stunt - self.goal_pos_s,
            dim=-1,
        )
        return torch.where(
            self.mode != self.DRIVE,
            distance,
            torch.zeros_like(distance),
        )

    @property
    def traveled_distance_current(self) -> torch.Tensor:
        """Current d_t including motion since the last command update."""
        pending = torch.linalg.vector_norm(
            self._base_pos_w() - self.previous_base_pos_w,
            dim=-1,
        )
        return self.traveled_distance + torch.where(
            self.mode != self.DRIVE,
            pending,
            torch.zeros_like(pending),
        )

    @property
    def active_guideline_distance(self) -> torch.Tensor:
        """Cumulative reference arc length d_i for the active waypoint."""
        return self.guideline_arc_lengths[self.skill_mode, self.current_waypoint_idx]

    def _init_guidelines(self):
        """Load mode->JSON mapping and pad unequal-length paths for batched indexing."""
        files = self.cfg.guideline_files
        if files is None:
            if self.cfg.guideline_file is None:
                raise ValueError('Provide guideline_file or guideline_files={"01": ..., "10": ..., "11": ...}.')
            files = {"01": self.cfg.guideline_file}
        if not files or any(code not in ("01", "10", "11") for code in files):
            raise ValueError('guideline_files must use non-empty string keys from {"01", "10", "11"}.')
        # Config dictionary keys must be strings for IsaacLab serialization/Hydra.
        files = {int(code, 2): path for code, path in files.items()}
        self.guideline_names = {}
        self.guidelines = {}
        for mode, path in sorted(files.items()):
            name, points = self._load_guideline_json(path)
            if not 0 <= self.cfg.start_waypoint_index < len(points):
                raise ValueError(f"Invalid start_waypoint_index for skill {mode}: {path}")
            self.guideline_names[mode] = name
            self.guidelines[mode] = points
        self.enabled_skill_modes = torch.tensor(sorted(files), device=self.device, dtype=torch.long)
        weights = self.cfg.skill_sampling_weights
        if len(weights) not in (3, len(files)) or any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("skill_sampling_weights must match the configured skills or have length 3.")
        active_weights = [weights[mode - 1] for mode in sorted(files)] if len(weights) == 3 else list(weights)
        self.skill_sampling_weights = torch.tensor(
            active_weights, device=self.device, dtype=torch.float32
        )
        if self.skill_sampling_weights.sum() <= 0:
            raise ValueError("At least one configured skill must have a positive sampling weight.")
        max_points = max(len(points) for points in self.guidelines.values())
        self.guideline_points = torch.zeros(4, max_points, 3, device=self.device)
        self.guideline_lengths = torch.zeros(4, dtype=torch.long, device=self.device)
        self.guideline_arc_lengths = torch.zeros(4, max_points, device=self.device)
        for mode, points in self.guidelines.items():
            n = len(points)
            self.guideline_points[mode, :n] = points
            self.guideline_points[mode, n:] = points[-1]
            self.guideline_lengths[mode] = n
            arc = torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1).cumsum(0)
            self.guideline_arc_lengths[mode, 1:n] = arc
            self.guideline_arc_lengths[mode, n:] = arc[-1]

    def _waypoint_positions(self, env_ids):
        return self.guideline_points[self.skill_mode[env_ids], self.current_waypoint_idx[env_ids]]

    # ======================================================================
    # JSON guideline
    # ======================================================================

    def _load_guideline_json(self, file_path: str) -> tuple[str, torch.Tensor]:
        """Load a 3-D guideline from JSON.

        Expected format:

        {
            "name": "mini_hop",
            "frame": "stunt_start",
            "waypoints": [
                [x0, y0, z0],
                [x1, y1, z1],
                ...
            ]
        }
        """
        path = Path(file_path).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(f"Guideline JSON does not exist: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        name = str(data.get("name", path.stem))
        frame = str(data.get("frame", "stunt_start"))

        if frame != "stunt_start":
            raise ValueError(
                f"Unsupported guideline frame '{frame}'. Expected 'stunt_start'."
            )

        if "waypoints" not in data:
            raise KeyError(f"'waypoints' is missing from guideline JSON: {path}")

        points = torch.tensor(
            data["waypoints"],
            dtype=torch.float32,
            device=self.device,
        )

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "Guideline waypoints must have shape [N, 3], "
                f"got {tuple(points.shape)}."
            )

        if points.shape[0] < 2:
            raise ValueError("Guideline requires at least two waypoints.")

        if not torch.isfinite(points).all():
            raise ValueError("Guideline contains NaN or Inf.")

        return name, points

    # ======================================================================
    # CommandTerm interface
    # ======================================================================

    def _resample_command(self, env_ids: Sequence[int]):
        """Episode reset / base-term resample.

        RESET always enters DRIVE mode.
        """
        env_ids = self._env_ids_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        # RESET -> DRIVE.
        self.mode[env_ids] = self.DRIVE
        self.completed[env_ids] = False
        self.reached_waypoint_count[env_ids] = 0
        self.episode_skill_attempts[env_ids] = 0
        self.episode_skill_successes[env_ids] = 0
        self._metric_steps[env_ids] = 0.0
        self._drive_metric_steps[env_ids] = 0.0
        self._drive_metric_sums[env_ids] = 0.0

        # Stunt fields are inactive in DRIVE.
        self.goal_pos_s[env_ids] = 0.0
        self.current_waypoint_idx[env_ids] = self.cfg.start_waypoint_index
        self.traveled_distance[env_ids] = 0.0
        self.guideline_anchor_valid[env_ids] = False

        # Fresh drive commands.
        self._sample_linear_velocity(env_ids)
        self._sample_angular_velocity(env_ids)
        self._sample_rho(env_ids)

        # Random delay before next stunt.
        self._sample_stunt_switch_time(env_ids)

        self.previous_base_pos_w[env_ids] = self._base_pos_w(env_ids)

        self._build_command()

    def _update_command(self):
        """Advance RESET/DRIVE -> STUNT -> DRIVE state machine."""
        dt = float(self._env.step_dt)

        # ------------------------------------------------------------------
        # 1. DRIVE timers.
        # ------------------------------------------------------------------
        drive_mask = self.mode == self.DRIVE

        self.stunt_switch_time_left[drive_mask] -= dt
        self.lin_vel_time_left[drive_mask] -= dt
        self.ang_vel_time_left[drive_mask] -= dt
        self.rho_time_left[drive_mask] -= dt

        # Decide DRIVE -> STUNT first.
        enter_stunt_mask = drive_mask & ~self.drive_only & (self.stunt_switch_time_left <= 0.0)
        remaining_drive = drive_mask & ~enter_stunt_mask

        # Independent drive-command resampling.
        resample_lin = remaining_drive & (self.lin_vel_time_left <= 0.0)
        if resample_lin.any():
            self._sample_linear_velocity(
                resample_lin.nonzero(as_tuple=False).squeeze(-1)
            )

        resample_ang = remaining_drive & (self.ang_vel_time_left <= 0.0)
        if resample_ang.any():
            self._sample_angular_velocity(
                resample_ang.nonzero(as_tuple=False).squeeze(-1)
            )

        resample_rho = remaining_drive & (self.rho_time_left <= 0.0)
        if resample_rho.any():
            self._sample_rho(
                resample_rho.nonzero(as_tuple=False).squeeze(-1)
            )

        # ------------------------------------------------------------------
        # 2. DRIVE -> STUNT.
        # ------------------------------------------------------------------
        if enter_stunt_mask.any():
            self._enter_stunt(
                enter_stunt_mask.nonzero(as_tuple=False).squeeze(-1)
            )

        # ------------------------------------------------------------------
        # 3. STUNT progression.
        # ------------------------------------------------------------------
        stunt_mask = self.mode != self.DRIVE
        if stunt_mask.any():
            self._update_stunt_state(stunt_mask)

        # ------------------------------------------------------------------
        # 4. Final waypoint -> DRIVE.
        # ------------------------------------------------------------------
        finished_mask = (self.mode != self.DRIVE) & self.completed
        if finished_mask.any():
            self._enter_drive(
                finished_mask.nonzero(as_tuple=False).squeeze(-1)
            )

        # ------------------------------------------------------------------
        # 5. Build full R16 output.
        # ------------------------------------------------------------------
        self._build_command()

    def _update_metrics(self):
        self.metrics["distance_to_goal"][:] = self.distance_to_goal
        self.metrics["traveled_distance"][:] = self.traveled_distance_current
        drive = self.mode == self.DRIVE
        self._metric_steps += 1.0
        self._drive_metric_steps += drive.float()
        drive_fraction = self._drive_metric_steps / self._metric_steps.clamp_min(1.0)
        self.metrics["drive_mode"][:] = drive_fraction
        self.metrics["stunt_mode"][:] = 1.0 - drive_fraction
        # Positive bias means actual signed velocity is greater than the command.
        lin_error = self.robot.data.root_lin_vel_b[:, 0] - self.vel_command_b[:, 0]
        ang_error = self.robot.data.root_ang_vel_b[:, 2] - self.vel_command_b[:, 2]
        rho_actual = self.robot.data.joint_pos[:, [self.q_h_joint_id, self.q_n_joint_id]]
        rho_error = (self.rho_command - rho_actual).square().sum(-1).sqrt()
        success = (lin_error.abs() < 0.2) & (ang_error.abs() < 0.2)
        values = torch.stack(
            (lin_error.abs(), ang_error.abs(), rho_error, lin_error, ang_error, success.float()), dim=-1
        )
        self._drive_metric_sums += torch.where(drive.unsqueeze(-1), values, 0.0)
        means = self._drive_metric_sums / self._drive_metric_steps.clamp_min(1.0).unsqueeze(-1)
        for index, name in enumerate((
            "drive_lin_vel_error", "drive_ang_vel_error", "drive_rho_error",
            "drive_lin_vel_bias", "drive_ang_vel_bias", "drive_tracking_success",
        )):
            self.metrics[name][:] = means[:, index]

    # ======================================================================
    # State machine
    # ======================================================================

    def _enter_stunt(self, env_ids: torch.Tensor, skill_mode: int | None = None):
        """DRIVE -> STUNT."""
        if env_ids.numel() == 0:
            return

        if skill_mode is None:
            sampled = torch.multinomial(self.skill_sampling_weights, env_ids.numel(), replacement=True)
            self.skill_mode[env_ids] = self.enabled_skill_modes[sampled]
        else:
            if skill_mode not in self.guidelines:
                raise ValueError(f"No guideline configured for skill mode {skill_mode}.")
            self.skill_mode[env_ids] = skill_mode
        self.episode_skill_attempts[env_ids, self.skill_mode[env_ids]] += 1
        self.mode[env_ids] = self.skill_mode[env_ids]
        self.completed[env_ids] = False

        # --------------------------------------------------------------
        # Freeze the yaw-only stunt frame exactly at DRIVE -> STUNT.
        #
        # origin : base body position at the switching instant
        # +X     : base_link local +X projected onto the world XY plane
        # +Y     : right-handed lateral axis
        # +Z     : world +Z
        # roll/pitch of the bicycle are intentionally ignored.
        # --------------------------------------------------------------
        self.stunt_start_pos_w[env_ids] = self._base_pos_w(env_ids)
        self.stunt_start_yaw_quat_w[env_ids] = self._base_yaw_quat_w(env_ids)
        self.guideline_anchor_valid[env_ids] = True

        # Start tracking p_i.
        self.current_waypoint_idx[env_ids] = self.cfg.start_waypoint_index
        self.goal_pos_s[env_ids] = self._waypoint_positions(env_ids)

        # Reset stunt traveled distance.
        self.traveled_distance[env_ids] = 0.0
        self.previous_base_pos_w[env_ids] = self._base_pos_w(env_ids)

    def _enter_drive(self, env_ids: torch.Tensor):
        """STUNT -> DRIVE after successful final waypoint.

        Driving-command timers are frozen during STUNT and resume here.
        We therefore keep the previous vx / wz / rho commands instead of
        unconditionally resampling them. If a command timer happened to
        expire exactly on the DRIVE -> STUNT transition step, resample only
        that expired command before exposing the next DRIVE observation.
        """
        if env_ids.numel() == 0:
            return

        succeeded = env_ids[self.completed[env_ids]]
        self.episode_skill_successes[succeeded, self.skill_mode[succeeded]] += 1
        self.mode[env_ids] = self.DRIVE
        self.completed[env_ids] = False

        # p_goal is masked/inactive in DRIVE.
        self.goal_pos_s[env_ids] = 0.0

        # STUNT -> DRIVE: clear the frozen guideline visualization.
        # The next guideline will be anchored from the next DRIVE -> STUNT
        # switching pose.
        self.guideline_anchor_valid[env_ids] = False

        # Preserve unexpired driving commands across STUNT. Only commands
        # whose own hold timer has already expired are sampled again.
        expired_lin = env_ids[self.lin_vel_time_left[env_ids] <= 0.0]
        if expired_lin.numel() > 0:
            self._sample_linear_velocity(expired_lin)

        expired_ang = env_ids[self.ang_vel_time_left[env_ids] <= 0.0]
        if expired_ang.numel() > 0:
            self._sample_angular_velocity(expired_ang)

        expired_rho = env_ids[self.rho_time_left[env_ids] <= 0.0]
        if expired_rho.numel() > 0:
            self._sample_rho(expired_rho)

        # The paper explicitly samples a new DRIVE duration after stunt
        # completion, so only the mode-switch timer is always restarted.
        self._sample_stunt_switch_time(env_ids)

    def _update_stunt_state(self, stunt_mask: torch.Tensor):
        """Integrate d_t and advance p_i -> p_{i+1}."""
        current_pos_w = self._base_pos_w()

        # Actual traveled distance d_t.
        step_distance = torch.linalg.vector_norm(
            current_pos_w - self.previous_base_pos_w,
            dim=-1,
        )
        self.traveled_distance[stunt_mask] += step_distance[stunt_mask]
        self.previous_base_pos_w[stunt_mask] = current_pos_w[stunt_mask]

        # Current active p_i.
        self.goal_pos_s[stunt_mask] = self._waypoint_positions(stunt_mask)

        distance = self.distance_to_goal

        # Advance at most one waypoint per control step.
        last_idx = self.guideline_lengths[self.skill_mode] - 1
        reached = stunt_mask & (distance < self.cfg.waypoint_margin)
        self.reached_waypoint_count[reached] += 1
        can_advance = reached & (self.current_waypoint_idx < last_idx)
        # Do not mark a newly selected final waypoint completed in the same step.
        self.completed[:] = reached & (self.current_waypoint_idx == last_idx)

        if can_advance.any():
            self.current_waypoint_idx[can_advance] += 1
            self.goal_pos_s[can_advance] = self._waypoint_positions(can_advance)

    def set_mode(self, mode: int, env_ids: Sequence[int] | torch.Tensor | slice = slice(None)):
        """Select 0=DRIVE or a configured skill 1/2/3; normal timers resume afterward."""
        if type(mode) is not int or mode not in (0, 1, 2, 3):
            raise ValueError("mode must be an integer in {0, 1, 2, 3}.")
        ids = self._env_ids_tensor(env_ids)
        if mode == self.DRIVE:
            self._enter_drive(ids)
        else:
            self._enter_stunt(ids, skill_mode=mode)
        self._build_command()

    # ======================================================================
    # Driving command sampling
    # ======================================================================

    def _sample_linear_velocity(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        n = env_ids.numel()

        self.vel_command_b[env_ids, 0] = torch.empty(
            n, device=self.device
        ).uniform_(*self.cfg.ranges.lin_vel_x)

        # Keep vy explicit. Set range=(0,0) for bicycle operation.
        self.vel_command_b[env_ids, 1] = torch.empty(
            n, device=self.device
        ).uniform_(*self.cfg.ranges.lin_vel_y)
        self._limit_drive_yaw_rate(env_ids)

        self.lin_vel_time_left[env_ids] = torch.empty(
            n, device=self.device
        ).uniform_(*self.cfg.lin_vel_resampling_time_range)

    def _sample_angular_velocity(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        n = env_ids.numel()

        self.raw_ang_vel_command[env_ids] = torch.empty(
            n, device=self.device
        ).uniform_(*self.cfg.ranges.ang_vel_z)
        self._limit_drive_yaw_rate(env_ids)

        self.ang_vel_time_left[env_ids] = torch.empty(
            n, device=self.device
        ).uniform_(*self.cfg.ang_vel_resampling_time_range)

    def _limit_drive_yaw_rate(self, env_ids: torch.Tensor):
        """Apply |wz| <= curvature * |commanded vx| without measuring actual velocity.

        This is a training command envelope, not a measured physical limit.
        Recompute on both linear and angular resampling; do not restart timers.
        """
        limit = self.cfg.drive_max_curvature
        raw = self.raw_ang_vel_command[env_ids]
        if limit is None:
            self.vel_command_b[env_ids, 2] = raw
        else:
            yaw_limit = limit * self.vel_command_b[env_ids, 0].abs()
            self.vel_command_b[env_ids, 2] = torch.maximum(torch.minimum(raw, yaw_limit), -yaw_limit)

    def _sample_rho(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        n = env_ids.numel()

        self.rho_command[env_ids, 0] = torch.empty(
            n, device=self.device
        ).uniform_(*self.cfg.ranges.rho_upper)

        self.rho_command[env_ids, 1] = torch.empty(
            n, device=self.device
        ).uniform_(*self.cfg.ranges.rho_lower)

        self.rho_time_left[env_ids] = torch.empty(
            n, device=self.device
        ).uniform_(*self.cfg.rho_resampling_time_range)

    def _sample_stunt_switch_time(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        self.stunt_switch_time_left[env_ids] = torch.empty(
            env_ids.numel(), device=self.device
        ).uniform_(*self.cfg.stunt_switch_time_range)

    # ======================================================================
    # Full command construction
    # ======================================================================

    def _build_command(self):
        """Build full R16 command (including the frozen frame quaternion).

        Layout:
            [
                mode,
                vx, vy, wz,
                rho_upper, rho_lower,
                x_stunt_x, x_stunt_y, x_stunt_z,
                p_goal_x, p_goal_y, p_goal_z,
            ]
        """
        self.command_b.zero_()

        drive_mask = self.mode == self.DRIVE
        stunt_mask = self.mode != self.DRIVE

        # Explicit mode flag.
        self.command_b[:, 0] = self.mode.float()

        # DRIVE:
        # [0, vx, vy, wz, upper, lower, 0,0,0, 0,0,0]
        self.command_b[drive_mask, 1:4] = self.vel_command_b[drive_mask]
        self.command_b[drive_mask, 4:6] = self.rho_command[drive_mask]

        # STUNT:
        # [1, 0,0,0, 0,0, dx,dy,dz, px,py,pz]
        x_stunt = self._compute_x_stunt()
        self.command_b[stunt_mask, 6:9] = x_stunt[stunt_mask]
        self.command_b[stunt_mask, 9:12] = self.goal_pos_s[stunt_mask]

        self.command_b[stunt_mask, 12:16] = (
            self.stunt_start_yaw_quat_w[stunt_mask]
        )


    def _refresh_x_stunt_columns(self):
        """Refresh only dynamic x_stunt columns for immediate consumers."""
        if not hasattr(self, "mode"):
            return

        # Always zero the x_stunt block first.
        self.command_b[:, 6:9] = 0.0

        stunt_mask = self.mode != self.DRIVE
        if stunt_mask.any():
            value = self._compute_x_stunt()
            self.command_b[stunt_mask, 6:9] = value[stunt_mask]

    def _compute_x_stunt(self) -> torch.Tensor:
        """Compute current base displacement in the frozen yaw-only stunt frame."""
        delta_pos_w = self._base_pos_w() - self.stunt_start_pos_w
        return math_utils.quat_apply_inverse(
            self.stunt_start_yaw_quat_w,
            delta_pos_w,
        )

    # ======================================================================
    # Termination helper
    # ======================================================================

    def missed_active_waypoint(self) -> torch.Tensor:
        """Return LineRides traveled-distance failure mask.

        Failure when current p_i has not been reached before actual traveled
        distance d_t exceeds the guideline cumulative distance d_i.
        """
        return (
            (self.mode != self.DRIVE)
            & (self.traveled_distance_current > self.active_guideline_distance)
            & (self.distance_to_goal >= self.cfg.waypoint_margin)
        )

    # ======================================================================
    # Frame / debug helpers
    # ======================================================================

    def _base_pos_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return base-body position(s) in world frame."""
        if env_ids is None:
            return self.robot.data.body_pos_w[:, self.base_body_id]
        return self.robot.data.body_pos_w[env_ids, self.base_body_id]

    def _base_yaw_quat_w(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Yaw-only quaternion of ``base_body_name``.

        Coordinate convention:
            +X: base body local +X (vehicle forward)
            +Y: right-handed lateral axis
            +Z: world +Z

        Roll and pitch are removed at DRIVE -> STUNT, so the frozen stunt
        frame keeps its Z axis coincident with world Z.
        """
        base_quat_w = self.robot.data.body_quat_w[env_ids, self.base_body_id]
        return math_utils.yaw_quat(base_quat_w)

    def guideline_world(self, env_id: int) -> torch.Tensor:
        """Transform all local guideline points [N,3] into world frame."""
        points = self.guidelines[int(self.skill_mode[env_id].item())]
        quat = self.stunt_start_yaw_quat_w[
            env_id : env_id + 1
        ].expand(len(points), -1)

        return (
            self.stunt_start_pos_w[env_id]
            + math_utils.quat_apply(quat, points)
        )

    def _env_ids_tensor(self, env_ids: Sequence[int] | torch.Tensor | slice) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(
                self.num_envs,
                dtype=torch.long,
                device=self.device,
            )[env_ids]

        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long).flatten()

        return torch.as_tensor(
            list(env_ids),
            dtype=torch.long,
            device=self.device,
        ).flatten()

    # ======================================================================
    # Debug visualization
    # ======================================================================

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create/toggle DRIVE and STUNT visualizers."""
        if debug_vis:
            if not hasattr(self, "drive_vel_visualizer"):
                self.drive_vel_visualizer = VisualizationMarkers(
                    self.cfg.drive_vel_visualizer_cfg
                )

            if not hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer = VisualizationMarkers(
                    self.cfg.current_vel_visualizer_cfg
                )

            if not hasattr(self, "yaw_cmd_visualizer"):
                self.yaw_cmd_visualizer = VisualizationMarkers(
                    self.cfg.yaw_cmd_visualizer_cfg
                )

            if not hasattr(self, "rho_goal_visualizer"):
                self.rho_goal_visualizer = VisualizationMarkers(
                    self.cfg.rho_goal_visualizer_cfg
                )

            if not hasattr(self, "rho_current_visualizer"):
                self.rho_current_visualizer = VisualizationMarkers(
                    self.cfg.rho_current_visualizer_cfg
                )

            if not hasattr(self, "guideline_visualizer"):
                self.guideline_visualizer = VisualizationMarkers(
                    self.cfg.guideline_visualizer_cfg
                )

            self.drive_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
            self.yaw_cmd_visualizer.set_visibility(True)
            self.rho_goal_visualizer.set_visibility(True)
            self.rho_current_visualizer.set_visibility(True)
            self.guideline_visualizer.set_visibility(False)

        else:
            for name in (
                "drive_vel_visualizer",
                "current_vel_visualizer",
                "yaw_cmd_visualizer",
                "rho_goal_visualizer",
                "rho_current_visualizer",
                "guideline_visualizer",
            ):
                if hasattr(self, name):
                    getattr(self, name).set_visibility(False)

    def _debug_vis_callback(self, event):
        """Draw selected environments in one batch per marker collection."""
        if not hasattr(self, "robot") or not hasattr(self, "mode") or not self.robot.is_initialized:
            return

        env_ids = self._debug_env_ids()
        drive_ids = env_ids[self.mode[env_ids] == self.DRIVE]
        has_drive = drive_ids.numel() > 0
        for name in (
            "drive_vel_visualizer", "current_vel_visualizer", "yaw_cmd_visualizer",
            "rho_goal_visualizer", "rho_current_visualizer",
        ):
            getattr(self, name).set_visibility(has_drive)
        if has_drive:
            self._visualize_drive_velocity(drive_ids)
            self._visualize_current_velocity(drive_ids)
            self._visualize_yaw_command(drive_ids)
            self._visualize_rho_command(drive_ids)

        guideline_ids = env_ids[self.guideline_anchor_valid[env_ids] & (self.mode[env_ids] != self.DRIVE)]
        self.guideline_visualizer.set_visibility(guideline_ids.numel() > 0)
        if guideline_ids.numel() > 0:
            self._visualize_guideline(guideline_ids)

    def _debug_env_ids(self) -> torch.Tensor:
        """Select up to debug_num_envs starting at the legacy debug_env_id."""
        start = self.cfg.debug_env_id
        count = self.cfg.debug_num_envs
        if type(start) is not int or start < 0:
            raise ValueError("debug_env_id must be a non-negative integer.")
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("debug_num_envs must be a non-negative integer or None.")
        stop = self.num_envs if count is None else min(self.num_envs, start + count)
        return torch.arange(min(start, self.num_envs), stop, device=self.device, dtype=torch.long)

    def _visualize_drive_velocity(self, env_ids: torch.Tensor):
        """Draw one commanded planar velocity arrow per driving environment."""
        marker_pos_w = self.robot.data.root_pos_w[env_ids].clone()
        marker_pos_w[:, 2] += self.cfg.drive_vel_marker_height
        scale, quat = self._resolve_xy_velocity_to_arrow(
            self.vel_command_b[env_ids, :2],
            self.robot.data.root_quat_w[env_ids],
            self.drive_vel_visualizer,
        )
        self.drive_vel_visualizer.visualize(translations=marker_pos_w, orientations=quat, scales=scale)

    def _visualize_current_velocity(self, env_ids: torch.Tensor):
        """Draw one measured planar velocity arrow per driving environment."""
        marker_pos_w = self.robot.data.root_pos_w[env_ids].clone()
        marker_pos_w[:, 2] += self.cfg.current_vel_marker_height
        scale, quat = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_b[env_ids, :2],
            self.robot.data.root_quat_w[env_ids],
            self.current_vel_visualizer,
        )
        self.current_vel_visualizer.visualize(translations=marker_pos_w, orientations=quat, scales=scale)

    def _resolve_xy_velocity_to_arrow(
        self,
        xy_velocity_b: torch.Tensor,
        base_quat_w: torch.Tensor,
        visualizer: VisualizationMarkers,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert base-frame XY velocities to marker scales and world quaternions.

        This follows IsaacLab's ``UniformVelocityCommand`` implementation:
        marker local X is scaled by the velocity norm and rotated by the
        velocity heading, then transformed from base to world coordinates.
        ``xy_velocity_b`` and ``base_quat_w`` may contain one or many
        environments as long as their leading dimensions match.
        """
        default_scale = visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.as_tensor(
            default_scale,
            device=xy_velocity_b.device,
            dtype=xy_velocity_b.dtype,
        ).repeat(xy_velocity_b.shape[0], 1)

        speed = torch.linalg.vector_norm(xy_velocity_b, dim=1)
        arrow_scale[:, 0] *= (
            self.cfg.drive_vel_marker_min_scale
            + self.cfg.drive_vel_marker_gain * speed
        )

        heading_angle = torch.atan2(xy_velocity_b[:, 1], xy_velocity_b[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat_b = math_utils.quat_from_euler_xyz(
            zeros,
            zeros,
            heading_angle,
        )
        arrow_quat_w = math_utils.quat_mul(base_quat_w, arrow_quat_b)
        return arrow_scale, arrow_quat_w

    def _visualize_yaw_command(self, env_ids: torch.Tensor):
        """Draw one signed yaw-rate arrow per driving environment."""
        marker_pos_w = self.robot.data.root_pos_w[env_ids].clone()
        marker_pos_w[:, 2] += self.cfg.yaw_marker_height
        wz = self.vel_command_b[env_ids, 2]
        half_pi = torch.full_like(wz, torch.pi / 2)
        heading = torch.where(wz >= 0, half_pi, -half_pi)
        zeros = torch.zeros_like(heading)
        quat = math_utils.quat_mul(
            self.robot.data.root_quat_w[env_ids],
            math_utils.quat_from_euler_xyz(zeros, zeros, heading),
        )
        scale = torch.tensor(
            self.yaw_cmd_visualizer.cfg.markers["arrow"].scale, device=self.device
        ).repeat(env_ids.numel(), 1)
        wz_max = max(abs(self.cfg.ranges.ang_vel_z[0]), abs(self.cfg.ranges.ang_vel_z[1]), 1e-6)
        scale[:, 0] *= self.cfg.yaw_marker_min_scale + self.cfg.yaw_marker_gain * (wz.abs() / wz_max).clamp(0, 1)
        self.yaw_cmd_visualizer.visualize(translations=marker_pos_w, orientations=quat, scales=scale)

    def _visualize_rho_command(self, env_ids: torch.Tensor):
        """Batch desired/actual neck and head orientation arrows."""
        base_quat = self.robot.data.root_quat_w[env_ids]
        neck_pos = self.robot.data.body_pos_w[env_ids, self.neck_body_id]
        neck_quat = self.robot.data.body_quat_w[env_ids, self.neck_body_id]
        head_pos = self.robot.data.body_pos_w[env_ids, self.head_body_id]
        head_quat = self.robot.data.body_quat_w[env_ids, self.head_body_id]
        axis = self._joint_axis_y[env_ids]
        neck_goal = math_utils.quat_mul(
            base_quat, math_utils.quat_from_angle_axis(self.rho_command[env_ids, 1], axis)
        )
        head_goal = math_utils.quat_mul(
            neck_quat, math_utils.quat_from_angle_axis(self.rho_command[env_ids, 0], axis)
        )
        positions = torch.cat((neck_pos, head_pos))
        goals = torch.cat((neck_goal, head_goal))
        actual = torch.cat((neck_quat, head_quat))
        count = 2 * env_ids.numel()
        goal_scale = torch.tensor(
            self.rho_goal_visualizer.cfg.markers["arrow"].scale, device=self.device
        ).repeat(count, 1)
        actual_scale = torch.tensor(
            self.rho_current_visualizer.cfg.markers["arrow"].scale, device=self.device
        ).repeat(count, 1)
        self.rho_goal_visualizer.visualize(translations=positions, orientations=goals, scales=goal_scale)
        self.rho_current_visualizer.visualize(translations=positions, orientations=actual, scales=actual_scale)

    def _visualize_guideline(self, env_ids: torch.Tensor):
        """Batch unequal-length paths; highlight each environment's own target."""
        positions, indices = [], []
        for env_id in env_ids.tolist():
            points = self.guideline_world(env_id)
            markers = torch.zeros(len(points), dtype=torch.int32, device=self.device)
            if self.mode[env_id] != self.DRIVE:
                markers[self.current_waypoint_idx[env_id]] = 1
            positions.append(points)
            indices.append(markers)
        self.guideline_visualizer.visualize(
            translations=torch.cat(positions), marker_indices=torch.cat(indices)
        )


@configclass
class LineRidesCommandCfg(CommandTermCfg):
    """Configuration for the internal R16 / policy R13 LineRides command."""

    class_type: type = LineRidesCommand

    asset_name: str = MISSING
    guideline_file: str | None = None
    # String keys are policy mode codes: "01", "10", "11". "00" is DRIVE.
    # If omitted, the legacy guideline_file supplies mode 1 only.
    guideline_files: dict[str, str] | None = None
    # Either length 3 (IDs 1/2/3), or one weight per configured skill sorted by ID.
    skill_sampling_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    drive_only_fraction: float = 0.0
    # Optional training command envelope in 1/m. None preserves independent vx/wz.
    drive_max_curvature: float | None = None

    # Stunt-frame geometry.  At DRIVE -> STUNT, freeze the yaw of
    # ``base_body_name``.  Its local +X is vehicle forward; +Z is world +Z.
    # The resulting stunt frame is right-handed and ignores roll/pitch.
    base_body_name: str = "base_link"

    # Disable the generic periodic resampler. This command owns its timers.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    # DRIVE -> STUNT timer.
    stunt_switch_time_range: tuple[float, float] = (2.0, 5.0)

    # Independent driving-command timers.
    lin_vel_resampling_time_range: tuple[float, float] = (2.0, 15.0)
    ang_vel_resampling_time_range: tuple[float, float] = (3.0, 8.0)
    rho_resampling_time_range: tuple[float, float] = (3.0, 10.0)

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = (-1.0, 2.0)
        lin_vel_y: tuple[float, float] = (0.0, 0.0)
        ang_vel_z: tuple[float, float] = (-1.5, 1.5)

        # Keep explicit because your robot's q_h/q_n limits may require
        # a coordinate mapping/clipping different from the paper's raw range.
        rho_upper: tuple[float, float] = MISSING
        rho_lower: tuple[float, float] = MISSING

    ranges: Ranges = MISSING

    # Guideline tracking.
    waypoint_margin: float = 0.30
    start_waypoint_index: int = 1

    # Draw this many environments; None means all remaining environments, 0 means none.
    debug_num_envs: int | None = 1
    # Legacy ID is now an optional starting index; defaults to the first environment.
    debug_env_id: int = 0

    # ------------------------------------------------------------------
    # DRIVE visualization
    # ------------------------------------------------------------------
    drive_vel_visualizer_cfg: VisualizationMarkersCfg = (
        GREEN_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/LineRides/drive_velocity"
        )
    )
    # Readable at normal IsaacLab camera distances; X is extended further at
    # runtime according to the commanded planar speed.
    drive_vel_visualizer_cfg.markers["arrow"].scale = (0.50, 0.50, 0.50)

    drive_vel_marker_height: float = 1.15
    current_vel_visualizer_cfg: VisualizationMarkersCfg = (
        BLUE_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/LineRides/current_velocity"
        )
    )
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.50, 0.50, 0.50)

    current_vel_marker_height: float = 1.15
    drive_vel_marker_min_scale: float = 0.0
    drive_vel_marker_gain: float = 3.0

    yaw_cmd_visualizer_cfg: VisualizationMarkersCfg = (
        GREEN_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/LineRides/yaw_command"
        )
    )
    yaw_cmd_visualizer_cfg.markers["arrow"].scale = (0.35, 0.07, 0.07)

    yaw_marker_height: float = 1.15
    yaw_marker_min_scale: float = 0.10
    yaw_marker_gain: float = 1.0

    rho_goal_visualizer_cfg: VisualizationMarkersCfg = (
        GREEN_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/LineRides/rho_goal"
        )
    )
    rho_goal_visualizer_cfg.markers["arrow"].scale = (0.30, 0.06, 0.06)

    rho_current_visualizer_cfg: VisualizationMarkersCfg = (
        BLUE_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/LineRides/rho_current"
        )
    )
    rho_current_visualizer_cfg.markers["arrow"].scale = (0.23, 0.045, 0.045)

    # ------------------------------------------------------------------
    # STUNT visualization
    # ------------------------------------------------------------------
    guideline_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/LineRides/Guideline",
        markers={
            "reference": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.45, 0.05),
                ),
            ),
            "current_target": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.05, 1.0, 0.10),
                ),
            ),
        },
    )


# =============================================================================
# MDP accessors
# =============================================================================


def linerides_policy_command(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Policy-visible R13 command, including two mode bits and target displacement."""
    term: LineRidesCommand = env.command_manager.get_term(command_name)
    return term.policy_command


def linerides_full_command(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Full R16 command for reward/debug/internal calculations."""
    term: LineRidesCommand = env.command_manager.get_term(command_name)
    return term.command


def linerides_missed_waypoint(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Termination mask for traveled-distance waypoint failure."""
    term: LineRidesCommand = env.command_manager.get_term(command_name)
    return term.missed_active_waypoint()
