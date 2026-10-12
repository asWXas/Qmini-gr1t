from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import carb
import carb.input
# import omni.appwindow

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class LineRidesCommand_Play(CommandTerm):
    """Self-contained gamepad LineRides command for DRIVE -> STUNT -> DRIVE.

    This command owns the Omniverse gamepad subscription. In MODE 0 it reads
    the sticks directly. A-button rising edge enters MODE 1. After the final
    waypoint, MODE 1 exits back to MODE 0 automatically. Rumble is attempted
    directly through pygame when enabled and available.

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
        mode = 1 -> STUNT

    DRIVE output:
        [0, vx, 0, wz, upper, lower, 0, 0, 0, 0, 0, 0]

    STUNT output:
        [1, 0, 0, 0, 0, 0, dx, dy, dz, px, py, pz]

    The last three entries are always the current guideline target p_i while
    STUNT is active. The policy should normally consume only command[:, :9],
    so p_goal remains internal to reward / termination / visualization.
    """

    DRIVE = 0
    STUNT = 1

    cfg: "LineRidesCommand_PlayCfg"

    def __init__(self, cfg: "LineRidesCommand_PlayCfg", env: "ManagerBasedRLEnv"):
        # NOTE:
        # CommandTerm.__init__ may call _set_debug_vis_impl() immediately.
        # Therefore that method must not depend on buffers initialized below.
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # ------------------------------------------------------------------
        # Static 3-D guideline in the frozen yaw-only stunt-start frame.
        # ------------------------------------------------------------------
        self.guideline_name, self.guideline_s = self._load_guideline_json(
            cfg.guideline_file
        )
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

        # rho[:, 0] = q_h desired (upper body)
        # rho[:, 1] = q_n desired (lower body)
        self.rho_command = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float32,
            device=self.device,
        )

        # ------------------------------------------------------------------
        # Runtime gamepad state.
        # ------------------------------------------------------------------
        # MODE 0 is controlled by the physical gamepad. The latest stick
        # command is kept in ``vel_command_b`` continuously, even while STUNT
        # is active, so the robot resumes the *current* stick command
        # immediately after the trajectory finishes.
        self.stunt_trigger_requested = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.a_button_down = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # One-shot rumble events.  A counter (rather than a bool) guarantees
        # that two events cannot collapse into one if a consumer is delayed.
        # Event 1: after DRIVE -> STUNT.
        # Event 2: after STUNT -> DRIVE.
        self.rumble_event_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
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

        # ------------------------------------------------------------------
        # Physical gamepad I/O (play-time only).
        # ------------------------------------------------------------------
        self._gamepad_raw = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        self._input_iface = None
        self._gamepad = None
        self._gamepad_sub = None
        self._pygame = None
        self._rumble_joystick = None

        if not (0 <= int(self.cfg.gamepad_env_id) < self.num_envs):
            raise ValueError(
                f"gamepad_env_id={self.cfg.gamepad_env_id} is invalid for "
                f"num_envs={self.num_envs}."
            )

        self._init_gamepad_io()
        self._init_rumble_backend()

    # ======================================================================
    # Public API
    # ======================================================================

    @property
    def command(self) -> torch.Tensor:
        """Return full R16 command.

        x_stunt is refreshed on access. This matters in ManagerBasedRLEnv
        because reward/termination are evaluated before CommandManager.compute()
        in each control step.
        """
        if hasattr(self, "command_b"):
            self._refresh_x_stunt_columns()
        return self.command_b

    @property
    def policy_command(self) -> torch.Tensor:
        """R13 observation matching training; this gamepad still selects 00/01 only."""
        command = self.command
        mode = command[:, 0].long()
        bits = torch.stack((mode // 2, mode % 2), dim=-1).to(command.dtype)
        return torch.cat((bits, command[:, 1:9], command[:, 9:12] - command[:, 6:9]), dim=-1)

    def set_drive_command(
        self,
        vx: float,
        vy: float = 0.0,
        wz: float = 0.0,
        *,
        env_id: int = 0,
    ):
        """Set the MODE-0 drive command directly in physical units.

        This method may be called every play-loop iteration, including while
        STUNT is active.  During STUNT the policy still receives zero drive
        velocity, but the latest value is cached and becomes active as soon as
        the trajectory returns to DRIVE.
        """
        self._validate_env_id(env_id)
        self.vel_command_b[env_id, 0] = float(vx)
        self.vel_command_b[env_id, 1] = float(vy)
        self.vel_command_b[env_id, 2] = float(wz)

    def set_rho_command(
        self,
        rho_upper: float,
        rho_lower: float,
        *,
        env_id: int = 0,
    ):
        """Optionally set the MODE-0 rho command from the play loop."""
        self._validate_env_id(env_id)
        self.rho_command[env_id, 0] = float(rho_upper)
        self.rho_command[env_id, 1] = float(rho_lower)

    def set_gamepad_input(
        self,
        forward: float,
        lateral: float,
        yaw: float,
        a_pressed: bool,
        *,
        env_id: int = 0,
    ):
        """Feed normalized gamepad input for one environment.

        Args:
            forward: normalized forward/backward stick value in [-1, 1], where
                +1 means forward.
            lateral: normalized lateral stick value in [-1, 1].  For a bicycle
                this will normally be 0 because ``lin_vel_y`` is configured to
                (0, 0).
            yaw: normalized yaw stick value in [-1, 1].
            a_pressed: current A-button state.  Only the rising edge triggers
                DRIVE -> STUNT, so holding A cannot repeatedly retrigger it.
            env_id: controlled environment, normally 0 for play.
        """
        self._validate_env_id(env_id)

        forward = self._apply_deadzone(float(forward))
        lateral = self._apply_deadzone(float(lateral))
        yaw = self._apply_deadzone(float(yaw))

        vx = self._scale_normalized_axis(forward, self.cfg.ranges.lin_vel_x)
        vy = self._scale_normalized_axis(lateral, self.cfg.ranges.lin_vel_y)
        wz = self._scale_normalized_axis(yaw, self.cfg.ranges.ang_vel_z)
        self.set_drive_command(vx, vy, wz, env_id=env_id)

        self.set_a_button_state(a_pressed, env_id=env_id)

    def set_a_button_state(self, a_pressed: bool, *, env_id: int = 0):
        """Update the physical A-button state and trigger only on its rising edge.

        This is the preferred entry point when the play script receives raw
        ``carb.input.GamepadEvent`` events.  Presses during STUNT are ignored;
        after returning to DRIVE the button must first be released and pressed
        again before another trajectory can start.
        """
        self._validate_env_id(env_id)
        was_down = bool(self.a_button_down[env_id].item())
        is_down = bool(a_pressed)
        if is_down and not was_down and int(self.mode[env_id].item()) == self.DRIVE:
            self.stunt_trigger_requested[env_id] = True
        self.a_button_down[env_id] = is_down

    def request_stunt(self, *, env_id: int = 0) -> bool:
        """Request MODE 1 explicitly; returns False if already in STUNT."""
        self._validate_env_id(env_id)
        if int(self.mode[env_id].item()) != self.DRIVE:
            return False
        self.stunt_trigger_requested[env_id] = True
        return True

    def consume_rumble_event(self, *, env_id: int = 0) -> bool:
        """Consume one pending rumble event counter (optional diagnostics).

        Hardware rumble is already attempted internally; this method is kept
        only for logging/testing or an alternate external rumble backend.
        """
        self._validate_env_id(env_id)
        if int(self.rumble_event_count[env_id].item()) <= 0:
            return False
        self.rumble_event_count[env_id] -= 1
        return True

    @property
    def x_stunt(self) -> torch.Tensor:
        """Current [dx, dy, dz] in the stunt-start frame.

        Returns zeros in DRIVE mode.
        """
        value = self._compute_x_stunt()
        stunt_mask = (self.mode == self.STUNT).unsqueeze(-1)
        return torch.where(stunt_mask, value, torch.zeros_like(value))

    @property
    def distance_to_goal(self) -> torch.Tensor:
        """Current ||x_stunt - p_goal||, zero outside STUNT."""
        distance = torch.linalg.vector_norm(
            self.x_stunt - self.goal_pos_s,
            dim=-1,
        )
        return torch.where(
            self.mode == self.STUNT,
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
            self.mode == self.STUNT,
            pending,
            torch.zeros_like(pending),
        )

    @property
    def active_guideline_distance(self) -> torch.Tensor:
        """Cumulative reference arc length d_i for the active waypoint."""
        return self.guideline_arc_length[self.current_waypoint_idx]

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
        """Episode reset / base-term resample.  RESET always enters DRIVE."""
        env_ids = self._env_ids_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        self.mode[env_ids] = self.DRIVE
        self.completed[env_ids] = False

        self.goal_pos_s[env_ids] = 0.0
        self.current_waypoint_idx[env_ids] = self.cfg.start_waypoint_index
        self.traveled_distance[env_ids] = 0.0
        self.guideline_anchor_valid[env_ids] = False

        # Reset runtime controller state. No random DRIVE command is sampled:
        # MODE 0 starts stationary and the physical stick immediately takes over.
        self.vel_command_b[env_ids] = 0.0
        self.rho_command[env_ids] = 0.0
        self.stunt_trigger_requested[env_ids] = False
        self.a_button_down[env_ids] = False
        self.rumble_event_count[env_ids] = 0

        self.previous_base_pos_w[env_ids] = self._base_pos_w(env_ids)
        self._build_command()

    def _update_command(self):
        """Read gamepad and advance DRIVE -> STUNT -> DRIVE state machine."""
        # Always keep the latest stick command cached. During STUNT it is not
        # exposed to the policy, but it becomes live immediately on MODE 0.
        self._poll_gamepad_velocity()

        # ------------------------------------------------------------------
        # 1. A-button request: DRIVE -> STUNT.
        # ------------------------------------------------------------------
        enter_stunt_mask = (self.mode == self.DRIVE) & self.stunt_trigger_requested
        if enter_stunt_mask.any():
            env_ids = enter_stunt_mask.nonzero(as_tuple=False).squeeze(-1)
            self.stunt_trigger_requested[env_ids] = False
            self._enter_stunt(env_ids)

            # The mode has already changed to STUNT when the event is queued.
            self._queue_rumble(env_ids)

        # Never carry a stale request through STUNT.
        self.stunt_trigger_requested[self.mode == self.STUNT] = False

        # ------------------------------------------------------------------
        # 2. STUNT progression.
        # ------------------------------------------------------------------
        stunt_mask = self.mode == self.STUNT
        if stunt_mask.any():
            self._update_stunt_state(stunt_mask)

        # ------------------------------------------------------------------
        # 3. Final waypoint: first exit STUNT, then queue rumble.
        # ------------------------------------------------------------------
        finished_mask = (self.mode == self.STUNT) & self.completed
        if finished_mask.any():
            env_ids = finished_mask.nonzero(as_tuple=False).squeeze(-1)
            self._enter_drive(env_ids)
            self._queue_rumble(env_ids)

        # ------------------------------------------------------------------
        # 4. Build full R16 output.
        # ------------------------------------------------------------------
        self._build_command()

    def _update_metrics(self):
        self.metrics["distance_to_goal"][:] = self.distance_to_goal
        self.metrics["traveled_distance"][:] = self.traveled_distance_current
        self.metrics["stunt_mode"][:] = (self.mode == self.STUNT).float()

    # ======================================================================
    # State machine
    # ======================================================================

    def _enter_stunt(self, env_ids: torch.Tensor):
        """DRIVE -> STUNT."""
        if env_ids.numel() == 0:
            return

        self.mode[env_ids] = self.STUNT
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
        self.goal_pos_s[env_ids] = self.guideline_s[
            self.current_waypoint_idx[env_ids]
        ]

        # Reset stunt traveled distance.
        self.traveled_distance[env_ids] = 0.0
        self.previous_base_pos_w[env_ids] = self._base_pos_w(env_ids)

    def _enter_drive(self, env_ids: torch.Tensor):
        """STUNT -> DRIVE after the final waypoint is reached.

        The latest physical joystick command is intentionally kept.
        Therefore the next MODE-0 observation immediately resumes live manual
        speed control rather than sampling or restoring an old random command.
        """
        if env_ids.numel() == 0:
            return

        # IMPORTANT: change the mode first.  The caller queues the completion
        # rumble only after this function returns, matching: EXIT -> RUMBLE.
        self.mode[env_ids] = self.DRIVE
        self.completed[env_ids] = False
        self.goal_pos_s[env_ids] = 0.0
        self.guideline_anchor_valid[env_ids] = False

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
        self.goal_pos_s[stunt_mask] = self.guideline_s[
            self.current_waypoint_idx[stunt_mask]
        ]

        distance = self.distance_to_goal

        # Advance at most one waypoint per control step.
        can_advance = (
            stunt_mask
            & (distance < self.cfg.waypoint_margin)
            & (self.current_waypoint_idx < self.num_waypoints - 1)
        )

        if can_advance.any():
            self.current_waypoint_idx[can_advance] += 1
            self.goal_pos_s[can_advance] = self.guideline_s[
                self.current_waypoint_idx[can_advance]
            ]

        # Recompute distance for the possibly new target.
        distance = self.distance_to_goal
        final_target = self.current_waypoint_idx == (self.num_waypoints - 1)

        self.completed[:] = (
            (self.mode == self.STUNT)
            & final_target
            & (distance < self.cfg.waypoint_margin)
        )

    # ======================================================================
    # Gamepad helpers
    # ======================================================================

    def _init_gamepad_io(self):
        """Subscribe directly to Omniverse gamepad 0 (or cfg.gamepad_index)."""
        try:
            # Prevent the simulator camera from consuming the same gamepad.
            carb.settings.get_settings().set_bool(
                "/persistent/app/omniverse/gamepadCameraControl", False
            )

            app_window = omni.appwindow.get_default_app_window()
            if app_window is None:
                raise RuntimeError("Omniverse default app window is unavailable.")

            self._input_iface = carb.input.acquire_input_interface()
            self._gamepad = app_window.get_gamepad(int(self.cfg.gamepad_index))
            if self._gamepad is None:
                raise RuntimeError(
                    f"No gamepad at index {self.cfg.gamepad_index}."
                )

            self._gamepad_sub = self._input_iface.subscribe_to_gamepad_events(
                self._gamepad, self._on_gamepad_event
            )
            carb.log_info(
                f"[LineRidesCommand_Play] gamepad {self.cfg.gamepad_index} connected."
            )
        except Exception as exc:
            self._input_iface = None
            self._gamepad = None
            self._gamepad_sub = None
            message = f"[LineRidesCommand_Play] failed to initialize gamepad: {exc}"
            if self.cfg.gamepad_required:
                raise RuntimeError(message) from exc
            carb.log_warn(message)

    def _on_gamepad_event(self, event: carb.input.GamepadEvent):
        """Omniverse gamepad event callback.

        Sticks update a two-sided raw buffer. A only reacts to its rising edge,
        so holding A can never repeatedly re-enter MODE 1.
        """
        value = float(event.value)
        if abs(value) < float(self.cfg.gamepad_deadzone):
            value = 0.0

        mapping = {
            carb.input.GamepadInput.LEFT_STICK_UP: (0, 0),
            carb.input.GamepadInput.LEFT_STICK_DOWN: (1, 0),
            carb.input.GamepadInput.LEFT_STICK_RIGHT: (0, 1),
            carb.input.GamepadInput.LEFT_STICK_LEFT: (1, 1),
            carb.input.GamepadInput.RIGHT_STICK_RIGHT: (0, 2),
            carb.input.GamepadInput.RIGHT_STICK_LEFT: (1, 2),
        }

        slot = mapping.get(event.input)
        if slot is not None:
            direction, axis = slot
            self._gamepad_raw[direction][axis] = max(0.0, value)

        if event.input == carb.input.GamepadInput.A:
            self.set_a_button_state(
                value > 0.5,
                env_id=int(self.cfg.gamepad_env_id),
            )

        return True

    def _poll_gamepad_velocity(self):
        """Resolve the latest stick state into MODE-0 [vx, vy, wz]."""
        if self._gamepad is None:
            return

        normalized = []
        for axis in range(3):
            positive = float(self._gamepad_raw[0][axis])
            negative = float(self._gamepad_raw[1][axis])
            normalized.append(positive if positive >= negative else -negative)

        vx = self._scale_normalized_axis(normalized[0], self.cfg.ranges.lin_vel_x)
        vy = self._scale_normalized_axis(normalized[1], self.cfg.ranges.lin_vel_y)
        wz = self._scale_normalized_axis(normalized[2], self.cfg.ranges.ang_vel_z)

        self.set_drive_command(
            vx,
            vy,
            wz,
            env_id=int(self.cfg.gamepad_env_id),
        )

    def _init_rumble_backend(self):
        """Best-effort pygame rumble backend.

        Carbonite exposes gamepad input but no documented rumble output API.
        Keeping the pygame backend here preserves the requested one-file CMD.
        If pygame/controller rumble is unavailable, mode switching still works.
        """
        if not self.cfg.enable_rumble:
            return

        try:
            pygame = importlib.import_module("pygame")
            pygame.joystick.init()
            index = int(self.cfg.gamepad_index)
            if pygame.joystick.get_count() <= index:
                raise RuntimeError(
                    f"pygame sees {pygame.joystick.get_count()} joystick(s), "
                    f"cannot open index {index}."
                )
            joystick = pygame.joystick.Joystick(index)
            joystick.init()
            self._pygame = pygame
            self._rumble_joystick = joystick
        except Exception as exc:
            self._pygame = None
            self._rumble_joystick = None
            carb.log_warn(
                "[LineRidesCommand_Play] rumble backend unavailable; "
                f"mode switching will still work. Reason: {exc}"
            )

    def _rumble_once(self):
        """Trigger one short vibration pulse if supported."""
        if self._rumble_joystick is None:
            return
        try:
            self._rumble_joystick.rumble(
                float(self.cfg.rumble_strength),
                float(self.cfg.rumble_strength),
                int(self.cfg.rumble_duration_ms),
            )
        except Exception as exc:
            carb.log_warn(f"[LineRidesCommand_Play] rumble failed: {exc}")

    def _queue_rumble(self, env_ids: torch.Tensor):
        """Queue/debug-count the event and vibrate after the mode transition."""
        if env_ids.numel() == 0:
            return
        self.rumble_event_count[env_ids] += 1

        gamepad_env_id = int(self.cfg.gamepad_env_id)
        if bool((env_ids == gamepad_env_id).any().item()):
            self._rumble_once()

    def _validate_env_id(self, env_id: int):
        if env_id < 0 or env_id >= self.num_envs:
            raise IndexError(
                f"env_id={env_id} is outside [0, {self.num_envs - 1}]"
            )

    def _apply_deadzone(self, value: float) -> float:
        """Apply symmetric gamepad deadzone and re-normalize to [-1, 1]."""
        value = max(-1.0, min(1.0, float(value)))
        deadzone = float(self.cfg.gamepad_deadzone)
        magnitude = abs(value)
        if magnitude <= deadzone:
            return 0.0
        scaled = (magnitude - deadzone) / max(1.0 - deadzone, 1.0e-6)
        return scaled if value >= 0.0 else -scaled

    @staticmethod
    def _scale_normalized_axis(
        value: float,
        limits: tuple[float, float],
    ) -> float:
        """Map normalized [-1, 1] input into possibly asymmetric limits."""
        value = max(-1.0, min(1.0, float(value)))
        lo, hi = float(limits[0]), float(limits[1])
        if value >= 0.0:
            return value * max(hi, 0.0)
        return (-value) * min(lo, 0.0)

    def __del__(self):
        """Release gamepad subscription and optional rumble handle."""
        try:
            if (
                self._input_iface is not None
                and self._gamepad is not None
                and self._gamepad_sub is not None
            ):
                self._input_iface.unsubscribe_to_gamepad_events(
                    self._gamepad, self._gamepad_sub
                )
        except Exception:
            pass

        try:
            if self._rumble_joystick is not None:
                self._rumble_joystick.quit()
        except Exception:
            pass

    # ======================================================================
    # Full command construction
    # ======================================================================

    def _build_command(self):
        """Build full R16 command.

        Layout:
            [
                mode,
                vx, vy, wz,
                rho_upper, rho_lower,
                x_stunt_x, x_stunt_y, x_stunt_z,
                p_goal_x, p_goal_y, p_goal_z,
                stunt_start_quat_w,
            ]
        """
        self.command_b.zero_()

        drive_mask = self.mode == self.DRIVE
        stunt_mask = self.mode == self.STUNT

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

        stunt_mask = self.mode == self.STUNT
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
            (self.mode == self.STUNT)
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
        quat = self.stunt_start_yaw_quat_w[
            env_id : env_id + 1
        ].expand(self.num_waypoints, -1)

        return (
            self.stunt_start_pos_w[env_id]
            + math_utils.quat_apply(quat, self.guideline_s)
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
class LineRidesCommand_Play_Cfg(CommandTermCfg):
    """Configuration for the gamepad-driven LineRides R16 command."""

    class_type: type = LineRidesCommand_Play

    asset_name: str = MISSING
    guideline_file: str = MISSING

    # Stunt-frame geometry.  At DRIVE -> STUNT, freeze the yaw of
    # ``base_body_name``.  Its local +X is vehicle forward; +Z is world +Z.
    # The resulting stunt frame is right-handed and ignores roll/pitch.
    base_body_name: str = "base_link"

    # Disable CommandTerm's generic periodic resampler. Runtime mode changes
    # are driven exclusively by the physical gamepad.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    # Legacy training-time timer fields are intentionally kept for config
    # compatibility, but are NOT used by this play-time gamepad command.
    stunt_switch_time_range: tuple[float, float] = (2.0, 5.0)
    lin_vel_resampling_time_range: tuple[float, float] = (2.0, 15.0)
    ang_vel_resampling_time_range: tuple[float, float] = (3.0, 8.0)
    rho_resampling_time_range: tuple[float, float] = (3.0, 10.0)

    # Physical gamepad. This command is intended for play with env 0.
    gamepad_index: int = 0
    gamepad_env_id: int = 0
    gamepad_required: bool = True
    gamepad_deadzone: float = 0.08

    # Short vibration pulse on MODE0->MODE1 and MODE1->MODE0.
    enable_rumble: bool = True
    rumble_strength: float = 0.75
    rumble_duration_ms: int = 180

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
    # Legacy ID is now an optional starting index.
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
    env: "ManagerBasedRLEnv",
    command_name: str,
) -> torch.Tensor:
    """Policy-visible R13 command matching the training observation layout."""
    term: LineRidesCommand_Play = env.command_manager.get_term(command_name)
    return term.policy_command


def linerides_full_command(
    env: "ManagerBasedRLEnv",
    command_name: str,
) -> torch.Tensor:
    """Full R16 command for reward/debug/internal calculations."""
    term: LineRidesCommand_Play = env.command_manager.get_term(command_name)
    return term.command


def linerides_missed_waypoint(
    env: "ManagerBasedRLEnv",
    command_name: str,
) -> torch.Tensor:
    """Termination mask for traveled-distance waypoint failure."""
    term: LineRidesCommand_Play = env.command_manager.get_term(command_name)
    return term.missed_active_waypoint()
