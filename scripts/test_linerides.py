"""CPU logic tests for LineRides; no Isaac Sim launch is required.

Load production class/function AST nodes with a minimal manager/scene fixture.
This exercises tensor/state logic, not IsaacLab imports or simulator integration.
Run: python scripts/test_linerides.py -v
"""

import ast
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

MDP = Path(__file__).resolve().parents[1] / "source/Qmini/Qmini/tasks/manager_based/umv/mdp"


class ManagerFixture:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.metrics = {}


class EntityFixture:
    def __init__(self, name, **kwargs):
        self.name = name
        self.__dict__.update(kwargs)


def quat_apply(quat, value):
    xyz = quat[:, 1:]
    uv = torch.cross(xyz, value, dim=-1)
    return value + 2 * (quat[:, :1] * uv + torch.cross(xyz, uv, dim=-1))


def quat_inv(quat):
    return torch.cat((quat[:, :1], -quat[:, 1:]), dim=-1)


def quat_mul(left, right):
    w = left[:, :1] * right[:, :1] - (left[:, 1:] * right[:, 1:]).sum(-1, keepdim=True)
    xyz = left[:, :1] * right[:, 1:] + right[:, :1] * left[:, 1:]
    xyz += torch.cross(left[:, 1:], right[:, 1:], dim=-1)
    return torch.cat((w, xyz), dim=-1)


def quat_from_euler_xyz(roll, pitch, yaw):
    cr, sr = (roll / 2).cos(), (roll / 2).sin()
    cp, sp = (pitch / 2).cos(), (pitch / 2).sin()
    cy, sy = (yaw / 2).cos(), (yaw / 2).sin()
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )


class MarkerFixture:
    """Record whole marker batches like PointInstancer updates."""

    def __init__(self):
        self.cfg = SimpleNamespace(markers={"arrow": SimpleNamespace(scale=(0.5, 0.5, 0.5))})
        self.visible = False
        self.calls = []

    def set_visibility(self, visible):
        self.visible = visible

    def visualize(self, **kwargs):
        self.calls.append({name: value.clone() for name, value in kwargs.items()})


MATH = SimpleNamespace(
    quat_apply=quat_apply,
    quat_apply_inverse=lambda quat, value: quat_apply(quat_inv(quat), value),
    yaw_quat=lambda quat: quat,  # Fixture base orientations are yaw-only.
    quat_inv=quat_inv,
    quat_mul=quat_mul,
    quat_from_euler_xyz=quat_from_euler_xyz,
    quat_from_angle_axis=lambda angle, axis: torch.cat(
        ((angle / 2).cos().unsqueeze(-1), axis * (angle / 2).sin().unsqueeze(-1)), dim=-1
    ),
)


def load_logic(filename, class_names=()):
    tree = ast.parse((MDP / filename).read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef) and node.name in class_names
    ]
    namespace = dict(
        torch=torch,
        math=math,
        json=json,
        Path=Path,
        math_utils=MATH,
        CommandTerm=ManagerFixture,
        ManagerTermBase=ManagerFixture,
        SceneEntityCfg=EntityFixture,
        ManagerBasedRLEnv=object,
        ManagerBasedEnv=object,
    )
    # Postponed annotations match commands.py and avoid simulator type imports.
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(MDP / filename), "exec"), namespace)
    return namespace


COMMANDS = load_logic("commands.py", ("LineRidesCommand",))
PLAY = load_logic("commands_play.py", ("LineRidesCommand_Play",))
REWARDS = load_logic("reward.py", ("track_waypoint_line", "target_update_bonus", "track_waypoint_bonus"))
OBSERVATIONS = load_logic("observations.py")
TERMINATIONS = load_logic("terminations.py", ("target_not_update_death",))
CURRICULUM = load_logic("curriculum.py", ("LineRidesSkillCurriculum",))


class LineRidesTests(unittest.TestCase):
    def test_asymmetric_velocity_observation_config(self):
        tree = ast.parse((MDP.parent / "env/umv_env_cfg.py").read_text())
        observations = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ObservationsCfg"
        )
        groups = {node.name: node for node in observations.body if isinstance(node, ast.ClassDef)}

        def observes_base_linear_velocity(group):
            return any(isinstance(node, ast.Attribute) and node.attr == "base_lin_vel" for node in ast.walk(group))

        self.assertFalse(observes_base_linear_velocity(groups["PolicyCfg"]))
        self.assertTrue(observes_base_linear_velocity(groups["CriticCfg"]))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.files = {}
        self.paths = {
            1: [[0, 0, 0], [0.5, 0, 0], [1, 0, 0]],
            2: [[0, 0, 0], [0, 0.5, 0]],
            3: [[0, 0, 0], [0, 0, 0.4], [0.2, 0, 0.2], [0, 0, 0]],
        }
        for mode, points in self.paths.items():
            path = Path(self.tmp.name) / f"skill_{mode}.json"
            path.write_text(json.dumps({"frame": "stunt_start", "waypoints": points}))
            self.files[mode] = str(path)
        data = SimpleNamespace(
            body_pos_w=torch.zeros(4, 3, 3),
            body_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(4, 3, 1),
            root_lin_vel_b=torch.zeros(4, 3),
            root_ang_vel_b=torch.zeros(4, 3),
            joint_pos=torch.zeros(4, 2),
            default_joint_pos=torch.zeros(4, 2),
            projected_gravity_b=torch.tensor([0.2, 0.3, -0.9]).repeat(4, 1),
        )
        robot = SimpleNamespace(data=data, body_names=["base_link", "neck", "head"], joint_names=["q_h", "q_n"])
        self.env = SimpleNamespace(num_envs=4, device="cpu", step_dt=0.02, scene={"robot": robot})
        self.cfg = SimpleNamespace(
            asset_name="robot",
            base_body_name="base_link",
            guideline_files={f"{mode:02b}": path for mode, path in self.files.items()},
            guideline_file=None,
            start_waypoint_index=1,
            skill_sampling_weights=(1.0, 1.0, 1.0),
            drive_only_fraction=0.5,
            drive_max_curvature=None,
            waypoint_margin=0.05,
            stunt_switch_time_range=(2.0, 8.0),
            lin_vel_resampling_time_range=(2.0, 15.0),
            ang_vel_resampling_time_range=(3.0, 8.0),
            rho_resampling_time_range=(3.0, 10.0),
            ranges=SimpleNamespace(
                lin_vel_x=(-1.0, 2.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-1.5, 1.5),
                rho_upper=(0.06, 1.0),
                rho_lower=(-0.47, 0.52),
            ),
        )
        self.term = COMMANDS["LineRidesCommand"](self.cfg, self.env)
        self.env.command_manager = SimpleNamespace(
            get_command=lambda name: self.term.command,
            get_term=lambda name: self.term,
        )
        self.term._resample_command(slice(None))

    def assertTensor(self, actual, expected):
        torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=actual.dtype))

    def move_to_goal(self, env_id):
        self.term.robot.data.body_pos_w[env_id, 0] = self.term.goal_pos_s[env_id]

    def attach_debug_fixture(self):
        self.term.robot.is_initialized = True
        self.term.robot.data.root_pos_w = self.term.robot.data.body_pos_w[:, 0]
        self.term.robot.data.root_quat_w = self.term.robot.data.body_quat_w[:, 0]
        self.cfg.debug_num_envs = 1
        self.cfg.debug_env_id = 0
        self.cfg.drive_vel_marker_height = 1.15
        self.cfg.current_vel_marker_height = 1.15
        self.cfg.yaw_marker_height = 1.3
        self.cfg.drive_vel_marker_min_scale = 0.0
        self.cfg.drive_vel_marker_gain = 3.0
        self.cfg.yaw_marker_min_scale = 0.1
        self.cfg.yaw_marker_gain = 1.0
        for name in (
            "drive_vel_visualizer",
            "current_vel_visualizer",
            "yaw_cmd_visualizer",
            "rho_goal_visualizer",
            "rho_current_visualizer",
            "guideline_visualizer",
        ):
            setattr(self.term, name, MarkerFixture())

    def test_debug_count_bounds_and_legacy_start(self):
        self.attach_debug_fixture()
        self.cfg.debug_num_envs = 2
        self.assertTensor(self.term._debug_env_ids(), [0, 1])
        self.cfg.debug_num_envs = 100
        self.assertTensor(self.term._debug_env_ids(), [0, 1, 2, 3])
        self.cfg.debug_num_envs = None
        self.cfg.debug_env_id = 2
        self.assertTensor(self.term._debug_env_ids(), [2, 3])
        self.cfg.debug_num_envs = 0
        self.assertEqual(self.term._debug_env_ids().numel(), 0)
        self.cfg.debug_env_id = 20
        self.cfg.debug_num_envs = 2
        self.assertEqual(self.term._debug_env_ids().numel(), 0)
        self.cfg.debug_env_id = 0
        for invalid in (-1, 1.5, True):
            self.cfg.debug_num_envs = invalid
            with self.assertRaises(ValueError):
                self.term._debug_env_ids()

    def test_debug_batch_three_paths_and_driver_without_overwrite(self):
        self.attach_debug_fixture()
        self.cfg.debug_num_envs = None
        self.term.robot.data.root_pos_w[:, 0] = torch.tensor([0.0, 10.0, 20.0, 30.0])
        for mode in (1, 2, 3):
            self.term.set_mode(mode, [mode])
        original = self.term.robot.data.root_pos_w.clone()
        self.term._debug_vis_callback(None)
        paths = self.term.guideline_visualizer
        self.assertTrue(paths.visible)
        self.assertEqual(len(paths.calls), 1)
        self.assertTensor(
            paths.calls[0]["translations"], torch.cat([self.term.guideline_world(env_id) for env_id in (1, 2, 3)])
        )
        self.assertTensor(paths.calls[0]["marker_indices"], [0, 1, 0, 0, 1, 0, 1, 0, 0])
        for name in ("drive_vel_visualizer", "current_vel_visualizer", "yaw_cmd_visualizer"):
            marker = getattr(self.term, name)
            self.assertTrue(marker.visible)
            self.assertEqual(len(marker.calls), 1)
            self.assertEqual(marker.calls[0]["translations"].shape, (1, 3))
            self.assertEqual(marker.calls[0]["scales"].shape, (1, 3))
        self.assertEqual(self.term.rho_goal_visualizer.calls[0]["translations"].shape, (2, 3))
        self.assertTensor(self.term.robot.data.root_pos_w, original)

    def test_debug_multiple_drivers_and_hidden_old_paths(self):
        self.attach_debug_fixture()
        self.cfg.debug_num_envs = 4
        for mode in (1, 2, 3):
            self.term.set_mode(mode, [mode])
        self.term._debug_vis_callback(None)
        self.term.set_mode(0, [1, 2, 3])
        self.term._debug_vis_callback(None)
        self.assertFalse(self.term.guideline_visualizer.visible)
        self.assertEqual(len(self.term.guideline_visualizer.calls), 1)
        for name in ("drive_vel_visualizer", "current_vel_visualizer", "yaw_cmd_visualizer"):
            batch = getattr(self.term, name).calls[-1]
            self.assertEqual(batch["translations"].shape, (4, 3))
            self.assertEqual(batch["orientations"].shape, (4, 4))
            self.assertEqual(batch["scales"].shape, (4, 3))
        self.assertEqual(self.term.rho_goal_visualizer.calls[-1]["translations"].shape, (8, 3))
        self.assertEqual(self.term.rho_current_visualizer.calls[-1]["scales"].shape, (8, 3))
        self.cfg.debug_num_envs = 0
        self.term._debug_vis_callback(None)
        for name in (
            "drive_vel_visualizer",
            "current_vel_visualizer",
            "yaw_cmd_visualizer",
            "rho_goal_visualizer",
            "rho_current_visualizer",
            "guideline_visualizer",
        ):
            self.assertFalse(getattr(self.term, name).visible)

    def test_debug_only_selected_environments_are_drawn(self):
        self.attach_debug_fixture()
        self.cfg.debug_num_envs = 2
        self.cfg.debug_env_id = 2
        for mode in (1, 2, 3):
            self.term.set_mode(mode, [mode])
        self.term._debug_vis_callback(None)
        self.assertFalse(self.term.drive_vel_visualizer.visible)
        self.assertTensor(
            self.term.guideline_visualizer.calls[0]["translations"],
            torch.cat([self.term.guideline_world(2), self.term.guideline_world(3)]),
        )

    def test_gamepad_multi_environment_debug(self):
        self.attach_debug_fixture()
        self.cfg.debug_num_envs = 4
        self.term.set_mode(1, [1, 3])
        play = PLAY["LineRidesCommand_Play"].__new__(PLAY["LineRidesCommand_Play"])
        play.__dict__.update(self.term.__dict__)
        play._debug_vis_callback(None)
        self.assertEqual(play.drive_vel_visualizer.calls[0]["translations"].shape, (2, 3))
        self.assertEqual(play.guideline_visualizer.calls[0]["translations"].shape, (6, 3))
        self.assertTensor(play.guideline_visualizer.calls[0]["marker_indices"], [0, 1, 0, 0, 1, 0])

    def test_mode_encoding_offsets_and_observations(self):
        for mode in (1, 2, 3):
            self.term.set_mode(mode, [mode])
        policy = OBSERVATIONS["policy_command"](self.env, "linerides")
        self.assertEqual(policy.shape, (4, 13))
        self.assertTensor(policy[:, :2], [[0, 0], [0, 1], [1, 0], [1, 1]])
        self.assertTensor(policy, self.term.policy_command)
        self.assertTensor(self.term.command[0, 1:4], self.term.vel_command_b[0])
        self.assertTensor(self.term.command[0, 4:6], self.term.rho_command[0])
        self.assertTensor(self.term.command[1:, 1:6], torch.zeros(3, 5))
        self.assertTensor(policy[1:, 10:13], [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.4]])
        critic = OBSERVATIONS["critic_command"](self.env, "linerides")
        self.assertEqual(critic.shape, (4, 17))
        self.assertTensor(critic[:, :13], policy)

    def test_drive_only_envs_and_skill_sampling(self):
        self.term.stunt_switch_time_left[:] = 0
        self.term._update_command()
        self.assertTensor(self.term.mode[:2], [0, 0])
        self.assertTrue((self.term.mode[2:] > 0).all())
        self.cfg.skill_sampling_weights = (0.0, 0.0, 1.0)
        self.term._init_guidelines()
        self.term._enter_stunt(torch.arange(4))
        self.assertTensor(self.term.mode, [3, 3, 3, 3])

    def make_curriculum(self, start_stage=0):
        cfg = SimpleNamespace(params={"command_name": "linerides", "start_stage": start_stage})
        return CURRICULUM["LineRidesSkillCurriculum"](cfg, self.env)

    def test_curriculum_records_complete_and_failed_attempts(self):
        curriculum = self.make_curriculum()
        self.term._enter_stunt(torch.tensor([0, 1]), skill_mode=1)
        self.assertTensor(self.term.episode_skill_attempts[:, 1], [1, 1, 0, 0])
        # Reach every remaining waypoint for env 0, then return to DRIVE.
        for _ in range(len(self.paths[1]) - self.cfg.start_waypoint_index):
            self.move_to_goal(0)
            self.term._update_command()
        self.assertEqual(self.term.mode[0].item(), 0)
        self.assertTensor(self.term.episode_skill_successes[:, 1], [1, 0, 0, 0])
        state = curriculum(self.env, torch.tensor([0, 1]), min_attempts=2, consecutive_windows=2)
        self.assertEqual(state["completion_01"], 0.5)
        self.assertEqual(state["passed_windows"], 0)
        self.term._resample_command([0, 1])
        self.assertEqual(self.term.episode_skill_attempts.sum().item(), 0)
        self.assertEqual(self.term.episode_skill_successes.sum().item(), 0)

    def test_curriculum_requires_consecutive_windows_and_preserves_statistics(self):
        curriculum = self.make_curriculum()

        def window(successes):
            self.term.episode_skill_attempts[0, 1] = 4
            self.term.episode_skill_successes[0, 1] = successes
            state = curriculum(self.env, [0], min_attempts=4, consecutive_windows=2)
            self.term._resample_command([0])
            curriculum.reset([0])
            return state

        self.assertEqual(window(3)["passed_windows"], 1)
        self.assertEqual(window(1)["passed_windows"], 0)
        self.assertEqual(window(3)["passed_windows"], 1)
        self.assertEqual(window(3)["stage"], 2)
        self.assertTensor(self.term.skill_sampling_weights, [0.7, 0.3, 0])

    def test_curriculum_gates_current_mode_and_reaches_balanced_stage(self):
        curriculum = self.make_curriculum()
        for mode, expected in ((1, [0.7, 0.3, 0]), (2, [0.45, 0.45, 0.1]), (3, [1, 1, 1])):
            # Success in a different mode cannot promote the current stage.
            other = 2 if mode == 1 else 1
            self.term.episode_skill_attempts[0, other] = 2
            self.term.episode_skill_successes[0, other] = 2
            stage = curriculum.stage
            curriculum(self.env, [0], min_attempts=2, consecutive_windows=1)
            self.assertEqual(curriculum.stage, stage)
            self.term._resample_command([0])
            self.term.episode_skill_attempts[0, mode] = 2
            self.term.episode_skill_successes[0, mode] = 2
            curriculum(self.env, [0], min_attempts=2, consecutive_windows=1)
            self.assertTensor(self.term.skill_sampling_weights, expected)
            self.term._resample_command([0])
        self.assertEqual(curriculum.stage, 3)
        curriculum(self.env, [], min_attempts=2)
        self.assertEqual(curriculum.stage, 3)

    def test_curriculum_sample_minimum_manual_abort_and_initialization(self):
        curriculum = self.make_curriculum()
        self.term.set_mode(1, [0])
        self.term.set_mode(0, [0])
        self.assertEqual(self.term.episode_skill_successes.sum().item(), 0)
        state = curriculum(self.env, [0], min_attempts=2)
        self.assertEqual(state["window_attempts"], 1)
        self.assertEqual(state["stage"], 1)
        self.assertTensor(self.make_curriculum(start_stage=2).command.skill_sampling_weights, [0.45, 0.45, 0.1])
        for invalid in (-1, 4, 0.5, True):
            with self.assertRaises(ValueError):
                self.make_curriculum(invalid)
        for kwargs in ({"min_attempts": 0}, {"min_attempts": 1.5}, {"consecutive_windows": 0},
                       {"success_thresholds": (0.5,)}, {"success_thresholds": (0.5, 0.5, 2)}):
            with self.assertRaises(ValueError):
                curriculum(self.env, [], **kwargs)

    def test_curriculum_ignores_drive_only_resets_and_collects_selected_envs(self):
        curriculum = self.make_curriculum()
        self.term.episode_skill_attempts[1, 1] = 3
        self.term.episode_skill_successes[1, 1] = 2
        state = curriculum(self.env, [0], min_attempts=4)
        self.assertEqual(state["attempts_01"], 0)
        state = curriculum(self.env, [1], min_attempts=4)
        self.assertEqual(state["attempts_01"], 3)
        self.assertAlmostEqual(state["completion_01"], 2 / 3)
        self.assertEqual(state["stage"], 1)

    def test_gamepad_policy_layout_compatibility(self):
        self.term.set_mode(1, [1, 2, 3])
        play = PLAY["LineRidesCommand_Play"].__new__(PLAY["LineRidesCommand_Play"])
        play.__dict__.update(self.term.__dict__)
        self.assertTensor(play.policy_command, self.term.policy_command)

    def test_drive_metrics_are_masked(self):
        for mode in (1, 2, 3):
            self.term.set_mode(mode, [mode])
        self.term.vel_command_b[0] = torch.tensor([0.7, 0.0, 0.4])
        self.term.rho_command[0] = torch.tensor([0.3, 0.4])
        self.term._update_metrics()
        self.assertTensor(self.term.metrics["drive_mode"], [1, 0, 0, 0])
        self.assertTensor(self.term.metrics["drive_lin_vel_error"], [0.7, 0, 0, 0])
        self.assertTensor(self.term.metrics["drive_ang_vel_error"], [0.4, 0, 0, 0])
        self.assertTensor(self.term.metrics["drive_rho_error"], [0.5, 0, 0, 0])

    def test_drive_metrics_average_only_driving_steps_and_reset(self):
        self.term.vel_command_b[:] = 0
        self.term.robot.data.root_lin_vel_b[0, 0] = 1.0
        self.term._update_metrics()
        self.term.robot.data.root_lin_vel_b[0, 0] = 0.0
        self.term._update_metrics()
        self.term.set_mode(1, [0])
        self.term.robot.data.root_lin_vel_b[0, 0] = 100.0
        self.term._update_metrics()
        self.term._update_metrics()
        self.assertAlmostEqual(self.term.metrics["drive_lin_vel_error"][0].item(), 0.5)
        self.assertAlmostEqual(self.term.metrics["drive_lin_vel_bias"][0].item(), 0.5)
        self.assertAlmostEqual(self.term.metrics["drive_tracking_success"][0].item(), 0.5)
        self.assertAlmostEqual(self.term.metrics["drive_mode"][0].item(), 0.5)
        self.assertAlmostEqual(self.term.metrics["stunt_mode"][0].item(), 0.5)
        other_count = self.term._drive_metric_steps[1].clone()
        self.term._resample_command([0])
        self.assertEqual(self.term._drive_metric_steps[0].item(), 0)
        self.assertEqual(self.term._drive_metric_steps[1].item(), other_count.item())
        self.term.robot.data.root_lin_vel_b[0, 0] = 0.0
        self.term.vel_command_b[0] = 0
        self.term._update_metrics()
        self.assertEqual(self.term.metrics["drive_lin_vel_error"][0].item(), 0)
        self.assertEqual(self.term.metrics["drive_tracking_success"][0].item(), 1)

    def test_drive_yaw_command_envelope_and_raw_sample(self):
        self.cfg.drive_max_curvature = 1.5
        self.term.vel_command_b[:, 0] = torch.tensor([0.0, 0.1, -0.1, 2.0])
        self.term.raw_ang_vel_command[:] = torch.tensor([1.0, 1.0, -1.0, 1.0])
        angular_timers = self.term.ang_vel_time_left.clone()
        # The limit depends on the command, never privileged measured speed.
        self.term.robot.data.root_lin_vel_b[:, 0] = 100.0
        self.term._limit_drive_yaw_rate(torch.arange(4))
        self.assertTensor(self.term.vel_command_b[:, 2], [0.0, 0.15, -0.15, 1.0])
        self.assertTensor(self.term.ang_vel_time_left, angular_timers)
        self.assertTensor(self.term.raw_ang_vel_command, [1, 1, -1, 1])
        self.cfg.ranges.lin_vel_x = (0.1, 0.1)
        self.term._sample_linear_velocity(torch.arange(4))
        self.assertTensor(self.term.vel_command_b[:, 2], [0.15, 0.15, -0.15, 0.15])
        self.assertTensor(self.term.ang_vel_time_left, angular_timers)
        self.cfg.ranges.lin_vel_x = (2.0, 2.0)
        self.term._sample_linear_velocity(torch.arange(4))
        self.assertTensor(self.term.vel_command_b[:, 2], [1, 1, -1, 1])
        self.cfg.ranges.ang_vel_z = (0.5, 0.5)
        linear_timers = self.term.lin_vel_time_left.clone()
        self.term._sample_angular_velocity(torch.arange(4))
        self.assertTensor(self.term.vel_command_b[:, 2], [0.5] * 4)
        self.assertTensor(self.term.lin_vel_time_left, linear_timers)
        self.cfg.drive_max_curvature = None
        self.term.vel_command_b[:, 0] = 0
        self.term._limit_drive_yaw_rate(torch.arange(4))
        self.assertTensor(self.term.vel_command_b[:, 2], [0.5] * 4)

    def test_drive_soft_rewards_order_symmetry_and_masks(self):
        self.term.vel_command_b[:] = 0
        self.term._build_command()
        self.term.robot.data.root_lin_vel_b[:, 0] = torch.tensor([0.0, 0.5, 1.0, 2.0])
        self.term.robot.data.root_ang_vel_b[:, 2] = torch.tensor([0.0, -0.5, -1.0, -2.0])
        reward_fn = REWARDS["drive_velocity_tracking_soft"]
        expected = [1.0, 0.8, 1.0 / 1.75, 1.0 / 2.75]
        for component in ("lin_vel_x", "ang_vel_z"):
            reward = reward_fn(self.env, "linerides", component)
            self.assertTensor(reward, expected)
            self.assertTrue((reward[:-1] > reward[1:]).all())
        for mode in (1, 2, 3):
            self.term.set_mode(mode, [mode])
        self.assertTensor(reward_fn(self.env, "linerides", "lin_vel_x"), [1, 0, 0, 0])
        with self.assertRaises(ValueError):
            reward_fn(self.env, "linerides", "bad_axis")
        for parameter in ("beta", "max_error", "error_scale"):
            for value in (0.0, -1.0, math.nan, math.inf):
                with self.assertRaises(ValueError):
                    reward_fn(self.env, "linerides", "lin_vel_x", **{parameter: value})

    def test_drive_soft_reward_is_bounded_for_large_errors(self):
        self.term.vel_command_b[:] = 0
        self.term._build_command()
        self.term.robot.data.root_lin_vel_b[:, 0] = torch.tensor([0.0, 5.0, 10.0, 100.0])
        reward = REWARDS["drive_velocity_tracking_soft"](self.env, "linerides", "lin_vel_x")
        self.assertTensor(reward, [1.0, 1.0 / 5.75, 1.0 / 5.75, 1.0 / 5.75])

    def test_configured_skill_weights_and_curvature_validation(self):
        self.cfg.guideline_files = {"01": self.files[1]}
        self.cfg.skill_sampling_weights = (1.0,)
        self.term._init_guidelines()
        self.assertTensor(self.term.enabled_skill_modes, [1])
        self.assertTensor(self.term.skill_sampling_weights, [1])
        self.cfg.guideline_files = {"11": self.files[3], "01": self.files[1]}
        self.cfg.skill_sampling_weights = (0.0, 1.0)  # Sorted IDs: 01 then 11.
        self.term._init_guidelines()
        self.term._enter_stunt(torch.arange(4))
        self.assertTensor(self.term.mode, [3] * 4)
        for invalid in (0.0, -1.0, math.nan, math.inf):
            self.cfg.drive_max_curvature = invalid
            with self.assertRaises(ValueError):
                COMMANDS["LineRidesCommand"](self.cfg, self.env)

    def test_repository_curves_load_and_disabled_skill_can_be_selected(self):
        curve_dir = MDP.parent / "env"
        self.cfg.guideline_files = {
            "01": str(curve_dir / "mini_hop.json"),
            "10": str(curve_dir / "three_point_turn.json"),
            "11": str(curve_dir / "backflip.json"),
        }
        self.cfg.skill_sampling_weights = (1.0, 1.0, 0.0)
        self.term._init_guidelines()
        self.assertTensor(self.term.guideline_lengths, [0, 12, 25, 25])
        for mode in (1, 2, 3):
            self.term.set_mode(mode, [mode])
            self.assertTensor(self.term.goal_pos_s[mode], self.term.guidelines[mode][1])
            self.assertTensor(self.term.guidelines[mode][0], [0, 0, 0])
        self.assertEqual(self.term.mode[3].item(), 3)  # Zero random weight still allows manual choice.

    def test_unequal_paths_and_final_hit_bonus(self):
        bonus = REWARDS["target_update_bonus"](None, self.env)
        for mode in (1, 2, 3):
            self.term.set_mode(mode, [mode])
        self.assertTensor(bonus(self.env), [0, 0, 0, 0])  # No skill-entry bonus.
        self.move_to_goal(2)  # Mode 2 has only one tracking waypoint.
        self.term._update_command()
        self.assertEqual(self.term.mode[2].item(), 0)
        self.assertTensor(bonus(self.env), [0, 0, 1, 0])
        self.assertTensor(bonus(self.env), [0, 0, 0, 0])  # No repeated bonus.
        self.assertEqual(self.term.current_waypoint_idx[1].item(), 1)
        self.assertEqual(self.term.current_waypoint_idx[3].item(), 1)
        self.assertEqual(len(self.term.guideline_world(3)), 4)
        self.assertTensor(self.term.active_guideline_distance[[1, 3]], [0.5, 0.4])
        for _ in range(3):
            self.move_to_goal(3)
            self.term._update_command()
            self.assertTensor(bonus(self.env), [0, 0, 0, 1])
        self.assertEqual(self.term.mode[3].item(), 0)  # Final target is the origin.
        self.assertEqual(self.term.reached_waypoint_count[3].item(), 3)
        self.term._resample_command([3])
        bonus.reset([3])
        self.assertTensor(bonus(self.env), [0, 0, 0, 0])

    def test_progress_and_origin_targets_in_all_skills(self):
        progress = REWARDS["track_waypoint_line"](None, self.env)
        for mode in (1, 2, 3):
            self.term.set_mode(mode, [mode])
        self.assertTensor(progress(self.env), [0, 0, 0, 0])
        self.term.robot.data.body_pos_w[1, 0, 0] = 0.1
        self.term.robot.data.body_pos_w[2, 0, 1] = 0.1
        self.term.robot.data.body_pos_w[3, 0, 2] = 0.1
        self.assertTensor(progress(self.env), [0, 0.1, 0.1, 0.1])
        self.term.current_waypoint_idx[3] = 3
        self.term.goal_pos_s[3] = 0
        self.term._build_command()
        self.assertTensor(progress(self.env), [0, 0, 0, 0])
        self.term.robot.data.body_pos_w[3, 0, 2] = 0.05
        self.assertAlmostEqual(progress(self.env)[3].item(), 0.05, places=6)

    def test_reward_masks_and_drive_orientation(self):
        for mode in (1, 2, 3):
            self.term.set_mode(mode, [mode])
        self.term.vel_command_b[0] = 0
        self.term.rho_command[0] = 0
        self.term._build_command()
        for name in ("lin_vel_x_exp", "ang_vel_z_exp", "track_rho_exp"):
            kwargs = dict(std=0.5, command_name="linerides")
            if name == "track_rho_exp":
                kwargs["asset_cfg"] = EntityFixture("robot", joint_ids=[0, 1])
            self.assertTensor(REWARDS[name](self.env, **kwargs), [1, 0, 0, 0])
        self.assertTensor(REWARDS["flat_orientation_l2_m"](self.env), [0.13, 0, 0, 0])
        waypoint = REWARDS["track_waypoint_exp"](self.env, std=0.5)
        self.assertEqual(waypoint[0].item(), 0)
        self.assertTrue((waypoint[1:] > 0).all())

    def test_stunt_orientation_only_in_modes_01_and_11(self):
        for mode in (1, 2, 3):
            self.term.set_mode(mode, [mode])
        self.term.robot.data.body_quat_w[:, 0] = torch.tensor([math.sqrt(0.5), math.sqrt(0.5), 0, 0])
        self.assertTensor(REWARDS["stunt_orientation_error_p"](self.env, "linerides"), [0, 0.5, 0, 0.5])

    def test_stall_termination_all_skills_and_origin(self):
        stall = TERMINATIONS["target_not_update_death"](None, self.env)
        for mode in (1, 2, 3):
            self.term.set_mode(mode, [mode])
        self.term.current_waypoint_idx[3] = 3
        self.term.goal_pos_s[3] = 0
        self.term._build_command()
        self.assertTensor(stall(self.env, max_no_update_steps=2), [False] * 4)
        self.assertTensor(stall(self.env, max_no_update_steps=2), [False] * 4)
        self.assertTensor(stall(self.env, max_no_update_steps=2), [False, True, True, True])
        self.term.set_mode(0, [2])
        self.assertFalse(stall(self.env, max_no_update_steps=2)[2])
        stall.reset([3])
        self.assertFalse(stall(self.env, max_no_update_steps=2)[3])

    def test_validation_and_legacy_single_path(self):
        self.cfg.guideline_files = {0: self.files[1]}
        with self.assertRaises(ValueError):
            self.term._init_guidelines()
        self.cfg.guideline_files = None
        self.cfg.guideline_file = self.files[1]
        self.term._init_guidelines()
        self.assertTensor(self.term.enabled_skill_modes, [1])
        with self.assertRaises(ValueError):
            self.term.set_mode(2)
        with self.assertRaises(ValueError):
            self.term.set_mode(4)
        for weights in ((0.0, 0.0, 0.0), (math.nan, 1.0, 1.0), (-1.0, 1.0, 1.0)):
            self.cfg.skill_sampling_weights = weights
            with self.assertRaises(ValueError):
                self.term._init_guidelines()


if __name__ == "__main__":
    unittest.main()
