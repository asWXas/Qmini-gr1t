"""Local extensions of upstream Eureka's environment worker."""

import inspect
import traceback
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from isaaclab_eureka.managers import EurekaTaskManager


class QminiTaskManager(EurekaTaskManager):
    def __init__(self, *, num_envs=64, log_root="logs", **kwargs):
        self._num_envs = num_envs
        self._log_root = str(Path(log_root).resolve())
        super().__init__(**kwargs)

    def _create_environment(self):
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=True, device=self._device)
        self._simulation_app = launcher.app
        import gymnasium as gym

        from isaaclab_tasks.utils import parse_env_cfg

        import Qmini  # noqa: F401

        cfg = parse_env_cfg(self._task, device=self._device, num_envs=self._num_envs)
        cfg.seed = self._env_seed
        self._env = gym.make(self._task, cfg=cfg)

    def _worker(self, idx, rewards_queue):
        self._idx = idx
        try:
            self._create_environment()
            env = self._env.unwrapped
            if idx == 0:
                self._observations_queue.put(
                    inspect.getsource(env._get_observations) + "\n" + env.get_eureka_state_schema()
                )
                self._prompt_ready = True
            while not self.termination_event.is_set():
                source = rewards_queue.get()
                if source == "Stop":
                    break
                try:
                    # Reward/reset injection remains upstream implementation.
                    self._prepare_eureka_environment(source)
                    self._run_training()
                    result = {"success": True, "log_dir": self._log_dir}
                except Exception:
                    result = {"success": False, "exception": traceback.format_exc()}
                self._results_queue.put((idx, result))
        except Exception:
            error = traceback.format_exc()
            if idx == 0 and not getattr(self, "_prompt_ready", False):
                self._observations_queue.put("Worker initialization failed:\n" + error)
            # Continue answering requests so other worker failures do not hang train().
            while not self.termination_event.is_set():
                if rewards_queue.get() == "Stop":
                    break
                self._results_queue.put((idx, {"success": False, "exception": error}))
        finally:
            if hasattr(self, "_env"):
                self._env.close()
            if hasattr(self, "_simulation_app"):
                self._simulation_app.close()

    def _run_training(self, framework="rsl_rl"):
        if self._rl_library != "rsl_rl":
            raise ValueError("The local Qmini launcher currently supports RSL-RL only.")
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        cfg = load_cfg_from_registry(self._task, "rsl_rl_cfg_entry_point")
        cfg = handle_deprecated_rsl_rl_cfg(cfg, version("rsl-rl-lib"))
        cfg.device = self._device
        cfg.max_iterations = self._max_training_iterations
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self._log_dir = str(Path(self._log_root) / "rl_runs" / cfg.experiment_name / f"{stamp}_Run-{self._idx}")
        wrapper = RslRlVecEnvWrapper(self._env, clip_actions=cfg.clip_actions)
        runner = OnPolicyRunner(wrapper, cfg.to_dict(), log_dir=self._log_dir, device=cfg.device)
        runner.learn(num_learning_iterations=cfg.max_iterations, init_at_random_ep_len=True)

    def close(self):
        """Bounded cleanup also handles failed workers."""
        self.termination_event.set()
        for queue in self._rewards_queues:
            queue.put("Stop")
        for process in self._processes.values():
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
