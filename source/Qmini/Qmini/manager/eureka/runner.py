"""Local constructor with upstream reward-search and feedback loop."""

from datetime import datetime
from pathlib import Path

from isaaclab_eureka.config import DIRECT_WORKFLOW_INITIAL_PROMPT
from isaaclab_eureka.eureka import Eureka
from isaaclab_eureka.managers import LLMManager

from .task_config import TASKS
from .task_manager import QminiTaskManager


class QminiEureka(Eureka):
    def __init__(
        self, task, gpt_model=None, num_envs=64, device="cuda:0", env_seed=42,
        num_parallel_runs=1, max_training_iterations=100, feedback_subsampling=10,
        temperature=None, log_root="logs", llm_config=None,
    ):
        # Upstream constructor hardcodes TaskManager and TASKS_CFG. Supply local
        # dependencies here; inherit run(), feedback extraction and result logs.
        task_cfg = TASKS[task]
        self._task_description = task_cfg["description"]
        self._success_metric_to_win = task_cfg["success_metric_to_win"]
        self._success_metric_tolerance = task_cfg["success_metric_tolerance"]
        self._feedback_subsampling = feedback_subsampling
        self._num_processes = num_parallel_runs
        if llm_config is not None:
            from .llm_manager import QminiLLMManager

            self._llm_manager = QminiLLMManager(
                config=llm_config, num_suggestions=num_parallel_runs,
                system_prompt=DIRECT_WORKFLOW_INITIAL_PROMPT,
            )
        else:
            self._llm_manager = LLMManager(
                gpt_model=gpt_model, num_suggestions=num_parallel_runs,
                temperature=1.0 if temperature is None else temperature,
                system_prompt=DIRECT_WORKFLOW_INITIAL_PROMPT,
            )
        self._task_manager = QminiTaskManager(
            task=task, device=device, env_seed=env_seed, rl_library="rsl_rl",
            num_processes=num_parallel_runs, max_training_iterations=max_training_iterations,
            success_metric_string=task_cfg["success_metric"], num_envs=num_envs, log_root=log_root,
        )
        if self._task_manager.get_observations_method_as_string.startswith("Worker initialization failed:"):
            self._task_manager.close()
            raise RuntimeError(self._task_manager.get_observations_method_as_string)
        from torch.utils.tensorboard import SummaryWriter

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self._log_dir = str(Path(log_root).resolve() / "eureka" / task / stamp)
        Path(self._log_dir).mkdir(parents=True)
        self._tensorboard_writer = SummaryWriter(log_dir=self._log_dir, flush_secs=10)

    def run(self, max_eureka_iterations):
        try:
            return super().run(max_eureka_iterations)
        finally:
            self._task_manager.close()
            self._tensorboard_writer.close()
