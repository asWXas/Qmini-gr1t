"""Run from the Qmini root: python source/Qmini/Qmini/manager/eureka/train.py --help."""

import argparse
import importlib
import multiprocessing
import sys
import types
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Qmini Manager-based Eureka (RSL-RL)")
    parser.add_argument("--task", default="Qmini-Flat-Eureka-v0")
    parser.add_argument("--gpt_model")
    parser.add_argument("--llm_config", help="Bundled model profile name or path to a JSON profile.")
    parser.add_argument("--smoke_test", action="store_true", help="Train a fixed reward without calling an LLM.")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--env_seed", type=int, default=42)
    parser.add_argument("--num_parallel_runs", type=int, default=1)
    parser.add_argument("--max_training_iterations", type=int, default=100)
    parser.add_argument("--max_eureka_iterations", type=int, default=1)
    parser.add_argument("--feedback_subsampling", type=int, default=10)
    parser.add_argument("--temperature", type=float, help="Override profile temperature; legacy default is 1.0.")
    parser.add_argument("--log_root", default="logs")
    args = vars(parser.parse_args())
    for name in ("num_envs", "num_parallel_runs", "max_training_iterations",
                 "max_eureka_iterations", "feedback_subsampling"):
        if args[name] < 1:
            parser.error(f"{name} must be positive")
    iterations = args.pop("max_eureka_iterations")
    if args.pop("smoke_test"):
        module = importlib.import_module("qmini_eureka_local.task_manager")
        task_cfg = importlib.import_module("qmini_eureka_local.task_config").TASKS[args["task"]]
        manager = module.QminiTaskManager(
            task=args["task"], device=args["device"], env_seed=args["env_seed"],
            num_envs=args["num_envs"], num_processes=args["num_parallel_runs"],
            max_training_iterations=args["max_training_iterations"], rl_library="rsl_rl",
            success_metric_string=task_cfg["success_metric"], log_root=args["log_root"],
        )
        try:
            prompt = manager.get_observations_method_as_string
            if prompt.startswith("Worker initialization failed:"):
                raise RuntimeError(prompt)
            if "scale=" not in prompt:
                raise RuntimeError("Observation schema missing from worker prompt")
            source = (
                "def _get_rewards_eureka(self):\n"
                "    value = torch.ones(self.num_envs, device=self.device) * self.step_dt\n"
                "    return value, {'constant': value}\n"
            )
            results = manager.train([source] * args["num_parallel_runs"])
            for result in results:
                if not result["success"]:
                    raise RuntimeError(result["exception"])
                print("Smoke test passed:", result["log_dir"], flush=True)
        finally:
            manager.close()
        return
    if args["llm_config"]:
        loader = importlib.import_module("qmini_eureka_local.model_configs")
        try:
            args["llm_config"] = loader.load_profile(
                args["llm_config"], model=args["gpt_model"], temperature=args["temperature"],
            )
        except (OSError, ValueError, TypeError) as exc:
            parser.error(str(exc))
    elif not args["gpt_model"]:
        parser.error("--llm_config or --gpt_model is required unless --smoke_test is used")
    runner = importlib.import_module("qmini_eureka_local.runner").QminiEureka(**args)
    runner.run(iterations)


# Load the local extensions without importing Qmini's task package before Kit.
# This namespace is recreated when multiprocessing spawn imports this script.
package = types.ModuleType("qmini_eureka_local")
package.__path__ = [str(Path(__file__).resolve().parent)]
sys.modules["qmini_eureka_local"] = package

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
