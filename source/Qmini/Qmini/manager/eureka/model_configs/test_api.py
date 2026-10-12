"""Probe one real LLM reward generation without starting PPO training.

Run from the Qmini root with the same environment variable used for training:
python source/Qmini/Qmini/manager/eureka/model_configs/test_api.py --config deepseek
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import sys
from pathlib import Path


EUREKA_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EUREKA_DIR))
# Prefer an already installed package. Fall back to the user's pristine clone
# so this diagnostic can run from the Qmini root without an editable install.
UPSTREAM_SOURCE = Path(
    os.environ.get(
        "ISAACLAB_EUREKA_SOURCE",
        "/home/wx/Env/Isaac/IsaacLabEureka/source/isaaclab_eureka",
    )
).expanduser()
if UPSTREAM_SOURCE.is_dir():
    sys.path.append(str(UPSTREAM_SOURCE))

from llm_manager import QminiLLMManager  # noqa: E402
from model_configs import load_profile  # noqa: E402
from task_config import TASKS  # noqa: E402

PROMPT_TEMPLATES = UPSTREAM_SOURCE / "isaaclab_eureka" / "config" / "prompt_templates.py"
if not PROMPT_TEMPLATES.is_file():
    raise RuntimeError(
        f"Cannot find Eureka prompt templates at {PROMPT_TEMPLATES}. "
        "Set ISAACLAB_EUREKA_SOURCE to its source/isaaclab_eureka directory."
    )
spec = importlib.util.spec_from_file_location("qmini_eureka_prompt_templates", PROMPT_TEMPLATES)
prompt_templates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompt_templates)
DIRECT_WORKFLOW_INITIAL_PROMPT = prompt_templates.DIRECT_WORKFLOW_INITIAL_PROMPT
DIRECT_WORKFLOW_TASK_PROMPT = prompt_templates.DIRECT_WORKFLOW_TASK_PROMPT


# This mirrors the fields exposed by the current GR1T critic group. The real
# environment adds exact slices/scales at startup; this probe tests the same
# Eureka code-generation contract without allocating simulation/GPU resources.
OBSERVATION_CONTRACT = '''def _get_observations(self):
    """Use self.eureka_state, computed before reward and reset.

    Available vector fields:
      base_lin_vel[3], base_ang_vel[3], base_projected_gravity[3],
      torso_ang_vel[3], torso_projected_gravity[3], velocity_commands[3],
      joint_pos_rel[25], joint_vel_rel[25], gait_phase[2], gait_phase_h[2],
      gait_phase_l[2], actions[25], height_scan[187], air_time[2],
      feet_height[2], contact_force_violation[2].
    Values may be scaled or clipped; use only these named fields. Return
    (total_reward, reward_terms_dict), with every tensor shaped [num_envs].
    """
    return self.observation_manager.compute(update_history=False)
'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Test one real Eureka LLM reward-generation request")
    parser.add_argument("--config", default="deepseek", help="Profile name or JSON path")
    parser.add_argument("--task", default="Qmini-Flat-Eureka-v0")
    parser.add_argument("--model", help="Override the profile model")
    parser.add_argument("--timeout", type=float, help="Override request timeout in seconds")
    args = parser.parse_args()

    config = load_profile(args.config, model=args.model)
    if args.timeout is not None:
        if args.timeout <= 0:
            parser.error("--timeout must be positive")
        config["timeout_seconds"] = args.timeout
    key_name = config["api_key_env"]
    if not os.environ.get(key_name):
        parser.error(f"environment variable {key_name} is not set")
    task = TASKS.get(args.task)
    if task is None:
        parser.error(f"unknown task: {args.task}")

    user_prompt = DIRECT_WORKFLOW_TASK_PROMPT.format(
        task_description=task["description"],
        success_metric_to_win=task["success_metric_to_win"],
        get_observations_method_as_string=OBSERVATION_CONTRACT,
    )
    print(
        f"[TEST] provider={config['provider']} model={config['model']} "
        f"base_url={config.get('base_url')} timeout={config.get('timeout_seconds', 300)}s",
        flush=True,
    )
    print(
        f"[TEST] prompt: system={len(DIRECT_WORKFLOW_INITIAL_PROMPT)} chars, "
        f"user={len(user_prompt)} chars; API key loaded from {key_name}",
        flush=True,
    )
    manager = QminiLLMManager(config, num_suggestions=1, system_prompt=DIRECT_WORKFLOW_INITIAL_PROMPT)
    result = manager.prompt(user_prompt)
    raw = result["raw_outputs"][0]
    code = result["reward_strings"][0]
    if not code.strip():
        print("[FAIL] API responded, but no ```python ... ``` block could be extracted.")
        print("[RAW RESPONSE PREVIEW]")
        print(raw[:2000])
        return 2
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        print(f"[FAIL] Extracted Python is invalid: {exc}")
        print(code[:4000])
        return 3
    functions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if "_get_rewards_eureka" not in functions:
        print(f"[FAIL] Python was extracted, but functions are {functions!r}.")
        print(code[:4000])
        return 4
    print(f"[PASS] Valid _get_rewards_eureka received ({len(raw)} response chars, {len(code)} code chars).")
    print("[EXTRACTED REWARD CODE]")
    print(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
