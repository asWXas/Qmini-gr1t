"""Local task descriptions and evaluation targets for Eureka."""

TASKS = {
    "Qmini-Flat-Eureka-v0": {
        "description": (
            "The goal is natural, human-like walking and running for the GR1T2 humanoid, "
            "with coordinated whole-body motion and stable balance. Track commanded "
            "body-frame planar and yaw velocities: stand calmly near zero commands, "
            "walk naturally at low speeds, and transition smoothly toward running when "
            "the commanded speed warrants it. Do not reward moving faster than commanded. "
            "Prioritize avoiding falls and maintaining controllable velocity tracking, "
            "then improve gait quality and whole-body coordination. Encourage alternating "
            "left-right steps, suitable swing-foot clearance, controlled landings, and "
            "a stable pelvis and torso without excessive lateral sway or violent rotation. "
            "Encourage relaxed, coordinated arm and leg motion, including opposite-arm "
            "and leg swing when the available observations identify those joints. "
            "Allow speed-dependent stride length, cadence, moderate forward lean, and "
            "flight phases during running; do not enforce a rigid pose, a fixed cadence "
            "across all speeds, permanent ground contact, or identical left-right motion "
            "during turns. Discourage shuffling, foot dragging or sliding, persistent "
            "crouching, excessive hopping, flailing, and high-frequency joint oscillations "
            "where these behaviors can be measured from the provided state. Avoid "
            "penalties so strong that standing still becomes preferable to commanded locomotion. "
            "Use only self.eureka_state and the supplied observation schema. Do not invent "
            "joint indices, contact signals, previous-step values, reference motions, or "
            "other unavailable state; omit objectives that cannot be measured reliably. "
            "Account for observation scaling and clipping; clipped values cannot be fully "
            "recovered and relative joint values are not absolute joint angles. "
            "Return (total_reward, reward_terms_dict) with finite floating-point tensors "
            "of shape [self.num_envs] on self.device. Compute rewards independently for "
            "each parallel environment. Return per-step rewards, multiplying reward rates "
            "and their logged terms by self.step_dt exactly once. Do not modify environment "
            "state or recompute observations inside the reward function."
        ),
        "success_metric": "self.compute_eureka_success(env_ids)",
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.15,
    },
}
