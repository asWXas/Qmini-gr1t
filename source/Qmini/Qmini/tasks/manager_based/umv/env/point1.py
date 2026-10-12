"""ThreePoint-Turn position curve approximated from the supplied figure.

Both this path and the other skills start at [0, 0, 0]. These are illustrative
coordinates, not recovered original paper data. Run directly without Isaac Sim.
"""

import numpy as np

if __package__:
    from .curve_utils import HermiteConfig, cubic_hermite_curve, curve_cli, generate_guideline
else:
    from curve_utils import HermiteConfig, cubic_hermite_curve, curve_cli, generate_guideline

__all__ = ["HermiteConfig", "cubic_hermite_curve", "generate_guideline", "build_three_point_turn_segments"]


def build_three_point_turn_segments():
    """Approach, reverse leg and short exit, retaining both reversal points."""
    origin = np.array([0.0, 0.0, 0.0])
    upper_turn = np.array([2.5, 1.4, 0.45])
    lower_turn = np.array([3.8, 0.0, 0.0])
    finish = np.array([3.35, 0.0, 0.0])
    upper_tangent = np.array([0.6, 1.5, 0.65])
    lower_tangent = np.array([0.8, -0.15, 0.0])
    return [
        HermiteConfig(origin, upper_turn, np.array([3.5, 0.0, 0.0]), upper_tangent, num_dense=700),
        HermiteConfig(upper_turn, lower_turn, -upper_tangent, lower_tangent, num_dense=700),
        HermiteConfig(lower_turn, finish, -lower_tangent, np.array([-0.8, 0.0, 0.0]), num_dense=300),
    ]


if __name__ == "__main__":
    curve_cli(
        {
            "three-point-turn": (
                build_three_point_turn_segments,
                "three_point_turn",
                "Approximate three-leg ThreePoint-Turn from the supplied figure; position-only, with two reversals.",
                25,
            )
        },
        default_curve="three-point-turn",
    )
