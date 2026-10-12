"""Backflip position loop from the figure, starting exactly at [0, 0, 0].

The loop does NOT command a 360-degree body rotation. Existing upright stunt
rewards need separate changes for actual backflip training. Z=0 is the initial
base height, not world-ground height. The original jump remains --curve jump.
"""

import numpy as np

if __package__:
    from .curve_utils import HermiteConfig, cubic_hermite_curve, curve_cli, generate_piecewise_guideline
    from .curve_utils import save_guideline_json as _save_guideline_json
else:
    from curve_utils import HermiteConfig, cubic_hermite_curve, curve_cli, generate_piecewise_guideline
    from curve_utils import save_guideline_json as _save_guideline_json

__all__ = [
    "HermiteConfig",
    "cubic_hermite_curve",
    "generate_piecewise_guideline",
    "save_guideline_json",
    "build_backflip_segments",
    "build_jump_segments",
]


def build_backflip_segments():
    """Approach -> narrow vertical position loop -> landing -> exit."""
    origin = np.array([0.0, 0.0, 0.0])
    takeoff = np.array([1.3, 0.0, 0.0])
    center_x, radius_x = 1.95, 0.16
    bottom_z, top_z = 0.15, 1.10
    center_z, radius_z = (bottom_z + top_z) / 2, (top_z - bottom_z) / 2
    bottom = np.array([center_x, 0.0, bottom_z])
    right = np.array([center_x + radius_x, 0.0, center_z])
    top = np.array([center_x, 0.0, top_z])
    left = np.array([center_x - radius_x, 0.0, center_z])
    # Four cubic ellipse quarters: Bezier tangent coefficient multiplied by 3.
    factor = 4 * (np.sqrt(2) - 1)
    tangent_x = np.array([factor * radius_x, 0.0, 0.0])
    tangent_z = np.array([0.0, 0.0, factor * radius_z])
    landing = np.array([2.6, 0.0, 0.0])
    finish = np.array([3.1, 0.0, 0.0])
    return [
        HermiteConfig(origin, takeoff, np.array([1.3, 0.0, 0.0]), np.array([0.8, 0.0, 0.0]), num_dense=400),
        HermiteConfig(takeoff, bottom, np.array([0.6, 0.0, 0.0]), tangent_x, num_dense=400),
        HermiteConfig(bottom, right, tangent_x, tangent_z, num_dense=300),
        HermiteConfig(right, top, tangent_z, -tangent_x, num_dense=300),
        HermiteConfig(top, left, -tangent_x, -tangent_z, num_dense=300),
        HermiteConfig(left, bottom, -tangent_z, tangent_x, num_dense=300),
        HermiteConfig(bottom, landing, tangent_x, np.array([0.6, 0.0, 0.0]), num_dense=400),
        HermiteConfig(landing, finish, np.array([0.6, 0.0, 0.0]), np.array([0.5, 0.0, 0.0]), num_dense=400),
    ]


def build_jump_segments():
    """Original jump generator; no implicit overwrite of mini_hop.json."""
    return [
        HermiteConfig(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            num_dense=500,
        ),
        HermiteConfig(
            np.array([1.0, 0.0, 0.0]),
            np.array([3.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 4.0]),
            np.array([2.0, 0.0, -4.0]),
            num_dense=1000,
        ),
        HermiteConfig(
            np.array([3.0, 0.0, 0.0]),
            np.array([4.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            num_dense=500,
        ),
    ]


def save_guideline_json(guideline, filename="ballistic_jump_guideline.json", **kwargs):
    """Keep the original point2 export API, with explicit overwrite opt-in."""
    kwargs.setdefault("name", "ballistic_jump_guideline")
    kwargs.setdefault("description", "Piecewise Hermite curve jump trajectory")
    return _save_guideline_json(guideline, filename, **kwargs)


if __name__ == "__main__":
    curve_cli(
        {
            "backflip": (
                build_backflip_segments,
                "backflip",
                "Approximate Backflip position loop from the supplied figure; not a body-rotation reference.",
                25,
            ),
            "jump": (
                build_jump_segments,
                "ballistic_jump_guideline",
                "Original piecewise Hermite jump position curve.",
                12,
            ),
        },
        default_curve="backflip",
    )
