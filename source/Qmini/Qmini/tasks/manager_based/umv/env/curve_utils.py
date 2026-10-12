"""Standalone Hermite guideline tools. No IsaacLab dependency.

Samples are spaced by curve arc length, not equal straight-line chord length.
All exported curves must start at the stunt-frame origin [0, 0, 0].
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class HermiteConfig:
    x0: np.ndarray
    x1: np.ndarray
    m0: np.ndarray
    m1: np.ndarray
    num_dense: int = 1000
    num_waypoints: int = 20


def cubic_hermite_curve(u_array, cfg):
    """Evaluate a cubic Hermite segment as an [N, 3] array."""
    u = np.asarray(u_array, dtype=float).reshape(-1, 1)
    u2, u3 = u**2, u**3
    return (
        (2 * u3 - 3 * u2 + 1) * np.asarray(cfg.x0)
        + (u3 - 2 * u2 + u) * np.asarray(cfg.m0)
        + (-2 * u3 + 3 * u2) * np.asarray(cfg.x1)
        + (u3 - u2) * np.asarray(cfg.m1)
    )


def generate_piecewise_guideline(segments_cfg, num_waypoints=25, preserve_junctions=False):
    """Sample a connected path, optionally retaining reversal/loop extrema.

    Default: equal arc spacing across the whole path (legacy jump behavior).
    Retaining junctions: allocate intervals by segment length and sample at
    equal arc spacing within each segment. Opposite junction tangents are
    allowed for stop/reverse maneuvers.
    """
    configs = list(segments_cfg)
    if not configs or not isinstance(num_waypoints, int) or num_waypoints < 2:
        raise ValueError("Provide segments and at least two waypoints.")
    if preserve_junctions and num_waypoints < len(configs) + 1:
        raise ValueError("Preserving junctions needs at least num_segments + 1 waypoints.")
    parts, lengths = [], []
    for index, cfg in enumerate(configs):
        for name in ("x0", "x1", "m0", "m1"):
            value = np.asarray(getattr(cfg, name), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"Segment {index}: {name} must be a finite 3-D vector.")
        if not isinstance(cfg.num_dense, int) or cfg.num_dense < 2:
            raise ValueError("num_dense must be an integer >= 2.")
        if index and not np.allclose(configs[index - 1].x1, cfg.x0, atol=1e-8, rtol=0):
            raise ValueError(f"Segments {index - 1}/{index} are not connected.")
        points = cubic_hermite_curve(np.linspace(0, 1, cfg.num_dense), cfg)
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        if not np.isfinite(length) or length <= 1e-10:
            raise ValueError(f"Segment {index} has invalid or zero arc length.")
        parts.append(points if index == 0 else points[1:])
        lengths.append(length)
    dense = np.vstack(parts)
    arcs = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))))
    if preserve_junctions:
        lengths = np.asarray(lengths)
        extras = (num_waypoints - 1 - len(lengths)) * lengths / lengths.sum()
        counts = np.ones(len(lengths), dtype=int) + np.floor(extras).astype(int)
        remainder = num_waypoints - 1 - int(counts.sum())
        counts[np.argsort(-(extras - np.floor(extras)), kind="stable")[:remainder]] += 1
        targets, offset = [], 0.0
        for index, (length, count) in enumerate(zip(lengths, counts, strict=True)):
            values = offset + np.linspace(0, length, count + 1)
            targets.append(values if index == 0 else values[1:])
            offset += length
        targets = np.concatenate(targets)
    else:
        targets = np.linspace(0, arcs[-1], num_waypoints)
    distinct = np.concatenate(([True], np.diff(arcs) > 1e-12))
    points = np.column_stack([np.interp(targets, arcs[distinct], dense[distinct, axis]) for axis in range(3)])
    points[0], points[-1] = configs[0].x0, configs[-1].x1
    guideline = [{"p": point, "d": float(distance)} for point, distance in zip(points, targets, strict=True)]
    return guideline, dense, arcs


def generate_guideline(cfg):
    """Preserve point1's original single-segment API."""
    return generate_piecewise_guideline([cfg], cfg.num_waypoints)


def guideline_payload(guideline, name, description):
    points = np.asarray([wp["p"] for wp in guideline], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Guideline must contain at least two finite 3-D points.")
    if not np.allclose(points[0], 0, atol=1e-8, rtol=0):
        raise ValueError("Guideline must start at [0, 0, 0].")
    points = np.round(points, 4)
    points[0] = 0.0
    return {
        "name": name,
        "frame": "stunt_start",
        "description": description,
        "position_only": True,
        "waypoints": points.tolist(),
    }


def save_guideline_json(guideline, filename, *, name="guideline", description="", overwrite=False):
    """Export the existing command schema; never silently overwrite a file."""
    payload = guideline_payload(guideline, name, description)
    with Path(filename).open("w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def plot_guideline(guideline, dense, title, filename=None, show=False):
    """Preview positions; blue arrows represent travel, not body rotation."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    points = np.asarray([wp["p"] for wp in guideline])
    ax.plot(*dense.T, color="black", linewidth=1.8, label="Hermite path")
    ax.scatter(*points.T, color="red", s=22, depthshade=False, label=f"Waypoints ({len(points)})")
    for fraction in (0.12, 0.4, 0.65, 0.88):
        index = min(int(fraction * len(dense)), len(dense) - 2)
        direction = dense[index + 1] - dense[index]
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            ax.quiver(*dense[index], *(direction / norm), length=0.3, color="dodgerblue", linewidth=2)
    ax.scatter(*points[0], marker="s", color="green", s=60, label="Start (0, 0, 0)")
    ax.scatter(*points[-1], marker="X", color="orange", s=65, label="Finish")
    ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Relative Z (m)", title=title)
    if np.ptp(points[:, 1]) < 1e-8:
        ax.set_ylim(-0.45, 0.45)
    ax.set_zlim(-0.03, max(1.0, float(dense[:, 2].max()) * 1.1))
    ax.view_init(elev=24, azim=-65)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    if filename:
        fig.savefig(filename)
    if show:
        plt.show()
    plt.close(fig)


def curve_cli(curves, default_curve):
    """Each curve entry is (builder, name, description, default_waypoint_count)."""
    parser = argparse.ArgumentParser(description="Generate a position-only guideline starting at (0, 0, 0).")
    parser.add_argument("--curve", choices=curves, default=default_curve)
    parser.add_argument("--num-waypoints", type=int)
    parser.add_argument("--output", help="JSON path, or '-' for JSON on stdout.")
    parser.add_argument("--plot", help="Optional preview image path.")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--force", action="store_true", help="Allow replacing existing output files.")
    args = parser.parse_args()
    builder, name, description, default_count = curves[args.curve]
    output = args.output or str(Path(__file__).with_name(f"{name}.json"))
    if args.plot and output != "-" and Path(output).resolve() == Path(args.plot).resolve():
        parser.error("JSON and plot outputs must use different paths.")
    for target in (output, args.plot):
        if target and target != "-" and Path(target).exists() and not args.force:
            parser.error(f"Output exists: {target}. Use --force to replace it.")
    try:
        guideline, dense, _ = generate_piecewise_guideline(
            builder(),
            args.num_waypoints if args.num_waypoints is not None else default_count,
            preserve_junctions=args.curve != "jump",
        )
        payload = guideline_payload(guideline, name, description)
    except ValueError as error:
        parser.error(str(error))
    if output == "-":
        json.dump(payload, sys.stdout, indent=2, allow_nan=False)
        print()
    else:
        save_guideline_json(guideline, output, name=name, description=description, overwrite=args.force)
        print(f"Saved {len(guideline)} waypoints: {output}")
    if args.plot or args.show:
        plot_guideline(guideline, dense, name.replace("_", " ").title(), args.plot, args.show)
