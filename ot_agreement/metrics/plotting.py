"""Visualization of optimal-transport plans."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .annotations import validate_boxes


def _split_transport_plan(
    transport_plan: np.ndarray,
    n_a: int,
    n_b: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    plan = np.asarray(transport_plan, dtype=float)
    if plan.shape == (n_a, n_b):
        return plan, plan, False
    if plan.shape == (n_a + 1, n_b + 1):
        return plan[:n_a, :n_b], plan, True
    raise ValueError(f"transport_plan must have shape {(n_a, n_b)} or padded shape {(n_a + 1, n_b + 1)}")


def _draw_boxes(
    axis: Axes,
    boxes: np.ndarray,
    *,
    color: str,
    linestyle: str,
) -> None:
    for box in boxes:
        axis.add_patch(
            patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                edgecolor=color,
                facecolor="none",
                linewidth=1.8,
                linestyle=linestyle,
                zorder=2,
            )
        )


def _set_box_axis_limits(
    axis: Axes,
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> None:
    boxes = np.vstack([boxes_a, boxes_b])
    if len(boxes) == 0:
        axis.set_xlim(0, 1)
        axis.set_ylim(1, 0)
        return
    x_min = float(boxes[:, 0].min())
    y_min = float(boxes[:, 1].min())
    x_max = float(boxes[:, 2].max())
    y_max = float(boxes[:, 3].max())
    x_margin = max((x_max - x_min) * 0.05, 1e-6)
    y_margin = max((y_max - y_min) * 0.05, 1e-6)
    axis.set_xlim(x_min - x_margin, x_max + x_margin)
    axis.set_ylim(y_max + y_margin, y_min - y_margin)


def plot_transport_plan(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
    transport_plan: np.ndarray,
    *,
    title: str | None = None,
    match_threshold: float = 0.0,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Plot two sets of ``[N, 4]`` xyxy boxes and their transport plan.

    Boxes must have positive area. The plan may include a dustbin row and column.
    """
    boxes_a = validate_boxes(boxes_a, "boxes_a", require_positive_area=True)
    boxes_b = validate_boxes(boxes_b, "boxes_b", require_positive_area=True)
    real_plan, displayed_plan, is_padded = _split_transport_plan(
        transport_plan,
        len(boxes_a),
        len(boxes_b),
    )
    if not np.isfinite(displayed_plan).all():
        raise ValueError("transport_plan contains non-finite values")
    if (displayed_plan < 0).any():
        raise ValueError("transport_plan contains negative values")
    if match_threshold < 0:
        raise ValueError("match_threshold must be non-negative")

    figure, axes = plt.subplots(2, 1, figsize=(10, 10))
    boxes_axis, plan_axis = axes

    _draw_boxes(
        boxes_axis,
        boxes_a,
        color="C0",
        linestyle="-",
    )
    _draw_boxes(
        boxes_axis,
        boxes_b,
        color="C1",
        linestyle="--",
    )

    centres_a = np.column_stack(
        (
            (boxes_a[:, 0] + boxes_a[:, 2]) / 2,
            (boxes_a[:, 1] + boxes_a[:, 3]) / 2,
        )
    )
    centres_b = np.column_stack(
        (
            (boxes_b[:, 0] + boxes_b[:, 2]) / 2,
            (boxes_b[:, 1] + boxes_b[:, 3]) / 2,
        )
    )
    positive_mass = real_plan[real_plan > match_threshold]
    max_mass = float(positive_mass.max()) if positive_mass.size else 1.0
    for index_a, index_b in np.argwhere(real_plan > match_threshold):
        mass = real_plan[index_a, index_b]
        boxes_axis.plot(
            [centres_a[index_a, 0], centres_b[index_b, 0]],
            [centres_a[index_a, 1], centres_b[index_b, 1]],
            color="C2",
            alpha=0.8,
            linewidth=1 + 2 * mass / max_mass,
            zorder=1,
        )

    _set_box_axis_limits(boxes_axis, boxes_a, boxes_b)
    boxes_axis.set_aspect("equal", adjustable="box")
    boxes_axis.set_xlabel("x")
    boxes_axis.set_ylabel("y")
    boxes_axis.grid(alpha=0.25)
    boxes_axis.set_title(title or "Matched bounding boxes")
    boxes_axis.legend(
        handles=[
            patches.Patch(
                facecolor="none",
                edgecolor="C0",
                label="annotation 1",
            ),
            patches.Patch(
                facecolor="none",
                edgecolor="C1",
                linestyle="--",
                label="annotation 2",
            ),
            Line2D([0], [0], color="C2", label="matched"),
        ],
        loc="best",
    )

    image = plan_axis.imshow(
        displayed_plan,
        cmap="jet",
        aspect="auto",
        interpolation="nearest",
    )
    plan_axis.set_xlabel("annotation 2" + (" + dustbin" if is_padded else ""))
    plan_axis.set_ylabel("annotation 1" + (" + dustbin" if is_padded else ""))
    plan_axis.set_xticks(np.arange(displayed_plan.shape[1]))
    plan_axis.set_yticks(np.arange(displayed_plan.shape[0]))
    if is_padded:
        plan_axis.set_xticklabels(
            [
                *(str(index) for index in range(len(boxes_b))),
                "dustbin",
            ]
        )
        plan_axis.set_yticklabels(
            [
                *(str(index) for index in range(len(boxes_a))),
                "dustbin",
            ]
        )
    plan_axis.set_title("Transport plan")
    figure.colorbar(image, ax=plan_axis, label="transported mass")
    figure.tight_layout()

    if show:
        plt.show()
    return figure, axes


__all__ = ["plot_transport_plan"]
