"""NumPy validation for separate bounding-box and class-label arrays."""

from __future__ import annotations

import numpy as np


def validate_boxes(
    boxes: np.ndarray,
    name: str = "boxes",
    *,
    require_positive_area: bool = False,
) -> np.ndarray:
    """Validate finite ``[N, 4]`` xyxy coordinates and return them as floats.

    Require positive width and height only when requested by the caller.
    The input is never modified.
    """
    boxes = np.asarray(boxes)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"{name} must have shape [N, 4]")
    if boxes.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric coordinates")
    boxes = boxes.astype(float, copy=False)
    if not np.isfinite(boxes).all():
        raise ValueError(f"{name} contains non-finite values")
    if require_positive_area:
        invalid_geometry = (boxes[:, 0] >= boxes[:, 2]) | (boxes[:, 1] >= boxes[:, 3])
        if invalid_geometry.any():
            raise ValueError(f"{name} contains invalid bounding boxes")
    return boxes


def validate_annotations(
    boxes: np.ndarray,
    labels: np.ndarray,
    name: str = "annotations",
    *,
    require_positive_area: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate ``[N, 4]`` boxes and ``[N]`` non-negative integer class IDs.

    Return float coordinates and labels with their original integer dtype.
    Empty inputs must have shapes ``[0, 4]`` and ``[0]``. Neither input is modified.
    """
    boxes = validate_boxes(boxes, f"{name}.boxes", require_positive_area=require_positive_area)
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"{name}.labels must have shape [N]")
    if len(boxes) != len(labels):
        raise ValueError(f"{name}: boxes and labels must have the same length")
    if labels.dtype.kind not in "iu" or (labels < 0).any():
        raise ValueError(f"{name}.labels must contain non-negative integer class IDs")
    return boxes, labels


__all__ = ["validate_annotations", "validate_boxes"]
