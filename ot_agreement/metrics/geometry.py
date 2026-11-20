"""Geometry funcs for bounding boxes in xyxy format."""

from __future__ import annotations

import numpy as np


def _as_box_array(boxes: np.ndarray, name: str) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"{name} must have shape [N, 4]")
    return boxes


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise intersection over union."""
    boxes_a = _as_box_array(boxes_a, "boxes_a")
    boxes_b = _as_box_array(boxes_b, "boxes_b")
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=float)

    x_1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y_1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x_2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y_2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    intersection = np.maximum(x_2 - x_1, 0) * np.maximum(y_2 - y_1, 0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / np.maximum(union, 1e-8)


def giou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise Generalized IoU."""
    boxes_a = _as_box_array(boxes_a, "boxes_a")
    boxes_b = _as_box_array(boxes_b, "boxes_b")
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=float)

    x_1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y_1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x_2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y_2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    intersection = np.maximum(x_2 - x_1, 0) * np.maximum(y_2 - y_1, 0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection

    enclosing_x_1 = np.minimum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    enclosing_y_1 = np.minimum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    enclosing_x_2 = np.maximum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    enclosing_y_2 = np.maximum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)
    enclosing_area = (enclosing_x_2 - enclosing_x_1) * (enclosing_y_2 - enclosing_y_1)

    iou = intersection / np.maximum(union, 1e-8)
    return iou - (enclosing_area - union) / np.maximum(
        enclosing_area,
        1e-8,
    )


__all__ = ["giou_matrix", "iou_matrix"]
