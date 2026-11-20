import numpy as np
import torch
from torchvision.ops import box_iou

from ot_agreement.metrics.annotations import validate_annotations


def compute_f1(
    boxes_a: np.ndarray,
    labels_a: np.ndarray,
    boxes_b: np.ndarray,
    labels_b: np.ndarray,
    iou_threshold: float = 0.5,
    *,
    return_metric_only: bool = False,
) -> float | tuple[int, int, int, float]:
    """Compare [N, 4] boxes in xyxy format and [N] integer labels.

    Set A is the reference; set B is evaluated against it.

    Return (tp, fp, fn, f1) by default, or only F1 as a float when
    return_metric_only=True. F1 is 0 when both sets are empty.
    """
    boxes_a, labels_a = validate_annotations(boxes_a, labels_a, "annotations_a")
    boxes_b, labels_b = validate_annotations(boxes_b, labels_b, "annotations_b")
    class_match_matrix = torch.from_numpy(labels_a[:, None] == labels_b[None, :])
    boxes_a = torch.from_numpy(boxes_a.astype(np.float32, copy=True))
    boxes_b = torch.from_numpy(boxes_b.astype(np.float32, copy=True))

    # Compute IoU.
    iou_matrix = box_iou(boxes_a, boxes_b)

    # Zero IoU for pairs with different classes.
    # Valid pairs must meet the IoU threshold and have matching classes.
    valid_iou_matrix = iou_matrix * class_match_matrix

    # Greedy matching.
    matched_a = torch.zeros(len(boxes_a), dtype=torch.bool)
    matched_b = torch.zeros(len(boxes_b), dtype=torch.bool)

    # Sort pairs by descending IoU.
    values, indices = valid_iou_matrix.flatten().sort(descending=True)

    tp = 0
    m = len(boxes_b)
    for val, idx in zip(values, indices, strict=False):
        if val < iou_threshold:
            break

        idx_a = idx.item() // m
        idx_b = idx.item() % m

        if not matched_a[idx_a] and not matched_b[idx_b]:
            matched_a[idx_a] = True
            matched_b[idx_b] = True
            tp += 1

    fp = len(boxes_b) - tp
    fn = len(boxes_a) - tp
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    if return_metric_only:
        return float(f1)
    return tp, fp, fn, f1


__all__ = ["compute_f1"]
