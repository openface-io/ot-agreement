from collections.abc import Hashable, Mapping

import krippendorff
import numpy as np
from scipy.optimize import linear_sum_assignment

from ot_agreement.metrics.annotations import validate_annotations
from ot_agreement.metrics.geometry import iou_matrix


def compute_krippendorff_alpha(
    boxes_a: np.ndarray,
    labels_a: np.ndarray,
    boxes_b: np.ndarray,
    labels_b: np.ndarray,
    iou_threshold: float = 0.5,
    *,
    return_metric_only: bool = False,
) -> float:
    """Compute Krippendorff's Alpha for a pair of annotators on one image.

    Uses Hungarian matching on IoU to align bounding boxes, then builds a
    reliability matrix where unmatched boxes are coded as "absent" (category 0).
    Labels from the input are shifted by +1 so that 0 is reserved for absence.

    Parameters
    ----------
    boxes_a : np.ndarray, shape (N, 4)
        Boxes from annotator A: [x_min, y_min, x_max, y_max].
    labels_a : np.ndarray, shape (N,)
        Non-negative integer class IDs from annotator A.
    boxes_b : np.ndarray, shape (M, 4)
        Boxes from annotator B, same format.
    labels_b : np.ndarray, shape (M,)
        Non-negative integer class IDs from annotator B.
    iou_threshold : float
        Minimum IoU for a pair to be considered a match.
    return_metric_only : bool, default False
        Accepted for consistency with other metrics. Both modes return alpha
        as a float because this metric has no additional result fields.

    Returns
    -------
    alpha : float
        Krippendorff's Alpha (nominal).  1.0 = perfect agreement,
        0.0 = agreement at chance level, negative = worse than chance.

    Based on https://github.com/Madave94/kalphacv

    """
    boxes_a, labels_a = validate_annotations(boxes_a, labels_a, "annotations_a")
    boxes_b, labels_b = validate_annotations(boxes_b, labels_b, "annotations_b")
    rel = _build_reliability_matrix(boxes_a, labels_a, boxes_b, labels_b, iou_threshold)
    return _safe_alpha(rel)


def compute_krippendorff_alpha_dataset(
    annotations_a: Mapping[Hashable, tuple[np.ndarray, np.ndarray]],
    annotations_b: Mapping[Hashable, tuple[np.ndarray, np.ndarray]],
    iou_threshold: float = 0.5,
    *,
    return_metric_only: bool = False,
) -> float:
    """Compute one nominal Krippendorff's Alpha across a dataset.

    Each mapping contains ``image_id -> (boxes, labels)`` with ``[N, 4]``
    xyxy boxes and ``[N]`` non-negative integer class IDs. The mappings must
    contain the same non-empty set of image IDs. Labels must use a shared
    class encoding, and coordinates must share a scale within each image.

    Empty annotations must have shapes ``[0, 4]`` and ``[0]``. Images empty
    on both sides contribute no columns; an entirely empty annotation set
    across a non-empty dataset returns 1.0. Invalid inputs raise ValueError.
    Negative alpha values are preserved. Both values of ``return_metric_only``
    return a float, as for :func:`compute_krippendorff_alpha`.

    Based on https://github.com/Madave94/kalphacv
    """
    if not isinstance(annotations_a, Mapping) or not isinstance(annotations_b, Mapping):
        raise ValueError("annotations_a and annotations_b must be mappings by image ID")
    if annotations_a.keys() != annotations_b.keys():
        raise ValueError("Annotation image ID sets must match")
    if not annotations_a:
        raise ValueError("The dataset must contain at least one image")
    if (
        not isinstance(iou_threshold, (int, float, np.integer, np.floating))
        or not np.isfinite(iou_threshold)
        or not 0 <= iou_threshold <= 1
    ):
        raise ValueError("iou_threshold must be finite and in [0, 1]")

    matrices = []
    for image_id in annotations_a:
        sources = []
        for name, annotations in (
            ("annotations_a", annotations_a),
            ("annotations_b", annotations_b),
        ):
            try:
                boxes, labels = annotations[image_id]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name}[{image_id!r}] must be a (boxes, labels) pair") from exc
            sources.append(validate_annotations(boxes, labels, f"{name}[{image_id!r}]"))
        matrices.append(_build_reliability_matrix(*sources[0], *sources[1], iou_threshold))

    return _safe_alpha(np.concatenate(matrices, axis=1))


def _build_reliability_matrix(
    boxes_a: np.ndarray,
    labels_a: np.ndarray,
    boxes_b: np.ndarray,
    labels_b: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Align validated annotations for one image, with category 0 for absence."""
    n_a, n_b = len(boxes_a), len(boxes_b)
    if n_a == 0 or n_b == 0:
        # Every box from the non-empty annotator is unmatched
        n = max(n_a, n_b)
        labels = (labels_a if n_a > 0 else labels_b).astype(int) + 1
        if n_a == 0:
            return np.stack([np.zeros(n, dtype=int), labels])
        return np.stack([labels, np.zeros(n, dtype=int)])

    # --- IoU matrix & Hungarian matching ---
    iou_mat = iou_matrix(boxes_a, boxes_b)  # (N, M)
    cost = 1.0 - iou_mat
    row_idx, col_idx = linear_sum_assignment(cost)

    # Keep only matches above IoU threshold
    matched_a = set()
    matched_b = set()
    pairs = []
    for r, c in zip(row_idx, col_idx, strict=False):
        if iou_mat[r, c] >= iou_threshold:
            matched_a.add(r)
            matched_b.add(c)
            pairs.append((r, c))

    unmatched_a = sorted(set(range(n_a)) - matched_a)
    unmatched_b = sorted(set(range(n_b)) - matched_b)
    total_cols = len(pairs) + len(unmatched_a) + len(unmatched_b)

    labels_a = labels_a.astype(int) + 1  # shift so 0 = absent
    labels_b = labels_b.astype(int) + 1

    # Build reliability matrix [2, total_cols]
    rel = np.zeros((2, total_cols), dtype=int)
    col = 0
    for r, c in pairs:
        rel[0, col] = labels_a[r]
        rel[1, col] = labels_b[c]
        col += 1
    for r in unmatched_a:
        rel[0, col] = labels_a[r]
        # rel[1, col] stays 0 (absent)
        col += 1
    for c in unmatched_b:
        # rel[0, col] stays 0 (absent)
        rel[1, col] = labels_b[c]
        col += 1

    return rel


def _safe_alpha(rel):
    """Call krippendorff.alpha with edge-case handling."""
    unique = np.unique(rel)
    if len(unique) <= 1:
        return 1.0
    alpha = krippendorff.alpha(rel, level_of_measurement="nominal")
    if np.isnan(alpha):
        return 0.0
    return float(alpha)


__all__ = ["compute_krippendorff_alpha", "compute_krippendorff_alpha_dataset"]
