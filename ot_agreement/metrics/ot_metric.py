"""Image-level OT cost and dataset-level chance-corrected OT agreement."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass

import numpy as np
import ot

from ot_agreement.metrics.annotations import validate_annotations
from ot_agreement.metrics.geometry import giou_matrix
from ot_agreement.metrics.plotting import plot_transport_plan


@dataclass(frozen=True)
class OTChanceCorrectedResult:
    """Dataset-level OT agreement, mean disagreements, and sampling metadata."""

    agreement: float
    observed_disagreement: float
    expected_disagreement: float
    n_observed: int
    n_expected: int
    sample_seed: int


def compute_ot_cost(
    boxes_a: np.ndarray,
    labels_a: np.ndarray,
    boxes_b: np.ndarray,
    labels_b: np.ndarray,
    plot_ot_plan: bool = False,
    lam: float = 0.5,
    beta: float = 1.0,
    label_dist: float = 1.0,
    *,
    return_metric_only: bool = False,
) -> (
    float
    | tuple[
        np.ndarray,
        float,
        float,
        float,
        float,
        float,
        np.ndarray,
    ]
):
    """Compute OT cost between ``[N, 4]`` xyxy boxes with ``[N]`` class IDs.

    Labels must be non-negative integers and boxes must have positive area.
    Return the transport plan and cost components by default, or only total
    cost when ``return_metric_only=True``.
    """
    boxes_a, labels_a = validate_annotations(boxes_a, labels_a, "annotations_a", require_positive_area=True)
    boxes_b, labels_b = validate_annotations(boxes_b, labels_b, "annotations_b", require_positive_area=True)
    if not 0 <= lam <= 1:
        raise ValueError("lam must be in [0, 1]")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if label_dist < 0:
        raise ValueError("label_dist must be non-negative")

    spatial_dist_matr = (1 - giou_matrix(boxes_a, boxes_b)) / 2

    label_mismatch = (labels_a[:, None] != labels_b[None, :]).astype(float)
    label_dist_matr = label_mismatch * label_dist

    cost_matrix = lam * spatial_dist_matr + (1 - lam) * label_dist_matr

    n_1, n_2 = len(boxes_a), len(boxes_b)
    n_unmatched = abs(n_1 - n_2)
    padded_cost = np.full((n_1 + 1, n_2 + 1), beta)
    padded_cost[:n_1, :n_2] = cost_matrix

    source_weights = np.ones(n_1 + 1)
    source_weights[-1] = float(n_2)
    target_weights = np.ones(n_2 + 1)
    target_weights[-1] = float(n_1)
    padded_plan = (
        np.zeros_like(padded_cost) if n_1 == 0 and n_2 == 0 else ot.emd(source_weights, target_weights, padded_cost)
    )

    padded_plan[-1, -1] = 0.0
    plan_sum = padded_plan.sum()
    normalized_plan = padded_plan / plan_sum if plan_sum > 0 else padded_plan
    total_cost = float(np.sum(padded_cost * normalized_plan))

    if return_metric_only and not plot_ot_plan:
        return total_cost

    unmatched_mass = padded_plan[:n_1, n_2].sum() + padded_plan[n_1, :n_2].sum()
    unmatched_ratio = unmatched_mass / plan_sum if plan_sum > 0 else 0.0
    transport_plan = padded_plan[:n_1, :n_2]

    spatial_cost = float(np.sum((transport_plan / plan_sum) * lam * spatial_dist_matr)) if plan_sum > 0 else 0.0
    label_cost = float(np.sum((transport_plan / plan_sum) * (1 - lam) * label_dist_matr)) if plan_sum > 0 else 0.0
    unmatched_plan = padded_plan.copy()
    unmatched_plan[:n_1, :n_2] = 0.0
    unmatched_cost = float(np.sum((unmatched_plan / plan_sum) * padded_cost)) if plan_sum > 0 else 0.0
    matched_distances = transport_plan[transport_plan > 0] * cost_matrix[transport_plan > 0]

    if plot_ot_plan:
        title = (
            f"OT (dustbin)  n1={n_1}, n2={n_2}\n"
            f"Spatial: {spatial_cost:.4f}  "
            f"Label: {label_cost:.4f}  "
            f"Unmatched: {unmatched_cost:.4f}\n"
            f"Unmatched: {n_unmatched}/{n_1 + n_2} boxes "
            f"(ratio={unmatched_ratio:.2f})\n"
            f"Total cost: {total_cost:.4f}"
        )
        plot_transport_plan(
            boxes_a,
            boxes_b,
            padded_plan,
            title=title,
            show=True,
        )

    if return_metric_only:
        return total_cost
    return (
        transport_plan,
        total_cost,
        unmatched_ratio,
        spatial_cost,
        label_cost,
        unmatched_cost,
        matched_distances,
    )


def _sample_cross_image_pairs(
    n_items: int,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    """Sample unique ordered image pairs without constructing the full grid."""
    if n_items < 2:
        raise ValueError("At least two images are required for chance correction")
    if isinstance(sample_size, (bool, np.bool_)) or not isinstance(sample_size, (int, np.integer)) or sample_size <= 0:
        raise ValueError("n_expected_pairs must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("sample_seed must be a non-negative integer")

    total_pairs = n_items * (n_items - 1)
    rng = np.random.default_rng(seed)
    flat_indices = rng.choice(total_pairs, size=min(sample_size, total_pairs), replace=False)
    source_indices = flat_indices // (n_items - 1)
    target_offsets = flat_indices % (n_items - 1)
    target_indices = target_offsets + (target_offsets >= source_indices)
    return np.column_stack([source_indices, target_indices]).astype(np.int64, copy=False)


def _validate_ot_dataset(
    annotations: Mapping[Hashable, tuple[np.ndarray, np.ndarray]],
    image_ids: list[Hashable],
    name: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Validate every image before computing any dataset-level costs."""
    dataset = []
    for image_id in image_ids:
        try:
            boxes, labels = annotations[image_id]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{image_id!r}] must be a (boxes, labels) pair") from exc
        dataset.append(validate_annotations(boxes, labels, f"{name}[{image_id!r}]", require_positive_area=True))
    return dataset


def compute_ot_chance_corrected(
    annotations_a: Mapping[Hashable, tuple[np.ndarray, np.ndarray]],
    annotations_b: Mapping[Hashable, tuple[np.ndarray, np.ndarray]],
    *,
    lam: float = 0.5,
    beta: float = 1.0,
    label_dist: float = 1.0,
    n_expected_pairs: int = 10_000,
    sample_seed: int = 20260828,
    return_metric_only: bool = False,
) -> OTChanceCorrectedResult | float:
    """Compute dataset-level agreement as ``1 - D_o / D_e`` using OT costs.

    Each mapping contains image IDs and pairs of ``[N, 4]`` xyxy boxes and
    ``[N]`` non-negative integer labels. Both mappings must contain the same
    image IDs, with at least two images. Coordinates must already use a common
    scale and labels must use a shared class encoding.

    D_o is the mean cost for matching image IDs, with equal weight per image.
    D_e is the mean cost over up to ``n_expected_pairs`` unique ordered pairs
    of different images. Sampling uses ``sample_seed`` after sorting IDs by
    their string representations. Empty annotations remain part of the dataset.

    Return disagreement means and sampling metadata by default, or only the
    agreement with ``return_metric_only=True``. Values are neither rounded nor
    clipped; agreement can be negative. Non-finite results and non-positive
    expected disagreement raise ValueError.
    """
    if not isinstance(annotations_a, Mapping) or not isinstance(annotations_b, Mapping):
        raise ValueError("annotations_a and annotations_b must be mappings by image ID")
    if annotations_a.keys() != annotations_b.keys():
        raise ValueError("Annotation image ID sets must match")
    image_ids = sorted(annotations_a, key=str)
    chance_pairs = _sample_cross_image_pairs(len(image_ids), n_expected_pairs, sample_seed)
    dataset_a = _validate_ot_dataset(annotations_a, image_ids, "annotations_a")
    dataset_b = _validate_ot_dataset(annotations_b, image_ids, "annotations_b")
    if not np.isfinite([lam, beta, label_dist]).all():
        raise ValueError("lam, beta, and label_dist must be finite")

    observed_sum = 0.0
    for first, second in zip(dataset_a, dataset_b, strict=True):
        observed_sum += compute_ot_cost(
            *first,
            *second,
            lam=lam,
            beta=beta,
            label_dist=label_dist,
            return_metric_only=True,
        )
    expected_sum = 0.0
    for source_index, target_index in chance_pairs:
        expected_sum += compute_ot_cost(
            *dataset_a[source_index],
            *dataset_b[target_index],
            lam=lam,
            beta=beta,
            label_dist=label_dist,
            return_metric_only=True,
        )

    observed = observed_sum / len(image_ids)
    expected = expected_sum / len(chance_pairs)
    if not np.isfinite(observed):
        raise ValueError("Observed disagreement must be finite")
    if not np.isfinite(expected) or expected <= 0:
        raise ValueError("Expected disagreement must be positive and finite")
    agreement = 1.0 - observed / expected
    if not np.isfinite(agreement):
        raise ValueError("Chance-corrected OT agreement must be finite")
    if return_metric_only:
        return float(agreement)
    return OTChanceCorrectedResult(
        agreement=float(agreement),
        observed_disagreement=float(observed),
        expected_disagreement=float(expected),
        n_observed=len(image_ids),
        n_expected=len(chance_pairs),
        sample_seed=int(sample_seed),
    )


__all__ = [
    "OTChanceCorrectedResult",
    "compute_ot_chance_corrected",
    "compute_ot_cost",
    "giou_matrix",
    "plot_transport_plan",
]
