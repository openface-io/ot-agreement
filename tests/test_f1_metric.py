import numpy as np
import pytest

from ot_agreement.metrics import compute_f1


@pytest.mark.parametrize(("target_label", "expected"), [(0, (1, 0, 0, 1.0)), (1, (0, 1, 1, 0.0))])
def test_f1_requires_matching_labels_at_zero_threshold(target_label, expected):
    """Only same-class boxes can match when the threshold is zero."""
    first = np.array([[0, 0, 1, 1]])
    second = np.array([[1, 0, 2, 1]])
    assert compute_f1(first, np.array([0]), second, np.array([target_label]), iou_threshold=0.0) == expected


@pytest.mark.parametrize("threshold", [float("nan"), -0.1, 1.1])
def test_f1_rejects_invalid_thresholds(threshold):
    """Reject IoU thresholds outside the supported interval."""
    boxes = np.array([[0, 0, 1, 1]])
    labels = np.array([0])
    with pytest.raises(ValueError, match="iou_threshold"):
        compute_f1(boxes, labels, boxes, labels, iou_threshold=threshold)
