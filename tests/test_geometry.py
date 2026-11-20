import numpy as np
import pytest

from ot_agreement.metrics import compute_ot_cost
from ot_agreement.metrics.geometry import giou_matrix, iou_matrix


@pytest.mark.parametrize("scale", [1.0, 1e-5])
def test_overlap_is_invariant_to_coordinate_scale(scale):
    """Preserve overlap ratios and OT self-agreement after rescaling."""
    first = np.array([[0.0, 0.0, 1.0, 1.0]]) * scale
    second = np.array([[0.5, 0.0, 1.5, 1.0]]) * scale
    np.testing.assert_allclose(iou_matrix(first, second), [[1 / 3]])
    np.testing.assert_allclose(giou_matrix(first, second), [[1 / 3]])
    labels = np.array([0])
    assert compute_ot_cost(first, labels, first, labels, return_metric_only=True) == 0.0

    disjoint = np.array([[2.0, 0.0, 3.0, 1.0]]) * scale
    np.testing.assert_allclose(giou_matrix(first, disjoint), [[-1 / 3]])
    cost = compute_ot_cost(first, labels, disjoint, labels, lam=1.0, return_metric_only=True)
    assert cost == pytest.approx(2 / 3)
