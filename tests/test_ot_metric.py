import numpy as np
import pytest

from ot_agreement.metrics import compute_ot_cost


@pytest.mark.parametrize("beta", [1.0, 1, np.int64(1)])
def test_ot_preserves_fractional_cost_with_integer_beta(beta):
    """Use the same fractional spatial cost for integer and floating-point beta."""
    first = np.array([[0.0, 0.0, 2.0, 1.0]])
    second = np.array([[1.0, 0.0, 3.0, 1.0]])
    labels = np.array([0])
    cost = compute_ot_cost(first, labels, second, labels, lam=1.0, beta=beta, return_metric_only=True)
    assert cost == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "options",
    [{"beta": float("nan")}, {"beta": float("inf")}, {"label_dist": float("nan")}, {"label_dist": float("inf")}],
)
def test_ot_rejects_nonfinite_cost_parameters(options):
    """Validate cost parameters even when both annotation sets are empty."""
    boxes = np.empty((0, 4))
    labels = np.empty(0, dtype=int)
    with pytest.raises(ValueError, match=next(iter(options))):
        compute_ot_cost(boxes, labels, boxes, labels, **options)


def test_transport_plan_title_counts_boxes_matched_to_dummies(monkeypatch):
    """Count unmatched boxes even when both sources contain the same number."""
    titles = []

    def capture_plot(*args, title, **kwargs):
        titles.append(title)

    monkeypatch.setattr("ot_agreement.metrics.ot_metric.plot_transport_plan", capture_plot)
    first = np.array([[0.0, 0.0, 1.0, 1.0]])
    second = np.array([[2.0, 0.0, 3.0, 1.0]])
    labels = np.array([0])
    _, _, unmatched_ratio, *_ = compute_ot_cost(first, labels, second, labels, plot_ot_plan=True, lam=1.0, beta=0.2)
    assert unmatched_ratio == 1.0
    assert "Unmatched: 2/2 boxes" in titles[0]
