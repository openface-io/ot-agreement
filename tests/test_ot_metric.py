import numpy as np
import pytest

from ot_agreement.metrics import compute_ot_cost


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
