import numpy as np
import pandas as pd
import pytest

from ot_agreement.prepare_data.generate_perturbed_datasets import shift_boxes


@pytest.mark.parametrize("seed", [42, 69, 420])
@pytest.mark.parametrize("noise_type", ["uniform", "gauss"])
def test_zero_shift_preserves_annotations(seed, noise_type):
    """Keep fractional and out-of-frame coordinates at zero noise."""
    frame = pd.DataFrame(
        {"x_min": [-0.25], "x_max": [1.75], "y_min": [4.5], "y_max": [5.5], "image_width": [10], "image_height": [10]}
    )
    actual = shift_boxes(frame, 0.0, np.random.default_rng(seed), noise_type)
    pd.testing.assert_frame_equal(actual, frame)
    assert actual is not frame
