import pandas as pd
import pytest

from ot_agreement.prepare_data.generate_theoretical_results import DEFAULT_N_X, generate_results


@pytest.mark.parametrize(
    ("n_x", "noise_type", "beta", "gamma_percent", "expected"),
    [
        pytest.param(DEFAULT_N_X, "uniform", 1.0, 110.0, 1.0, id="default-uniform-endpoint"),
        pytest.param(DEFAULT_N_X, "gauss", 1.0, 110.0, 1.0, id="default-gauss-endpoint"),
        pytest.param(2, "uniform", 0.0, 110.0, 0.0, id="zero-endpoint"),
        pytest.param(2, "uniform", 0.75, 20.0, 1.0, id="coarse-grid-overshoot"),
        pytest.param(
            DEFAULT_N_X,
            "uniform",
            0.5,
            110.0,
            pytest.approx(0.34062520926536055, rel=1e-10),
            id="in-range-interior",
        ),
    ],
)
def test_theoretical_curve_probability(tmp_path, n_x, noise_type, beta, gamma_percent, expected):
    """Write bounded probabilities in both scans, with exact beta endpoints."""
    scan = {"beta": [beta], "gamma_percent": [gamma_percent]}
    paths = generate_results(
        {
            "data_dir": tmp_path,
            "theory": {
                "n_x": n_x,
                "noise_types": [noise_type],
                "beta_scan": scan,
                "gamma_scan": scan,
            },
        }
    )

    assert len(paths) == 2
    for path in paths:
        probability = pd.read_csv(path)["cdf"].item()
        assert probability == expected
