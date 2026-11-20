import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ot_agreement.create_paper_figures import get_distances_to_bounds, plot_discrepancy_agreement


def test_score_distances_use_analytical_bounds():
    """Classify exact boundary scores at ratios between grid points."""
    data = get_distances_to_bounds([1 / 3, 3, 1 / 3], [0.4, 0.4, 0.401], lam=0.7, beta=0.9)
    assert data["upper_bound"].tolist() == pytest.approx([0.4, 0.4, 0.4])
    assert data["within_bounds"].tolist() == [True, True, False]
    assert data["rel_dist_to_lower"].iloc[0] == pytest.approx(1.0)


@pytest.mark.parametrize("scores", [(0.1, 0.1, 0.1), (0.1, 0.1, 0.2)])
def test_plot_discrepancy_handles_identical_seed_scores(tmp_path, monkeypatch, scores):
    """Plot zero-length error bars when every seed has the same score."""
    monkeypatch.setattr("ot_agreement.create_paper_figures.PAPER_PLOTS_DIR", tmp_path / "plots")
    for seed, score in zip((42, 69, 420), scores, strict=True):
        path = tmp_path / "acne04_orig" / f"seed_{seed}" / "discrep_drop.csv"
        path.parent.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "dataset": "acne04_orig",
                    "seed": seed,
                    "perturbation": "drop",
                    "noise_type": "uniform",
                    "level": 10,
                    "metric_name": "f1",
                    "metric_kind": "f1",
                    "lambda": "",
                    "beta": "",
                    "label_dist": "",
                    "iou_threshold": 0.5,
                    "transform": "",
                    "n_expected_pairs": "",
                    "chance_sample_seed": "",
                    "scope": "image",
                    "image_id": "image.jpg",
                    "value": score,
                }
            ]
        ).to_csv(path, index=False)

    figure, _, summary = plot_discrepancy_agreement(results_dir=tmp_path, show=False)
    try:
        assert summary["mean"].item() == pytest.approx(sum(scores) / len(scores))
        for extension in ("jpg", "pdf"):
            assert (tmp_path / "plots" / f"discrepancy_acne04_orig_drop_uniform.{extension}").stat().st_size > 0
    finally:
        plt.close(figure)
