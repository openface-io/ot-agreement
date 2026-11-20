import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ot_agreement.create_paper_figures import plot_discrepancy_agreement


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
