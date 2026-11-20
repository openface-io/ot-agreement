import pytest

from ot_agreement.create_paper_figures import main


@pytest.mark.parametrize("figure_group", ["matching", "discrepancy"])
def test_figure_cli_fails_when_results_are_missing(tmp_path, capsys, figure_group):
    """Return a failure when a requested plot group has no usable data."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--data-dir", str(tmp_path), "--figures", figure_group])
    assert exc_info.value.code == 1
    output = capsys.readouterr()
    assert figure_group in output.err
    assert not (tmp_path / "paper_plots").exists()


def test_figure_cli_accepts_one_matching_scan(tmp_path):
    """Render a matching figure when the data provides a beta scan only."""
    theory = tmp_path / "results" / "theor"
    theory.mkdir(parents=True)
    (theory / "theor_beta_shift_uni_30.csv").write_text("beta,cdf\n0,0\n1,1\n")
    assert main(["--data-dir", str(tmp_path), "--figures", "matching"]) == 0
    assert (tmp_path / "paper_plots" / "cdf_theory.pdf").stat().st_size > 0
