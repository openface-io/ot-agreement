"""Run the selected data generators in their explicit, fixed order."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from ot_agreement.prepare_data import (
    generate_discrepancy_results,
    generate_empirical_results,
    generate_perturbed_datasets,
    generate_theoretical_results,
    prepare_datasets,
)
from ot_agreement.prepare_data.config import DEFAULT_CONFIG, STAGES, load_config

GENERATORS = {
    "prepare": prepare_datasets.generate_results,
    "perturbations": generate_perturbed_datasets.generate_results,
    "theory": generate_theoretical_results.generate_results,
    "empirical": generate_empirical_results.generate_results,
    "discrepancy": generate_discrepancy_results.generate_results,
}


def main(argv: list[str] | None = None) -> int:
    """Load YAML once and call selected generators without inferring dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML config (default: project configs/paper_plots_repro.yaml)",
    )
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    args = parser.parse_args(argv)
    started = perf_counter()
    try:
        config = load_config(args.config)
        for stage, generator in GENERATORS.items():
            if stage in args.stages:
                print(f"[{stage}]")
                paths = generator(config)
                print(f"Wrote {len(paths)} CSV files")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"Total execution time: {(perf_counter() - started) / 60:.2f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
