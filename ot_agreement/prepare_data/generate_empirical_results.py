"""Generate configured empirical real-real OT matching curves from existing CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from ot_agreement.metrics import compute_ot_cost
from ot_agreement.prepare_data.config import (
    beta_token,
    number_token,
    read_input,
    run_cli,
    variant_filename,
)
from ot_agreement.prepare_data.prepare_datasets import to_image_annotations, write_csv


def match_statistics(first: dict, second: dict, beta: float) -> tuple[int, float]:
    """Return matched-box count and ratio weighted by reference boxes."""
    if first.keys() != second.keys() or not first:
        raise ValueError("Matching annotations must have the same non-empty image set")
    count = int(round(sum(float(compute_ot_cost(*first[i], *second[i], lam=1, beta=beta)[0].sum()) for i in first)))
    reference_count = sum(len(boxes) for boxes, _ in first.values())
    if reference_count == 0:
        raise ValueError("Matching probability requires reference boxes")
    return count, count / reference_count


def generate_results(config: dict) -> list[Path]:
    """Read prepared and perturbed CSV and write only the configured empirical curves."""
    spec = config.get("empirical")
    if not spec:
        return []
    root, dataset = config["data_dir"], spec["dataset"]
    reference = read_input(root / "prepared" / f"{dataset}.csv", "prepare")
    label_map = {label: i for i, label in enumerate(sorted(reference.class_id.unique(), key=str))}
    first = to_image_annotations(reference, label_map)
    beta_scan, gamma_scan = spec["beta_scan"], spec["gamma_scan"]
    levels = sorted(set(beta_scan.get("gamma_percent", [])) | set(gamma_scan.get("gamma_percent", [])))
    paths = []
    for seed in tqdm(spec["seeds"], desc=f"Empirical {dataset}: seeds", unit="seed", dynamic_ncols=True):
        for noise in tqdm(
            spec["noise_types"], desc=f"seed={seed}: noise", unit="distribution", leave=False, dynamic_ncols=True
        ):
            # Cache only within this call to reuse points shared by the two scans.
            statistics = {}
            for gamma in tqdm(levels, desc=f"{noise}: gamma (%)", unit="level", leave=False, dynamic_ncols=True):
                source = (
                    root / "perturbed" / dataset / f"seed_{seed}" / variant_filename(dataset, "shift", gamma, noise)
                )
                second = to_image_annotations(read_input(source, "perturbations"), label_map)
                betas = set(beta_scan.get("beta", [])) if gamma in beta_scan.get("gamma_percent", []) else set()
                if gamma in gamma_scan.get("gamma_percent", []):
                    betas.update(gamma_scan.get("beta", []))
                for beta in tqdm(
                    sorted(betas), desc=f"gamma={gamma:g}%: beta", unit="point", leave=False, dynamic_ncols=True
                ):
                    statistics[gamma, beta] = match_statistics(first, second, beta)
            suffix = "" if noise == "uniform" else "_gauss"
            curves = {}
            for gamma in beta_scan.get("gamma_percent", []):
                curves[f"beta_{dataset}_shift{suffix}_{number_token(gamma)}.csv"] = [
                    {
                        "beta": beta,
                        "num_matched": statistics[gamma, beta][0],
                        "matched_ratio": statistics[gamma, beta][1],
                    }
                    for beta in sorted(beta_scan["beta"])
                ]
            for beta in gamma_scan.get("beta", []):
                curves[f"rel_{dataset}_shift{suffix}_{beta_token(beta)}.csv"] = [
                    {
                        "rel": gamma,
                        "num_matched": statistics[gamma, beta][0],
                        "matched_ratio": statistics[gamma, beta][1],
                    }
                    for gamma in sorted(gamma_scan["gamma_percent"])
                ]
            for name, records in tqdm(
                curves.items(), desc="Write empirical CSV", unit="file", leave=False, dynamic_ncols=True
            ):
                path = root / "results" / "empirical" / f"seed_{seed}" / name
                write_csv(pd.DataFrame(records), path)
                paths.append(path)
    tqdm.write(f"empirical: wrote {len(paths)} curves")
    return paths


def main(argv: list[str] | None = None) -> int:
    """Generate empirical curves from --config YAML."""
    return run_cli(generate_results, argv)


if __name__ == "__main__":
    raise SystemExit(main())
