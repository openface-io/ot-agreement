"""Generate configured discrepancy tables directly for the original paper plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from ot_agreement.metrics import (
    compute_f1,
    compute_krippendorff_alpha,
    compute_miou,
    compute_ot_chance_corrected,
    compute_ot_cost,
)
from ot_agreement.prepare_data.config import read_input, run_cli, variant_filename
from ot_agreement.prepare_data.prepare_datasets import to_image_annotations, write_csv

COLUMNS = [
    "dataset",
    "seed",
    "perturbation",
    "noise_type",
    "level",
    "metric_name",
    "metric_kind",
    "lambda",
    "beta",
    "label_dist",
    "iou_threshold",
    "transform",
    "scope",
    "image_id",
    "value",
    "n_orig",
    "n_perturbed",
    "observed_disagreement",
    "expected_disagreement",
    "n_observed",
    "n_expected",
    "n_expected_pairs",
    "chance_sample_seed",
]


def compute_discrepancy_results(
    reference: pd.DataFrame,
    comparisons_by_level: dict[float, pd.DataFrame],
    dataset: str,
    seed: int,
    operation: str,
    *,
    metrics: dict[str, dict],
    noise_type: str = "uniform",
    n_expected_pairs: int = 10000,
    chance_sample_seed: int = 20260828,
) -> pd.DataFrame:
    """Save image scores and one dataset-level OT coefficient with their actual parameters."""
    labels = sorted(
        set(reference.class_id).union(*(set(f.class_id) for f in comparisons_by_level.values())),
        key=str,
    )
    label_map = {label: i for i, label in enumerate(labels)}
    first = to_image_annotations(reference, label_map, normalize=True)
    raw_first = to_image_annotations(reference, label_map)
    shapes = reference.groupby("file_name")[["image_height", "image_width"]].first()
    records = []
    for level, comparison in tqdm(sorted(comparisons_by_level.items())):
        second = to_image_annotations(comparison, label_map, normalize=True)
        raw_second = to_image_annotations(comparison, label_map)
        if first.keys() != second.keys() or not first:
            raise ValueError("Discrepancy annotations must have the same non-empty image set")
        for name, metric in metrics.items():
            kind = metric["kind"]
            metadata = {
                "dataset": dataset,
                "seed": seed,
                "perturbation": operation,
                "noise_type": noise_type,
                "level": level,
                "metric_name": name,
                "metric_kind": kind,
            }
            if kind in {"ot", "ot_chance_corrected"}:
                metadata.update({key: metric[key] for key in ("lambda", "beta", "label_dist")})
            if kind in {"f1", "kalpha"}:
                metadata["iou_threshold"] = metric["iou_threshold"]
            if kind == "kalpha":
                metadata["transform"] = metric["transform"]
            if kind == "ot_chance_corrected":
                metadata.update(
                    n_expected_pairs=n_expected_pairs,
                    chance_sample_seed=chance_sample_seed,
                )
                chance = compute_ot_chance_corrected(
                    first,
                    second,
                    lam=metric["lambda"],
                    beta=metric["beta"],
                    label_dist=metric["label_dist"],
                    n_expected_pairs=n_expected_pairs,
                    sample_seed=chance_sample_seed,
                )
                records.append(
                    {
                        **metadata,
                        "scope": "dataset",
                        "image_id": "",
                        "value": chance.agreement,
                        "n_orig": sum(len(boxes) for boxes, _ in first.values()),
                        "n_perturbed": sum(len(boxes) for boxes, _ in second.values()),
                        "observed_disagreement": chance.observed_disagreement,
                        "expected_disagreement": chance.expected_disagreement,
                        "n_observed": chance.n_observed,
                        "n_expected": chance.n_expected,
                    }
                )
            for image in sorted(first, key=str):
                if kind in {"ot", "ot_chance_corrected"}:
                    value = 1 - compute_ot_cost(
                        *first[image],
                        *second[image],
                        lam=metric["lambda"],
                        beta=metric["beta"],
                        label_dist=metric["label_dist"],
                        return_metric_only=True,
                    )
                elif kind == "f1":
                    value = compute_f1(
                        *first[image],
                        *second[image],
                        iou_threshold=metric["iou_threshold"],
                        return_metric_only=True,
                    )
                elif kind == "kalpha":
                    alpha = compute_krippendorff_alpha(
                        *first[image],
                        *second[image],
                        iou_threshold=metric["iou_threshold"],
                    )
                    value = (alpha + 1) / 2 if metric["transform"] == "unit" else alpha
                else:
                    value = compute_miou(
                        *raw_first[image],
                        *raw_second[image],
                        img_shape=tuple(shapes.loc[image]),
                        num_classes=len(label_map),
                        return_metric_only=True,
                    )
                records.append(
                    {
                        **metadata,
                        "scope": "image",
                        "image_id": image,
                        "value": value,
                        "n_orig": len(first[image][0]),
                        "n_perturbed": len(second[image][0]),
                    }
                )
    return pd.DataFrame(records, columns=COLUMNS)


def generate_results(config: dict) -> list[Path]:
    """Read existing inputs and write the discrepancy configurations selected in YAML."""
    root = config["data_dir"]
    prepared, paths = {}, []
    for job in tqdm(config.get("discrepancy", []), desc="Discrepancy", unit="job", dynamic_ncols=True):
        dataset, operation = job["dataset"], job["operation"]
        if dataset not in prepared:
            prepared[dataset] = read_input(root / "prepared" / f"{dataset}.csv", "prepare")
        metrics = {name: config["metrics"][name] for name in job["metrics"]}
        for seed in tqdm(
            job["seeds"],
            desc=f"{dataset}/{operation}/{job['noise_type']}: seeds",
            unit="seed",
            leave=False,
            dynamic_ncols=True,
        ):
            comparisons = {
                level: read_input(
                    root
                    / "perturbed"
                    / dataset
                    / f"seed_{seed}"
                    / variant_filename(dataset, operation, level, job["noise_type"]),
                    "perturbations",
                )
                for level in tqdm(
                    job["levels"], desc="Read perturbed CSV", unit="file", leave=False, dynamic_ncols=True
                )
            }
            result = compute_discrepancy_results(
                prepared[dataset],
                comparisons,
                dataset,
                seed,
                operation,
                metrics=metrics,
                noise_type=job["noise_type"],
                n_expected_pairs=job["n_expected_pairs"],
                chance_sample_seed=job["chance_sample_seed"],
            )
            suffix = "_gauss" if job["noise_type"] == "gauss" else ""
            path = root / "results" / "discrepancy" / dataset / f"seed_{seed}" / f"discrep_{operation}{suffix}.csv"
            write_csv(result, path)
            paths.append(path)
            tqdm.write(f"{dataset}/seed_{seed}/{operation}: wrote {len(result)} rows to {path}")
    return paths


def main(argv: list[str] | None = None) -> int:
    """Generate discrepancy results from --config YAML."""
    return run_cli(generate_results, argv)


if __name__ == "__main__":
    raise SystemExit(main())
