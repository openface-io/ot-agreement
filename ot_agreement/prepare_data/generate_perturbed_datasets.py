"""Generate configured annotation perturbations with deterministic RNG."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ot_agreement.prepare_data.config import read_input, run_cli, variant_filename
from ot_agreement.prepare_data.prepare_datasets import write_csv


def drop_boxes(
    df: pd.DataFrame,
    drop_ratio: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Remove boxes independently while keeping at least one per image."""
    assert 0.0 <= drop_ratio < 1.0, "drop_ratio must be in [0, 1)"

    kept = []
    for _, group in df.groupby("file_name", sort=False):
        n = len(group)
        mask = rng.random(n) >= drop_ratio
        if mask.sum() == 0:
            keep_idx = rng.integers(n)
            mask[keep_idx] = True
        kept.append(group[mask])

    return pd.concat(kept, ignore_index=True)


def shift_boxes(
    df: pd.DataFrame,
    max_shift_rel: float,
    rng: np.random.Generator,
    noise_type: str = "uniform",
) -> pd.DataFrame:
    """Shift boxes by size-relative offsets and clip nonzero shifts to the image."""
    df = df.copy()
    n = len(df)
    box_w = (df["x_max"] - df["x_min"]).values
    box_h = (df["y_max"] - df["y_min"]).values

    if noise_type == "uniform":
        dx = rng.uniform(-max_shift_rel, max_shift_rel, n) * box_w
        dy = rng.uniform(-max_shift_rel, max_shift_rel, n) * box_h
    elif noise_type == "gauss":
        dx = rng.normal(0, max_shift_rel, n) * box_w
        dy = rng.normal(0, max_shift_rel, n) * box_h
    else:
        raise ValueError("Invalid noise type")

    if max_shift_rel == 0:
        return df

    df["x_min"] = (df["x_min"] + dx).clip(lower=0, upper=df["image_width"] - 1).astype(int)
    df["x_max"] = (df["x_max"] + dx).clip(lower=1, upper=df["image_width"]).astype(int)
    df["y_min"] = (df["y_min"] + dy).clip(lower=0, upper=df["image_height"] - 1).astype(int)
    df["y_max"] = (df["y_max"] + dy).clip(lower=1, upper=df["image_height"]).astype(int)

    swap_x = df["x_min"] >= df["x_max"]
    df.loc[swap_x, ["x_min", "x_max"]] = df.loc[swap_x, ["x_max", "x_min"]].values

    swap_y = df["y_min"] >= df["y_max"]
    df.loc[swap_y, ["y_min", "y_max"]] = df.loc[swap_y, ["y_max", "y_min"]].values

    return df


def swap_labels(
    df: pd.DataFrame,
    swap_ratio: float,
    label_col: str,
    rng: np.random.Generator,
    paired_label_col: str | None = None,
) -> pd.DataFrame:
    """Swap labels in pairs, optionally applying the same swaps to a paired label."""
    assert 0.0 <= swap_ratio <= 1.0, "swap_ratio must be in [0, 1]"
    df = df.copy()

    label_columns = [label_col]
    if paired_label_col is not None and paired_label_col != label_col:
        label_columns.append(paired_label_col)

    for _, group in df.groupby("file_name", sort=False):
        n = len(group)
        k = round(swap_ratio * n)
        if k % 2 != 0:
            k -= 1
        if k < 2:
            continue

        chosen_pos = rng.choice(n, size=k, replace=False)
        chosen_idx = group.index[chosen_pos]

        for column in label_columns:
            labels = df.loc[chosen_idx, column].values.copy()
            for pair_start in range(0, k, 2):
                labels[pair_start], labels[pair_start + 1] = (
                    labels[pair_start + 1],
                    labels[pair_start],
                )
            df.loc[chosen_idx, column] = labels

    return df


def generate_results(config: dict) -> list[Path]:
    """Read prepared CSV and generate exactly the perturbations listed in YAML."""
    prepared, done, paths = {}, set(), []
    root = config["data_dir"]
    for job in tqdm(config.get("perturbations", []), desc="Perturbations", unit="job", dynamic_ncols=True):
        dataset, operation, noise = job["dataset"], job["operation"], job["noise_type"]
        if dataset not in prepared:
            prepared[dataset] = read_input(root / "prepared" / f"{dataset}.csv", "prepare")
        for seed in tqdm(
            job["seeds"], desc=f"{dataset}/{operation}/{noise}: seeds", unit="seed", leave=False, dynamic_ncols=True
        ):
            for level in tqdm(
                job["levels"], desc=f"seed={seed}: levels (%)", unit="level", leave=False, dynamic_ncols=True
            ):
                key = (dataset, operation, noise, seed, level)
                if key in done:
                    continue
                rng = np.random.default_rng(seed)
                frame = prepared[dataset]
                if operation == "shift":
                    result = shift_boxes(frame, level / 100, rng, noise)
                elif operation == "drop":
                    result = drop_boxes(frame, level / 100, rng)
                else:
                    result = swap_labels(frame, level / 100, "class_id", rng, "class_name")
                path = (
                    root / "perturbed" / dataset / f"seed_{seed}" / variant_filename(dataset, operation, level, noise)
                )
                write_csv(result, path)
                paths.append(path)
                done.add(key)
                tqdm.write(f"{dataset}/seed_{seed}: wrote {path.name}")
    return paths


def main(argv: list[str] | None = None) -> int:
    """Generate selected perturbations from --config YAML."""
    return run_cli(generate_results, argv)


if __name__ == "__main__":
    raise SystemExit(main())
