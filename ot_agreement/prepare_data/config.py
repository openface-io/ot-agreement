"""Load the five data-generation sections and share small CSV helpers."""

from __future__ import annotations

import argparse
import math
import re
from decimal import Decimal
from pathlib import Path

import pandas as pd
import yaml

STAGES = ("prepare", "perturbations", "theory", "empirical", "discrepancy")
DATASETS = {"acne04_orig", "acne04_v2", "acnescu", "texbig_v1"}
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "paper_plots_repro.yaml"


def grid(value: list | dict, name: str = "grid") -> list[float]:
    """Expand an explicit or inclusive decimal grid without rounding its values."""
    if isinstance(value, dict):
        if set(value) != {"start", "stop", "step"}:
            raise ValueError(f"{name}: expected start, stop, step")
        try:
            start, stop, step = (Decimal(str(value[k])) for k in ("start", "stop", "step"))
            if not all(v.is_finite() for v in (start, stop, step)) or step <= 0 or stop < start:
                raise ValueError(f"{name}: invalid grid range")
            count = int((stop - start) // step) + 1
        except ArithmeticError as exc:
            raise ValueError(f"{name}: invalid grid range") from exc
        if count > 1_000_000:
            raise ValueError(f"{name}: grid is too large")
        value = [float(start + i * step) for i in range(count)]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name}: expected a non-empty grid")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        raise ValueError(f"{name}: values must be numbers")
    result = [float(v) for v in value]
    if not all(map(math.isfinite, result)) or len(set(result)) != len(result):
        raise ValueError(f"{name}: values must be finite and unique")
    return result


def _fields(value, required: set, optional: set, name: str) -> None:
    if not isinstance(value, dict) or required - value.keys() or value.keys() - required - optional:
        raise ValueError(f"{name}: required fields {sorted(required)}; optional fields {sorted(optional)}")


def _integer(value, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name}: expected an integer >= {minimum}")


def _seeds(job: dict) -> None:
    seeds = job.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds: expected a non-empty list")
    for seed in seeds:
        _integer(seed, "seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")


def load_config(path: str | Path) -> dict:
    """Read YAML into a plain dictionary, resolving paths and validating parameters."""
    path = Path(path).resolve()
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML: {exc}") from exc
    _fields(config, {"data_dir"}, {"datasets", "metrics", *STAGES}, "config")
    if not isinstance(config["data_dir"], str):
        raise ValueError("data_dir: expected a path string")
    config["data_dir"] = (path.parent / config["data_dir"]).resolve()
    datasets = config.setdefault("datasets", {})
    metrics = config.setdefault("metrics", {})
    if not isinstance(datasets, dict) or set(datasets) - DATASETS:
        raise ValueError("datasets: only built-in datasets are supported")
    for name, data in datasets.items():
        _fields(data, {"source"}, set(), name)
        if not isinstance(data["source"], str):
            raise ValueError(f"{name}.source: expected a path string")
        data["source"] = (path.parent / data["source"]).resolve()
    if not isinstance(metrics, dict):
        raise ValueError("metrics: expected named configurations")
    for name, metric in metrics.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Invalid metric name {name!r}")
        _fields(
            metric,
            {"kind"},
            {"lambda", "beta", "label_dist", "iou_threshold", "transform"},
            name,
        )
        if metric["kind"] not in {"ot", "ot_chance_corrected", "f1", "kalpha", "miou"}:
            raise ValueError(f"{name}: unknown metric kind")
        for key, default in (
            ("lambda", 1.0),
            ("beta", 0.5),
            ("label_dist", 1.0),
            ("iou_threshold", 0.5),
        ):
            value = metric.setdefault(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name}.{key}: expected a finite number")
            if value < 0 or (key != "label_dist" and value > 1):
                raise ValueError(f"{name}.{key}: invalid range")
        metric.setdefault("transform", "identity")
        if metric["transform"] not in {"identity", "unit"} or (
            metric["transform"] == "unit" and metric["kind"] != "kalpha"
        ):
            raise ValueError(f"{name}: unsupported transform")
    prepare = config.get("prepare") or []
    if not isinstance(prepare, list) or any(name not in datasets for name in prepare):
        raise ValueError("prepare: expected a list of configured datasets")
    config["prepare"] = list(dict.fromkeys(prepare))
    for stage in ("perturbations", "discrepancy"):
        jobs = config.get(stage) or []
        if not isinstance(jobs, list):
            raise ValueError(f"{stage}: expected a list of configurations")
        targets = set()
        for job in jobs:
            required = {"dataset", "operation", "levels", "seeds"}
            optional = {"noise_type"}
            if stage == "discrepancy":
                required |= {"metrics"}
                optional |= {"n_expected_pairs", "chance_sample_seed"}
            _fields(job, required, optional, stage)
            if job["dataset"] not in datasets or job["operation"] not in {
                "shift",
                "drop",
                "swap",
            }:
                raise ValueError(f"{stage}: unknown dataset or operation")
            if job["operation"] == "swap" and job["dataset"] != "acnescu":
                raise ValueError("swap is supported for acnescu")
            job.setdefault("noise_type", "uniform")
            if job["noise_type"] not in {"uniform", "gauss"} or (
                job["operation"] != "shift" and job["noise_type"] != "uniform"
            ):
                raise ValueError(f"{stage}: unsupported noise_type")
            job["levels"] = grid(job["levels"], "levels")
            if any(
                g < 0 or (job["operation"] == "drop" and g >= 100) or (job["operation"] == "swap" and g > 100)
                for g in job["levels"]
            ):
                raise ValueError(f"{stage}: invalid perturbation levels")
            _seeds(job)
            if stage == "discrepancy":
                keys = {(job["dataset"], job["operation"], job["noise_type"], seed) for seed in job["seeds"]}
                if targets & keys:
                    raise ValueError(
                        "discrepancy: overlapping output files; combine levels and metrics into one configuration"
                    )
                targets.update(keys)
                selected = job["metrics"]
                if not isinstance(selected, list) or not selected or any(m not in metrics for m in selected):
                    raise ValueError("discrepancy: unknown or empty metrics")
                job["metrics"] = list(dict.fromkeys(selected))
                _integer(job.setdefault("n_expected_pairs", 10000), "n_expected_pairs", 1)
                _integer(job.setdefault("chance_sample_seed", 20260828), "chance_sample_seed")
        config[stage] = jobs
    for stage in ("theory", "empirical"):
        spec = config.get(stage) or {}
        if not spec:
            config[stage] = {}
            continue
        required = {"noise_types", "n_x"} if stage == "theory" else {"dataset", "noise_types", "seeds"}
        _fields(spec, required, {"beta_scan", "gamma_scan"}, stage)
        noises = spec["noise_types"]
        if not isinstance(noises, list) or not noises or set(noises) - {"uniform", "gauss"}:
            raise ValueError(f"{stage}: noise_types must contain uniform or gauss")
        spec["noise_types"] = list(dict.fromkeys(noises))
        if stage == "theory":
            _integer(spec["n_x"], "n_x", 2)
        else:
            if spec["dataset"] not in datasets:
                raise ValueError("empirical: unknown dataset")
            _seeds(spec)
        for key in ("beta_scan", "gamma_scan"):
            scan = spec.get(key) or {}
            if scan:
                _fields(scan, {"beta", "gamma_percent"}, set(), f"{stage}.{key}")
                scan["beta"] = grid(scan["beta"], "beta")
                scan["gamma_percent"] = grid(scan["gamma_percent"], "gamma_percent")
                if any(not 0 <= b <= 1 for b in scan["beta"]) or any(g <= 0 for g in scan["gamma_percent"]):
                    raise ValueError(f"{stage}.{key}: beta must be in [0, 1], gamma must be positive")
            spec[key] = scan
        config[stage] = spec
    return config


def number_token(value: float) -> str:
    """Represent numeric filename parameters without rounding."""
    if value == 0:
        return "0"
    token = format(Decimal(str(float(value))).normalize(), "f")
    return token if len(token) <= 64 else repr(float(value))


def beta_token(value: float) -> str:
    """Preserve historical 02/05/08 filenames and distinguish other beta values."""
    token = number_token(value)
    if "e" in token:
        return token
    return (token if "." in token else token + ".0").replace(".", "")


def variant_filename(dataset: str, operation: str, level: float, noise_type: str = "uniform") -> str:
    """Name a perturbed annotation CSV using the previous convention."""
    suffix = "_gauss" if operation == "shift" and noise_type == "gauss" else ""
    return f"{dataset}_{operation}{suffix}_{number_token(level)}.csv"


def read_input(path: Path, stage: str) -> pd.DataFrame:
    """Read an existing CSV or report the explicit preparation stage to run."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run prepare_all_data --config <yaml> --stages {stage}")
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=False)


def run_cli(generator, argv: list[str] | None = None) -> int:
    """Run one generator with the paper config or an explicit --config override."""
    parser = argparse.ArgumentParser(description=generator.__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML config (default: project configs/paper_plots_repro.yaml)",
    )
    args = parser.parse_args(argv)
    try:
        paths = generator(load_config(args.config))
        print(f"Wrote {len(paths)} CSV files")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0
