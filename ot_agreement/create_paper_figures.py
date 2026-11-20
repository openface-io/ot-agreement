"""Plot annotation agreement, matching probabilities, and score bounds from saved data."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from matplotlib import patches
from matplotlib import pyplot as plt

from ot_agreement.metrics.f1_metric import compute_f1
from ot_agreement.metrics.kalpha_metric import compute_krippendorff_alpha
from ot_agreement.metrics.miou_metric import compute_miou
from ot_agreement.metrics.ot_metric import compute_ot_cost

DATA_DIR = Path(__file__).resolve().parent / "data"
PREPARED_DIR = DATA_DIR / "prepared"
RESULTS_DIR = DATA_DIR / "results"
PAPER_PLOTS_DIR = DATA_DIR / "paper_plots"


def _save_figure(figure, name):
    """Save a paper figure in the supported publication formats."""
    PAPER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for file_format in ("pdf", "jpg"):
        figure.savefig(
            PAPER_PLOTS_DIR / f"{name}.{file_format}",
            format=file_format,
            bbox_inches="tight",
        )


def get_bboxes(df, file_name):
    """Return [N, 5] array (xmin, ymin, xmax, ymax, label) for one image (raw pixel coords)."""
    sub = df[df["file_name"] == file_name]
    coords = sub[["x_min", "y_min", "x_max", "y_max"]].values.astype(float)
    # labels are in category_id column
    labels = sub[["class_id"]].values.astype(float)
    return np.hstack([coords, labels])


def get_distances_to_bounds(N_vs_M_empirical, ot_scores, N_vs_M_theory, lower_bound, upper_bound):
    """Get distances to metric bounds.

    For each empirical point, find the closest N/M value in the theoretical grid
    and compute the vertical distance to upper and lower bounds.
    """
    distances = []

    for nm, score in zip(N_vs_M_empirical, ot_scores, strict=False):
        # Find closest index in theoretical N/M array
        idx = np.argmin(np.abs(N_vs_M_theory - nm))

        lb = lower_bound[idx]
        ub = upper_bound[idx]

        distances.append(
            {
                "N_vs_M": nm,
                "ot_score": score,
                "lower_bound": lb,
                "upper_bound": ub,
                "dist_to_lower": score - lb,  # positive = above lower bound
                "dist_to_upper": ub - score,  # positive = below upper bound
                "within_bounds": (score >= lb) and (score <= ub),
                "band_width": ub - lb,
                "rel_dist_to_upper": (ub - score) / (ub - lb),
                "rel_dist_to_lower": (score - lb) / (ub - lb),
            }
        )

    return pd.DataFrame(distances)


def plot_score_bounds(show=True):
    """Plot ot score bounds."""
    BETA = [0.9, 0.4]
    LAM = [0.7, 0.2]
    colors = ["tab:orange", "#3772FF"]

    N_vs_M_less_1 = np.arange(0.001, 1.0, 0.01)
    N_vs_M_bigger_1 = np.arange(1, 100, 0.01)

    ax_text_fontsize = 14
    figure, ax = plt.subplots(figsize=(9, 6))
    for ii, (lam, beta) in enumerate(zip(LAM, BETA, strict=False)):
        # N < M
        lower_bound_less = 1 - (beta * (1 - N_vs_M_less_1) + lam * N_vs_M_less_1)
        upper_bound_less = 1 - beta * (1 - N_vs_M_less_1)
        # N > M
        lower_bound_bigger = 1 - (beta * (1 - 1 / N_vs_M_bigger_1) + lam * 1 / N_vs_M_bigger_1)
        upper_bound_bigger = 1 - beta * (1 - 1 / N_vs_M_bigger_1)

        N_vs_M = np.hstack((N_vs_M_less_1, N_vs_M_bigger_1))
        lower_bound = np.hstack((lower_bound_less, lower_bound_bigger))
        upper_bound = np.hstack((upper_bound_less, upper_bound_bigger))

        log2_N_vs_M = np.log2(N_vs_M)
        ax.plot(
            log2_N_vs_M,
            lower_bound,
            label=f"$\\lambda=${lam}; $\\beta=${beta} lower",
            ls="--",
            color=colors[ii],
            lw=2,
        )
        ax.plot(
            log2_N_vs_M,
            upper_bound,
            label=f"$\\lambda=${lam}; $\\beta=${beta} upper",
            color=colors[ii],
            lw=2,
        )
        ax.fill_between(log2_N_vs_M, lower_bound, upper_bound, color=colors[ii], alpha=0.15)

    ax.axhline(0.1, color=colors[0], linestyle="-.", linewidth=1.5)
    ax.text(
        -2.9,
        0.06,
        "$Score = 1 - \\beta = 0.1$",
    )

    ax.axhline(0.6, color=colors[1], linestyle="-.", linewidth=1.5)
    ax.text(
        -2.9,
        0.56,
        "$Score = 1 - \\beta = 0.6$",
    )

    arrow_style = {"arrowstyle": "->", "color": "gray", "linewidth": 1.5}
    arrow_y = 0.01
    ax.annotate(
        "",
        xy=(-0.9, arrow_y),
        xytext=(0, arrow_y),
        xycoords=("data", "axes fraction"),
        arrowprops=arrow_style,
    )
    ax.annotate(
        "",
        xy=(0.9, arrow_y),
        xytext=(0, arrow_y),
        xycoords=("data", "axes fraction"),
        arrowprops=arrow_style,
    )
    ax.text(
        -0.9,
        0.02,
        "$N<M$",
        ha="center",
        va="bottom",
    )
    ax.text(
        0.9,
        0.02,
        "$N>M$",
        ha="center",
        va="bottom",
    )
    ax.text(
        0.0,
        0.02,
        "$M=N$",
        ha="center",
        va="bottom",
    )

    ax.set_xlabel(
        "Ratio of number of boxes in two annotations, $\\log_2(\\frac{N}{M})$",
        fontsize=ax_text_fontsize,
    )
    ax.set_ylabel("OT Score", fontsize=ax_text_fontsize)
    ax.set_xlim([-3, 3])
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=10)
    ax.set_title("OT Score Bounds", fontsize=ax_text_fontsize)
    ax.tick_params(axis="both", labelsize=ax_text_fontsize)
    ax.grid()
    _save_figure(figure, "score_bounds")
    if show:
        plt.show()


def _decode_beta(token):
    """Decode the historical compact token without rounding arbitrary beta."""
    if token == "10":
        value = 1.0
    elif re.fullmatch(r"0\d+", token):
        value = float("0." + token[1:])
    else:
        value = float(token)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid beta token {token!r}")
    return value


def _matching_files(results_dir):
    """Read curve identity from the filenames and seed folders that generators write."""
    found = []
    for path in sorted((results_dir / "theor").glob("*.csv")):
        beta = re.fullmatch(r"theor_beta_shift_(uni|gauss)_(.+)\.csv", path.name)
        gamma = re.fullmatch(r"theor_rel_shift(_gauss)?_(.+)\.csv", path.name)
        if beta:
            fixed = float(beta[2])
            if not np.isfinite(fixed) or fixed <= 0:
                raise ValueError(f"{path}: invalid gamma")
            scan, noise = "beta", "uniform" if beta[1] == "uni" else "gauss"
        elif gamma:
            scan, noise, fixed = (
                "gamma",
                "gauss" if gamma[1] else "uniform",
                _decode_beta(gamma[2]),
            )
        else:
            continue
        found.append(
            {
                "path": path,
                "scan": scan,
                "noise": noise,
                "fixed": fixed,
                "dataset": None,
                "seed": None,
            }
        )
    for path in sorted((results_dir / "empirical").glob("seed_*/*.csv")):
        match = re.fullmatch(r"(beta|rel)_(.+)_shift(_gauss)?_(.+)\.csv", path.name)
        seed = re.fullmatch(r"seed_(\d+)", path.parent.name)
        if not match or not seed:
            continue
        scan = "beta" if match[1] == "beta" else "gamma"
        fixed = float(match[4]) if scan == "beta" else _decode_beta(match[4])
        if not np.isfinite(fixed) or (scan == "beta" and fixed <= 0):
            raise ValueError(f"{path}: invalid fixed parameter")
        found.append(
            {
                "path": path,
                "scan": scan,
                "noise": "gauss" if match[3] else "uniform",
                "fixed": fixed,
                "dataset": match[2],
                "seed": int(seed[1]),
            }
        )
    return found


def _curve_table(item):
    frame = pd.read_csv(item["path"], float_precision="round_trip")
    axis = "beta" if item["scan"] == "beta" else "rel"
    value = "cdf" if item["dataset"] is None else "matched_ratio"
    missing = {axis, value} - set(frame)
    if missing:
        raise ValueError(f"{item['path']}: missing columns {sorted(missing)}")
    frame = frame[[axis, value]].apply(pd.to_numeric, errors="raise").rename(columns={axis: "x", value: "value"})
    if frame.empty or not np.isfinite(frame.to_numpy()).all() or frame.x.duplicated().any():
        raise ValueError(f"{item['path']}: empty, non-finite or duplicate curve points")
    if item["scan"] == "gamma" and item["dataset"] is not None:
        frame["x"] /= 100  # Empirical rel is percent, theoretical rel is a fraction.
    if (frame.x < 0).any() or (item["scan"] == "beta" and (frame.x > 1).any()):
        raise ValueError(f"{item['path']}: invalid horizontal grid")
    frame["seed"] = item["seed"]
    return frame.sort_values("x")


def _style(index):
    """Extend the original curve palette when additional series are available."""
    colors = ("#3772FF", "tab:orange", "tab:green", "#A61C3C")
    return (
        colors[index] if index < len(colors) else plt.get_cmap("tab20")((index - len(colors)) % 20),
        ("o", "D", "x", "^", "v", "s", "+")[index % 7],
        ("-", "--", "-.", ":")[index % 4],
    )


def _number(value):
    """Keep distinct generated parameter values distinct in legends."""
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _report_samples(description, frame):
    counts = sorted(frame.n_seeds.unique().tolist())
    seeds = sorted({seed for sample in frame.seeds for seed in sample})
    print(f"{description}: seeds = {seeds}; n_seeds per point = {counts}")
    if (frame.n_seeds < len(seeds)).any():
        print(f"{description}: partial seed coverage; each point uses its available seeds")


def _plot_matching(scan, show, results_dir, datasets, seeds):
    x_scale = 100 if scan == "gamma" else 1
    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    files = [f for f in _matching_files(results_dir) if f["scan"] == scan]
    theory = [f for f in files if f["dataset"] is None]
    empirical = [f for f in files if f["dataset"] is not None and (seeds is None or f["seed"] in seeds)]
    names = sorted({f["dataset"] for f in empirical})
    if datasets is not None:
        names = [datasets] if isinstance(datasets, str) else list(dict.fromkeys(datasets))
    elif not names and theory:
        names = ["theory"]
    if not names:
        raise FileNotFoundError(f"No {scan} matching curves found in {results_dir}")
    plots = {}
    for dataset in names:
        items = theory + [f for f in empirical if f["dataset"] == dataset]
        if not items:
            raise FileNotFoundError(f"No {scan} matching curves found for dataset {dataset}")
        noises = [noise for noise in ("uniform", "gauss") if any(f["noise"] == noise for f in items)]
        values = sorted({f["fixed"] for f in items})
        figure, axes = plt.subplots(1, len(noises), figsize=(8 * len(noises), 6), squeeze=False)
        rows = []
        for axis, noise in zip(axes[0], noises, strict=True):
            for index, fixed in enumerate(values):
                selected = [f for f in items if f["noise"] == noise and f["fixed"] == fixed]
                if not selected:
                    continue
                color, marker, line = _style(index)
                label = rf"$\gamma={_number(fixed)}$" if scan == "beta" else rf"$\beta={_number(fixed)}$"
                theoretical = [f for f in selected if f["dataset"] is None]
                observed = [f for f in selected if f["dataset"] is not None]
                if len(theoretical) > 1:
                    raise ValueError(
                        f"Duplicate theoretical curves for {noise}, {fixed}: {[f['path'] for f in theoretical]}"
                    )
                if theoretical:
                    curve = _curve_table(theoretical[0])
                    axis.plot(
                        curve.x * x_scale,
                        curve.value,
                        color=color,
                        linestyle=line,
                        linewidth=3,
                        label=label,
                    )
                else:
                    print(f"{dataset}/{noise}/{label}: no theory; plotting empirical data")
                if observed:
                    data = pd.concat([_curve_table(f) for f in observed], ignore_index=True)
                    if data.duplicated(["seed", "x"]).any():
                        raise ValueError(f"Duplicate empirical seed/grid points for {dataset}/{noise}/{fixed}")
                    summary = (
                        data.groupby("x", as_index=False)
                        .agg(
                            mean=("value", "mean"),
                            std=("value", "std"),
                            n_seeds=("seed", "nunique"),
                            seeds=("seed", lambda values: tuple(sorted(set(values)))),
                        )
                        .fillna({"std": 0})
                    )
                    _report_samples(f"{dataset}/{noise}/{label}", summary)
                    summary = summary.assign(dataset=dataset, noise_type=noise, fixed=fixed)
                    rows.append(summary)
                    axis.errorbar(
                        summary.x * x_scale,
                        summary["mean"],
                        yerr=summary["std"],
                        fmt=marker,
                        linestyle="none",
                        color=color,
                        capsize=3,
                        elinewidth=1.2,
                        label="_nolegend_" if theoretical else label,
                    )
                else:
                    print(f"{dataset}/{noise}/{label}: no empirical data; plotting theory")
            axis.set_xlabel(
                r"$\beta$" if scan == "beta" else r"Noise level $\gamma$ (relative to box width/height), %",
                fontsize=16,
            )
            axis.set_ylabel(r"$F_c(\beta)=P(cost<\beta)$" if scan == "beta" else r"$P(cost<\beta|\beta)$", fontsize=16)
            axis.set_title(
                r"u, $\vartheta \sim Uniform(0, a)$"
                if noise == "uniform"
                else r"u, $\vartheta \sim Half-Normal(0, \sigma^2)$",
                fontsize=16,
            )
            axis.grid()
            axis.tick_params(labelsize=16)
            axis.legend(
                fontsize=16,
                title=r"Noise level $\gamma$ (%):" if scan == "beta" else None,
                title_fontsize=13,
            )
        figure.tight_layout()
        name = ("cdf" if scan == "beta" else "prob_vs_noise") + f"_{dataset}"
        _save_figure(figure, name)
        plots[dataset] = (
            figure,
            axes[0],
            pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
        )
    if show:
        plt.show()
    return plots


def plot_cdf(show=True, *, results_dir=None, datasets=None, seeds=None):
    """Discover all available matching curves versus beta and plot each dataset."""
    return _plot_matching("beta", show, results_dir, datasets, seeds)


def plot_prob_vs_noise(show=True, *, results_dir=None, datasets=None, seeds=None):
    """Discover all available matching curves versus gamma and plot each dataset."""
    return _plot_matching("gamma", show, results_dir, datasets, seeds)


SERIES_COLUMNS = [
    "metric_kind",
    "lambda",
    "beta",
    "label_dist",
    "iou_threshold",
    "transform",
    "n_expected_pairs",
    "chance_sample_seed",
]


def _discrepancy_files(results_dir):
    found = []
    for path in sorted(Path(results_dir).glob("*/seed_*/discrep_*.csv")):
        match = re.fullmatch(r"discrep_(shift|drop|swap)(_gauss)?\.csv", path.name)
        seed = re.fullmatch(r"seed_(\d+)", path.parent.name)
        if match and seed:
            found.append(
                {
                    "path": path,
                    "dataset": path.parent.parent.name,
                    "seed": int(seed[1]),
                    "perturbation": match[1],
                    "noise_type": "gauss" if match[2] else "uniform",
                }
            )
    return found


def _read_discrepancy(item):
    path = item["path"]
    frame = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
    required = {
        "dataset",
        "seed",
        "perturbation",
        "noise_type",
        "level",
        "metric_name",
        "scope",
        "image_id",
        "value",
        *SERIES_COLUMNS,
    }
    if required - set(frame):
        print(f"Skipping {path}: discrepancy CSV lacks metric parameters; regenerate with --stages discrepancy")
        return None
    if frame.empty:
        print(f"Skipping {path}: empty discrepancy CSV")
        return None
    for key in ("dataset", "seed", "perturbation", "noise_type"):
        if not frame[key].eq(item[key]).all():
            raise ValueError(f"{path}: {key} metadata does not match its path")
    for column in ("level", "value"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"{path}: non-finite {column}")
    if (frame.level < 0).any() or not frame.scope.isin(["image", "dataset"]).all():
        raise ValueError(f"{path}: invalid levels or scopes")
    for column in SERIES_COLUMNS:
        if column not in {"metric_kind", "transform"}:
            frame[column] = frame[column].map(lambda value: float(value) if value != "" else np.nan)
    allowed = {"ot", "ot_chance_corrected", "f1", "kalpha", "miou"}
    if not frame.metric_kind.isin(allowed).all() or frame.metric_name.eq("").any():
        raise ValueError(f"{path}: invalid metric kind or name")
    for kind, group in frame.groupby("metric_kind"):
        applicable = []
        if kind in {"ot", "ot_chance_corrected"}:
            applicable += ["lambda", "beta", "label_dist"]
        if kind in {"f1", "kalpha"}:
            applicable += ["iou_threshold"]
        if kind == "kalpha" and not group["transform"].isin(["identity", "unit"]).all():
            raise ValueError(f"{path}: invalid alpha transformation")
        if kind == "ot_chance_corrected":
            applicable += ["n_expected_pairs", "chance_sample_seed"]
        if not np.isfinite(group[applicable].to_numpy(dtype=float)).all():
            raise ValueError(f"{path}: missing or non-finite parameters for {kind}")
        for column in applicable:
            values = group[column]
            if (values < 0).any() or (column in {"lambda", "beta", "iou_threshold"} and (values > 1).any()):
                raise ValueError(f"{path}: invalid {column}")
            if column in {"n_expected_pairs", "chance_sample_seed"} and (
                (values % 1 != 0).any() or (column == "n_expected_pairs" and (values < 1).any())
            ):
                raise ValueError(f"{path}: invalid {column}")
        if kind != "ot_chance_corrected" and not group.scope.eq("image").all():
            raise ValueError(f"{path}: {kind} requires image-level scores")
    for _, group in frame[frame.metric_kind.eq("ot_chance_corrected")].groupby(
        SERIES_COLUMNS + ["metric_name", "level"], dropna=False
    ):
        if len(group[group.scope.eq("dataset")]) != 1:
            raise ValueError(f"{path}: chance-corrected OT requires one dataset-level row per metric/level")
    if frame.loc[frame.scope.eq("dataset"), "image_id"].ne("").any():
        raise ValueError(f"{path}: dataset-level rows must have empty image_id")
    if frame.loc[frame.scope.eq("image"), "image_id"].eq("").any():
        raise ValueError(f"{path}: missing image_id")
    return frame


def _discrepancy_summary(frame):
    # Chance correction uses its dataset row, never an average of its raw image OT scores.
    chosen = frame[frame.scope.eq(np.where(frame.metric_kind.eq("ot_chance_corrected"), "dataset", "image"))].copy()
    if chosen.empty:
        raise ValueError("No primary metric values; chance-corrected OT requires dataset-level rows")
    keys = SERIES_COLUMNS + ["seed", "level", "image_id"]
    counts = chosen.groupby(keys, dropna=False).value.nunique()
    if (counts > 1).any():
        raise ValueError("Conflicting duplicate metric/seed/level/image scores")
    chosen = chosen.drop_duplicates(keys)
    per_seed = chosen.groupby(SERIES_COLUMNS + ["seed", "level"], dropna=False, as_index=False).value.mean()
    rows = []
    for values, group in per_seed.groupby(SERIES_COLUMNS + ["level"], dropna=False, sort=True):
        sample = group.value.to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(SERIES_COLUMNS + ["level"], values, strict=True)),
                "mean": float(sample.mean()),
                "std": float(sample.std()),
                "min": float(sample.min()),
                "max": float(sample.max()),
                "n_seeds": len(sample),
                "seeds": tuple(sorted(group.seed.astype(int))),
            }
        )
    return pd.DataFrame(rows)


def _metric_label(row, *, show_sampling=False):
    """Use compact metric notation and include settings needed to distinguish variants."""
    kind = row["metric_kind"]
    if kind in {"ot", "ot_chance_corrected"}:
        label = r"$\alpha_{OT}$" if kind == "ot_chance_corrected" else "OT"
        label += rf": $\beta={float(row['beta'])}, \lambda={float(row['lambda'])}$"
        if row["label_dist"] != 1:
            label += f", label_dist={_number(row['label_dist'])}"
        if kind == "ot_chance_corrected" and show_sampling:
            label += f", pairs={_number(row['n_expected_pairs'])}, sample seed={_number(row['chance_sample_seed'])}"
        return label
    if kind == "miou":
        return "mIoU"
    label = "F1" if kind == "f1" else (r"$(\alpha+1)/2$" if row["transform"] == "unit" else r"$\alpha$")
    if row["iou_threshold"] != 0.5:
        label += f", IoU threshold={_number(row['iou_threshold'])}"
    return label


def _metric_order(row):
    """Keep OT configurations first, then F1, alpha, and mIoU."""
    kind = row["metric_kind"]
    if kind in {"ot", "ot_chance_corrected"}:
        return (0, row["beta"], -row["lambda"])
    return ({"f1": 1, "kalpha": 2, "miou": 3}[kind], 0, 0)


def plot_discrepancy_agreement(
    dataset="acne04_orig",
    operation="drop",
    seeds=None,
    results_dir=None,
    show=True,
    *,
    noise_type="uniform",
):
    """Plot all saved metric configurations for one discovered dataset/operation/noise group."""
    root = Path(results_dir) if results_dir is not None else RESULTS_DIR / "discrepancy"
    files = [
        f
        for f in _discrepancy_files(root)
        if f["dataset"] == dataset and f["perturbation"] == operation and f["noise_type"] == noise_type
    ]
    if seeds is not None:
        requested = set(seeds)
        missing = requested - {f["seed"] for f in files}
        if missing:
            print(f"{dataset}/{operation}/{noise_type}: missing seeds {sorted(missing)}; using available seeds")
        files = [f for f in files if f["seed"] in requested]
    frames = [_read_discrepancy(f) for f in files]
    frames = [f for f in frames if f is not None]
    if not frames:
        raise FileNotFoundError(
            f"No usable discrepancy results for {dataset}/{operation}/{noise_type} in {root}; "
            "regenerate with --stages discrepancy"
        )
    frame = pd.concat(frames, ignore_index=True)
    summary = _discrepancy_summary(frame)
    groups = list(summary.groupby(SERIES_COLUMNS, dropna=False, sort=True))
    groups.sort(key=lambda item: _metric_order(item[1].iloc[0]))
    figure, axis = plt.subplots(figsize=(9, 3.5) if operation == "shift" else (6, 5))
    for index, (_, group) in enumerate(groups):
        group = group.sort_values("level")
        row = group.iloc[0]
        kind = row["metric_kind"]
        related = summary[
            summary.metric_kind.eq(kind)
            & summary["lambda"].eq(row["lambda"])
            & summary.beta.eq(row["beta"])
            & summary.label_dist.eq(row["label_dist"])
        ]
        show_sampling = len(related[["n_expected_pairs", "chance_sample_seed"]].drop_duplicates()) > 1
        label = _metric_label(row, show_sampling=show_sampling)
        color = {"f1": "tab:orange", "kalpha": "tab:green", "miou": "#A61C3C"}.get(kind, "#3772FF")
        marker = ("o", "D", "^", "<", "x", ".", "v", "s", "+")[index % 9]
        line = {"f1": "--", "kalpha": "-.", "miou": ":"}.get(kind, "-")
        if index >= 9:
            line = ("-", "--", "-.", ":")[(index // 9) % 4]
        _report_samples(f"{dataset}/{operation}/{noise_type}/{label}", group)
        axis.errorbar(
            group.level,
            group["mean"],
            yerr=np.maximum(0, [group["mean"] - group["min"], group["max"] - group["mean"]]),
            fmt="none",
            ecolor=color,
            elinewidth=1,
            capsize=4,
        )
        axis.fill_between(
            group.level,
            group["mean"] - group["std"],
            group["mean"] + group["std"],
            color=color,
            alpha=0.25,
        )
        axis.plot(
            group.level,
            group["mean"],
            marker=marker,
            linestyle=line,
            color=color,
            markerfacecolor="none",
            markeredgewidth=2,
            label=label,
        )
    levels = np.sort(summary.level.unique())
    axis.set_xticks(levels, [f"{_number(v)}%" for v in levels])
    axis.set_xlabel(
        {
            "drop": "Dropped boxes (%)",
            "shift": r"Shift noise level $\gamma$ (%)",
            "swap": "Swapped labels fraction (%)",
        }[operation],
        fontsize=14,
    )
    axis.set_ylabel("Agreement score", fontsize=14)
    axis.legend(fontsize=10)
    axis.grid(alpha=0.4)
    margin = max(5.0, float(levels.max() - levels.min()) * 0.03)
    axis.set_xlim(levels.min() - margin, levels.max() + margin)
    figure.tight_layout()
    _save_figure(figure, f"discrepancy_{dataset}_{operation}_{noise_type}")
    if show:
        plt.show()
    return figure, axis, summary


def plot_ot_vs_ratio_acne04(show=True):
    """Plot ot vs ratio for acne04."""
    # Load both annotation CSVs
    df_first = pd.read_csv(PREPARED_DIR / "acne04_orig.csv")
    df_second = pd.read_csv(PREPARED_DIR / "acne04_v2.csv")

    # Find common images (by file_name)
    images_first = set(df_first["file_name"].unique())
    images_second = set(df_second["file_name"].unique())
    common_images = sorted(images_first & images_second)

    print(f"First: {len(images_first)} images, Second: {len(images_second)} images, Common: {len(common_images)}")

    BETA = 0.9
    LAM = 0.7
    LABEL_DIST = 1.0

    results = []
    # ===================
    for file_name in tqdm.tqdm(common_images):
        bboxes_a = get_bboxes(df_first, file_name)  # [N, 5]
        bboxes_b = get_bboxes(df_second, file_name)  # [M, 5]

        # --- OT metric
        _, ot_cost, unmatched_ratio, spatial_cost, label_cost, unmatched_cost, _ = compute_ot_cost(
            bboxes_a[:, :4].copy(),
            bboxes_a[:, 4].astype(int),
            bboxes_b[:, :4].copy(),
            bboxes_b[:, 4].astype(int),
            lam=LAM,
            beta=BETA,
            label_dist=LABEL_DIST,
        )
        ot_score = 1 - ot_cost

        results.append(
            {
                "file_name": file_name,
                "n_orig": len(bboxes_a),
                "n_v2": len(bboxes_b),
                "ot_score": round(ot_score, 4),
                "spatial_cost": round(spatial_cost, 4),
                "label_cost": round(label_cost, 4),
                "unmatched_cost": round(unmatched_cost, 4),
            }
        )

    df_results = pd.DataFrame(results)
    # ===================

    # ====== Logarithmic scale
    N_vs_M_less_1 = np.arange(0.001, 1.0, 0.01)
    N_vs_M_bigger_1 = np.arange(1, 100, 0.01)

    # N < M
    lower_bound_less = 1 - (BETA * (1 - N_vs_M_less_1) + LAM * N_vs_M_less_1)
    upper_bound_less = 1 - BETA * (1 - N_vs_M_less_1)
    # N > M
    lower_bound_bigger = 1 - (BETA * (1 - 1 / N_vs_M_bigger_1) + LAM * 1 / N_vs_M_bigger_1)
    upper_bound_bigger = 1 - BETA * (1 - 1 / N_vs_M_bigger_1)

    N_vs_M = np.hstack((N_vs_M_less_1, N_vs_M_bigger_1))
    lower_bound = np.hstack((lower_bound_less, lower_bound_bigger))
    upper_bound = np.hstack((upper_bound_less, upper_bound_bigger))

    N_vs_M_exp = df_results["n_v2"] / df_results["n_orig"]

    ax_text_fontsize = 14
    score_figure = plt.figure()

    plt.scatter(
        np.log2(N_vs_M_exp),
        df_results["ot_score"],
        label="$OT_{Score}$",
        alpha=1,
        lw=1,
        color="#3772FF",
        facecolors="none",
        s=16,
    )

    plt.plot(
        np.log2(N_vs_M),
        lower_bound,
        label=f"$\\lambda=${LAM}; $\\beta=${BETA} lower",
        ls="--",
        color="tab:orange",
        lw=2,
    )
    plt.plot(
        np.log2(N_vs_M),
        upper_bound,
        label=f"$\\lambda=${LAM}; $\\beta=${BETA} upper",
        color="tab:orange",
        lw=2,
    )
    plt.xlabel(
        "Ratio of number of boxes in two annotations, $\\log_2(\\frac{N}{M})$",
        fontsize=ax_text_fontsize,
    )
    plt.ylabel("OT Score", fontsize=ax_text_fontsize)
    plt.xlim([-4.5, 4.5])
    plt.grid()
    plt.legend(fontsize=10)
    plt.title("OT score, Acne04 dataset, two annotators", fontsize=ax_text_fontsize)

    plt.tick_params(axis="both", labelsize=ax_text_fontsize)
    _save_figure(score_figure, "acne04_ot_vs_ratio")
    # =============
    # Distribution between upper and lower bounds
    # Compute for each class
    dist_result = get_distances_to_bounds(N_vs_M_exp, df_results["ot_score"], N_vs_M, lower_bound, upper_bound)

    # Quick summary
    print(f"  Within bounds:     {dist_result['within_bounds'].sum()}/{len(dist_result)}")
    print(f"  Mean dist to upper: {dist_result['dist_to_upper'].mean():.4f}")
    print(f"  Mean dist to lower: {dist_result['dist_to_lower'].mean():.4f}")

    ax_text_fontsize = 14

    distance_figure = plt.figure(figsize=(8, 6))

    sns.histplot(
        y=dist_result["rel_dist_to_lower"],
        bins=100,
        color="#3772FF",
        kde=False,
        edgecolor="none",
        alpha=0.2,
        stat="density",
    )
    sns.kdeplot(
        y=dist_result["rel_dist_to_lower"],
        color="#3772FF",
        linewidth=2,
        alpha=1.0,
    )
    plt.hlines(y=1, xmin=0, xmax=50, lw=10, color="tab:orange")
    plt.hlines(y=0, xmin=0, xmax=50, lw=10, color="tab:orange", ls="--")

    plt.annotate(
        "",
        xy=(4, 1),
        xytext=(4, 0),
        arrowprops={"arrowstyle": "<->", "color": "gray", "lw": 2},
    )
    plt.text(
        x=4.1,
        y=0.20,
        s="$\\Delta=\\text{upper bound} - \\text{lower bound}$",
        fontsize=ax_text_fontsize,
        rotation=90,
    )

    plt.grid()
    plt.text(x=2, y=0.95, s="Upper bound", rotation=0, fontsize=ax_text_fontsize)
    plt.text(x=2, y=0.02, s="Lower bound", rotation=0, fontsize=ax_text_fontsize)
    plt.ylabel(
        "Relative distance from lower bound, $\\frac{OT_{score}-low. bound}{\\Delta}$",
        fontsize=ax_text_fontsize,
    )
    plt.xlabel("Density", fontsize=ax_text_fontsize)
    plt.tick_params(axis="both", labelsize=ax_text_fontsize)
    plt.xlim(0, 6)
    plt.ylim(0, 1.0)
    _save_figure(distance_figure, "acne04_ot_distance_distribution")
    if show:
        plt.show()


def _select_texbig_pair(frame, class_name):
    """Choose the two annotators with most objects of a class, then their common images."""
    selected = frame[frame.class_name.eq(class_name)]
    counts = (
        selected.groupby("annotator")
        .size()
        .rename("objects")
        .reset_index()
        .sort_values(["objects", "annotator"], ascending=[False, True])
    )
    if len(counts) < 2:
        raise ValueError(f"TexBiG class {class_name!r} requires at least two annotators")
    annotator_a, annotator_b = counts.annotator.iloc[:2]
    common_images = set(selected.loc[selected.annotator.eq(annotator_a), "file_name"]) & set(
        selected.loc[selected.annotator.eq(annotator_b), "file_name"]
    )
    if not common_images:
        raise ValueError(f"TexBiG class {class_name!r}: {annotator_a} and {annotator_b} have no common images")
    selected = selected[selected.annotator.isin([annotator_a, annotator_b]) & selected.file_name.isin(common_images)]
    return selected, annotator_a, annotator_b


def plot_ot_vs_ratio_texbig(show=True):
    """Plot ot vs ratio for texbig."""
    df_orig = pd.read_csv(PREPARED_DIR / "texbig_v1.csv")

    classes_list = ["paragraph", "image", "caption"]
    df_results_list = {}

    for most_common_class in classes_list:
        df, annotator_a, annotator_b = _select_texbig_pair(df_orig, most_common_class)

        image_list = df["file_name"].unique()

        print(f"{most_common_class}: {annotator_a} vs {annotator_b}; " f"common images: {len(image_list)}")

        BETA = 0.9
        LAM = 0.7
        LABEL_DIST = 1.0

        results = []

        for file_name in tqdm.tqdm(image_list):
            bboxes_a = get_bboxes(df[df["annotator"] == annotator_a], file_name)
            bboxes_b = get_bboxes(df[df["annotator"] == annotator_b], file_name)

            # Compare geometry for the selected class using the OT dustbin.
            _, ot_cost, unmatched_ratio, spatial_cost, label_cost, unmatched_cost, _ = compute_ot_cost(
                bboxes_a[:, :4].copy(),
                bboxes_a[:, 4].astype(int),
                bboxes_b[:, :4].copy(),
                bboxes_b[:, 4].astype(int),
                lam=LAM,
                beta=BETA,
                label_dist=LABEL_DIST,
            )
            ot_score = 1 - ot_cost

            results.append(
                {
                    "file_name": file_name,
                    "annotator_a": annotator_a,
                    "annotator_b": annotator_b,
                    "n_orig": len(bboxes_a),
                    "n_v2": len(bboxes_b),
                    "ot_score": round(ot_score, 4),
                    "spatial_cost": round(spatial_cost, 4),
                    "label_cost": round(label_cost, 4),
                    "unmatched_cost": round(unmatched_cost, 4),
                }
            )

        df_results = pd.DataFrame(results)
        df_results_list[most_common_class] = df_results

    # ===== OT score bounds plot
    N_vs_M_less_1 = np.arange(0.001, 1.0, 0.001)
    N_vs_M_bigger_1 = np.arange(1, 100, 0.001)

    # N < M
    lower_bound_less = lambda x: 1 - (BETA * (1 - x) + LAM * x)
    upper_bound_less = lambda x: 1 - BETA * (1 - x)
    # N > M
    lower_bound_bigger = lambda x: 1 - (BETA * (1 - 1 / x) + LAM * 1 / x)
    upper_bound_bigger = lambda x: 1 - BETA * (1 - 1 / x)

    N_vs_M = np.hstack((N_vs_M_less_1, N_vs_M_bigger_1))
    lower_bound = np.hstack((lower_bound_less(N_vs_M_less_1), lower_bound_bigger(N_vs_M_bigger_1)))
    upper_bound = np.hstack((upper_bound_less(N_vs_M_less_1), upper_bound_bigger(N_vs_M_bigger_1)))

    df_results_par, df_results_img, df_results_cap = df_results_list.values()

    N_vs_M_par = df_results_par["n_v2"] / df_results_par["n_orig"]
    N_vs_M_img = df_results_img["n_v2"] / df_results_img["n_orig"]
    N_vs_M_cap = df_results_cap["n_v2"] / df_results_cap["n_orig"]

    ax_text_fontsize = 14
    score_figure = plt.figure()

    plt.scatter(
        np.log2(N_vs_M_par),
        df_results_par["ot_score"],
        label='$OT_{Score}$ cls "paragraph"',
        alpha=0.7,
        lw=2,
        color="#3772FF",
        facecolors="none",
    )
    plt.scatter(
        np.log2(N_vs_M_img),
        df_results_img["ot_score"],
        label='$OT_{Score}$ cls "image"',
        alpha=0.7,
        lw=1.5,
        marker="D",
        color="#A61C3C",
        facecolors="none",
    )
    plt.scatter(
        np.log2(N_vs_M_cap),
        df_results_cap["ot_score"],
        label='$OT_{Score}$ cls "caption"',
        alpha=1,
        lw=1,
        marker="x",
        color="tab:green",
    )

    plt.plot(
        np.log2(N_vs_M),
        lower_bound,
        label=f"$\\lambda=${LAM}; $\\beta=${BETA} lower",
        ls="--",
        color="tab:orange",
        lw=2,
    )
    plt.plot(
        np.log2(N_vs_M),
        upper_bound,
        label=f"$\\lambda=${LAM}; $\\beta=${BETA} upper",
        color="tab:orange",
        lw=2,
    )
    plt.xlabel(
        "Ratio of number of boxes in two annotations, $\\log_2(\\frac{N}{M})$",
        fontsize=ax_text_fontsize,
    )
    plt.ylabel("OT Score", fontsize=ax_text_fontsize)
    plt.xlim([-4.9, 4.9])
    plt.grid()
    plt.legend(fontsize=10)
    plt.title("OT score, TexBig dataset, two annotators", fontsize=ax_text_fontsize)

    plt.tick_params(axis="both", labelsize=ax_text_fontsize)

    _save_figure(score_figure, "texbig_ot_vs_ratio")

    # ==============
    par_result = get_distances_to_bounds(N_vs_M_par, df_results_par["ot_score"], N_vs_M, lower_bound, upper_bound)
    img_result = get_distances_to_bounds(N_vs_M_img, df_results_img["ot_score"], N_vs_M, lower_bound, upper_bound)
    cap_result = get_distances_to_bounds(N_vs_M_cap, df_results_cap["ot_score"], N_vs_M, lower_bound, upper_bound)

    # Quick summary
    print(f"  Within bounds:     {par_result['within_bounds'].sum()}/{len(par_result)}")
    print(f"  Within bounds:     {img_result['within_bounds'].sum()}/{len(img_result)}")
    print(f"  Within bounds:     {cap_result['within_bounds'].sum()}/{len(cap_result)}")

    ax_text_fontsize = 14
    distance_figure = plt.figure(figsize=(8, 6))
    # ========
    sns.histplot(
        y=par_result["rel_dist_to_lower"],
        bins=100,
        color="#3772FF",
        alpha=0.2,
        edgecolor="none",
        stat="density",
    )
    sns.kdeplot(
        y=par_result["rel_dist_to_lower"],
        color="#3772FF",
        linewidth=2,
        alpha=1.0,
        label='class "paragraph"',
    )

    sns.histplot(
        y=img_result["rel_dist_to_lower"],
        bins=100,
        color="#A61C3C",
        alpha=0.2,
        edgecolor="none",
        stat="density",
    )
    sns.kdeplot(
        y=img_result["rel_dist_to_lower"],
        color="#A61C3C",
        linewidth=2,
        alpha=1.0,
        label='class "image"',
    )

    sns.histplot(
        y=cap_result["rel_dist_to_lower"],
        bins=100,
        color="tab:green",
        alpha=0.2,
        edgecolor="none",
        stat="density",
    )
    sns.kdeplot(
        y=cap_result["rel_dist_to_lower"],
        color="tab:green",
        linewidth=2,
        alpha=1.0,
        label='class "caption"',
    )
    # ========

    plt.hlines(y=1, xmin=0, xmax=30, lw=10, color="tab:orange")
    plt.hlines(y=0, xmin=0, xmax=30, lw=10, ls="--", color="tab:orange")

    plt.annotate(
        "",
        xy=(11, 1),
        xytext=(11, 0),
        arrowprops={"arrowstyle": "<->", "color": "gray", "lw": 2},
    )
    plt.text(
        x=10.3,
        y=0.2,
        s="$\\Delta=\\text{upper bound} - \\text{lower bound}$",
        fontsize=ax_text_fontsize,
        rotation=90,
    )

    plt.grid()
    plt.text(x=14.2, y=0.93, s="Upper bound", rotation=0, fontsize=ax_text_fontsize, ha="right")
    plt.text(x=14.2, y=0.02, s="Lower bound", rotation=0, fontsize=ax_text_fontsize, ha="right")
    plt.ylabel(
        "Relative distance from lower bound, $\\frac{OT_{score}-low. bound}{\\Delta}$",
        fontsize=ax_text_fontsize,
    )
    plt.xlabel("Density", fontsize=ax_text_fontsize)
    plt.tick_params(axis="both", labelsize=ax_text_fontsize)
    plt.xlim(0.0, 14.5)
    plt.ylim(0.0, 1)
    plt.legend(fontsize=14)
    _save_figure(distance_figure, "texbig_ot_distance_distribution")
    if show:
        plt.show()
    return df_results_list


def plot_example_with_face(show=True):
    """Plot example with face."""
    data_dir = DATA_DIR
    file_name = "levle1_633.jpg"
    lam = 0.8
    beta = 0.6
    iou_threshold = 0.5

    df_orig = pd.read_csv(data_dir / "prepared" / "acne04_orig.csv")
    df_noisy = pd.read_csv(data_dir / "perturbed" / "acne04_orig" / "seed_69" / "acne04_orig_shift_50.csv")

    orig_rows = df_orig[df_orig["file_name"] == file_name]
    orig = get_bboxes(df_orig, file_name)
    noisy = get_bboxes(df_noisy, file_name)
    corrupted = noisy.copy()
    corrupted[[1, 3, 5], :4] -= 300

    image_row = orig_rows.iloc[0]
    image_width = int(image_row["image_width"])
    image_height = int(image_row["image_height"])
    image = plt.imread(data_dir / "raw" / "levle1_633_anonymized.jpg")
    num_classes = int(max(orig[:, 4].max(), noisy[:, 4].max()) + 1)

    def compute_metrics(boxes):
        _, ot_cost, *_ = compute_ot_cost(
            orig[:, :4],
            orig[:, 4].astype(int),
            boxes[:, :4],
            boxes[:, 4].astype(int),
            lam=lam,
            beta=beta,
            label_dist=1.0,
        )
        _, _, _, f1 = compute_f1(
            orig[:, :4],
            orig[:, 4].astype(int),
            boxes[:, :4],
            boxes[:, 4].astype(int),
            iou_threshold=iou_threshold,
        )
        alpha = compute_krippendorff_alpha(
            orig[:, :4],
            orig[:, 4].astype(int),
            boxes[:, :4],
            boxes[:, 4].astype(int),
            iou_threshold=iou_threshold,
        )
        miou, _ = compute_miou(
            orig[:, :4],
            orig[:, 4].astype(int),
            boxes[:, :4],
            boxes[:, 4].astype(int),
            img_shape=(image_height, image_width),
            num_classes=num_classes,
        )
        return {
            "ot_score": 1 - ot_cost,
            "f1": f1,
            "alpha": alpha,
            "miou": miou,
        }

    original_metrics = compute_metrics(noisy)
    corrupted_metrics = compute_metrics(corrupted)

    colors = ["#A61C3C", "#3772FF"]
    class_name = orig_rows.iloc[0]["class_name"]

    def plot_annotations(boxes, metrics):
        fig, ax = plt.subplots(figsize=(6, 8))
        ax.imshow(image, extent=(0, image_width, image_height, 0))

        for annotation, color, label in (
            (orig, colors[0], "Acne04: original annotation"),
            (boxes, colors[1], "Acne04: noisy annotation"),
        ):
            for index, box in enumerate(annotation):
                ax.add_patch(
                    patches.Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        edgecolor=color,
                        facecolor="none",
                        lw=1.5,
                        label=label if index == 0 else "_nolegend_",
                    )
                )
                ax.text(
                    box[0],
                    box[1],
                    str(class_name),
                    fontsize=7,
                    color="white",
                    verticalalignment="top",
                    bbox={
                        "facecolor": color,
                        "edgecolor": "none",
                        "pad": 0.2,
                        "alpha": 0.7,
                    },
                    zorder=3,
                )

        ax.set_xlim(0, image_width)
        ax.set_ylim(image_height, 0)
        ax.set_title(
            f"OT score={metrics['ot_score']:.2f} "
            f"($\\lambda={lam}$, $\\beta={beta}$) | "
            f"F1={metrics['f1']:.3f}, "
            f"$\\alpha$={metrics['alpha']:.2f} "
            f"($IoU^{{thr}}={iou_threshold}$) | "
            f"mIoU={metrics['miou']:.2f}"
        )
        ax.axis("off")
        ax.legend(loc="upper right", framealpha=0.8)
        fig.tight_layout()
        return fig

    original_figure = plot_annotations(noisy, original_metrics)
    corrupted_figure = plot_annotations(corrupted, corrupted_metrics)
    _save_figure(original_figure, "example_with_face_original")
    _save_figure(corrupted_figure, "example_with_face_corrupted")
    if show:
        plt.show()
    return (original_figure, corrupted_figure), {
        "original": original_metrics,
        "corrupted": corrupted_metrics,
    }


def main(argv=None):
    """Build selected plot groups, discovering saved matching and agreement series without YAML."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=("bounds", "matching", "comparison", "discrepancy", "example"),
        default=("bounds", "matching", "comparison", "discrepancy", "example"),
    )
    args = parser.parse_args(argv)
    # The unchanged analytical/comparison/example functions use these module paths.
    previous = (DATA_DIR, PREPARED_DIR, RESULTS_DIR, PAPER_PLOTS_DIR)
    paths = (
        args.data_dir.resolve() if args.data_dir else DATA_DIR,
        args.data_dir.resolve() / "prepared" if args.data_dir else PREPARED_DIR,
        args.data_dir.resolve() / "results" if args.data_dir else RESULTS_DIR,
        args.output_dir.resolve()
        if args.output_dir
        else (args.data_dir.resolve() / "paper_plots" if args.data_dir else PAPER_PLOTS_DIR),
    )
    globals().update(
        zip(
            ("DATA_DIR", "PREPARED_DIR", "RESULTS_DIR", "PAPER_PLOTS_DIR"),
            paths,
            strict=True,
        )
    )
    try:
        if "bounds" in args.figures:
            plot_score_bounds(show=False)
            plt.close("all")
        if "matching" in args.figures:
            for build in (plot_cdf, plot_prob_vs_noise):
                try:
                    build(show=False)
                except FileNotFoundError as exc:
                    print(f"Skipping matching curves: {exc}")
                finally:
                    plt.close("all")
        if "discrepancy" in args.figures:
            groups = sorted(
                {
                    (f["dataset"], f["perturbation"], f["noise_type"])
                    for f in _discrepancy_files(RESULTS_DIR / "discrepancy")
                }
            )
            if not groups:
                print(f"Skipping discrepancy plots: no results in {RESULTS_DIR / 'discrepancy'}")
            for dataset, operation, noise in groups:
                try:
                    plot_discrepancy_agreement(dataset, operation, show=False, noise_type=noise)
                except FileNotFoundError as exc:
                    print(f"Skipping discrepancy plots: {exc}")
                finally:
                    plt.close("all")
        if "comparison" in args.figures:
            plot_ot_vs_ratio_acne04(show=False)
            plot_ot_vs_ratio_texbig(show=False)
            plt.close("all")
        if "example" in args.figures:
            plot_example_with_face(show=False)
            plt.close("all")
        print(f"Figures saved to {PAPER_PLOTS_DIR}")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    finally:
        globals().update(
            zip(
                ("DATA_DIR", "PREPARED_DIR", "RESULTS_DIR", "PAPER_PLOTS_DIR"),
                previous,
                strict=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
