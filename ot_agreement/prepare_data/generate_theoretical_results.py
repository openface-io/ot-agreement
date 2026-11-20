#!/usr/bin/env python3
"""Generate theoretical shift-noise probability distributions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import simpson
from scipy.special import erf
from tqdm.auto import tqdm

from ot_agreement.prepare_data.config import beta_token, number_token, run_cli
from ot_agreement.prepare_data.prepare_datasets import write_csv

DEFAULT_N_X = 50_000
EPSILON = 1e-6

DISTRIBUTIONS = ("uniform", "gauss")

ArrayFunction = Callable[[np.ndarray], np.ndarray]


def integrate_over_boundary_cdf(
    x_boundary: np.ndarray,
    y_low_bnd: np.ndarray,
    y_up_bnd: np.ndarray,
    pdf_x: ArrayFunction,
    cdf_y: ArrayFunction,
) -> float:
    """Integrate a 2D probability using an analytical inner-variable CDF."""
    x = np.asarray(x_boundary)
    y_low = np.asarray(y_low_bnd)
    y_upp = np.asarray(y_up_bnd)

    if x.ndim != 1 or len(x) < 2:
        raise ValueError("x_boundary must be a one-dimensional grid")
    if x.shape != y_low.shape or x.shape != y_upp.shape:
        raise ValueError("x and y boundary arrays must have identical shapes")

    inner_integrals = np.where(
        y_upp > y_low,
        cdf_y(y_upp) - cdf_y(y_low),
        0.0,
    )
    integrand_x = pdf_x(x) * inner_integrals
    return float(simpson(integrand_x, x=x))


def make_distribution(
    distr: str,
    sigma: float | None = None,
    a: float | None = None,
) -> tuple[ArrayFunction, ArrayFunction]:
    """Return PDF/CDF functions for half-normal or non-negative uniform noise."""
    if distr == "gauss":
        if sigma is None or sigma <= 0:
            raise ValueError("sigma must be positive for gauss distribution")

        def pdf(x: np.ndarray) -> np.ndarray:
            values = np.asarray(x)
            return (2 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(values**2) / (2 * sigma**2))

        def cdf(x: np.ndarray) -> np.ndarray:
            return erf(np.asarray(x) / (sigma * np.sqrt(2)))

    elif distr == "uniform":
        if a is None or a <= 0:
            raise ValueError("a must be positive for uniform distribution")

        def pdf(x: np.ndarray) -> np.ndarray:
            values = np.asarray(x)
            return np.where((values >= 0) & (values <= a), 1.0 / a, 0.0)

        def cdf(x: np.ndarray) -> np.ndarray:
            return np.clip(np.asarray(x) / a, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown distribution {distr!r}; choose from {DISTRIBUTIONS}")

    return pdf, cdf


@dataclass(frozen=True)
class BoundaryRegions:
    """Integration grids and upper boundaries for the three spatial regions."""

    x_unit: np.ndarray
    x_tail: np.ndarray
    y_region_1: np.ndarray
    y_region_2: np.ndarray
    y_region_3: np.ndarray


class BoundaryModel:
    """Reusable symbolic and numerical geometry for the notebook integrals."""

    def __init__(self, n_x: int = DEFAULT_N_X) -> None:
        """Build numerical grids and symbolic coefficients once."""
        if n_x < 2:
            raise ValueError("n_x must be at least 2")
        self.n_x = n_x
        self.x_unit = np.linspace(0, 1, n_x)
        self.x_tail = np.linspace(1, 50, 2 * n_x)
        self._coefficient_functions = self._build_coefficient_functions()

    @staticmethod
    def _build_coefficient_functions() -> tuple[Callable[..., np.ndarray], ...]:
        x, y, k = sp.symbols("x y k")
        expression = (
            (1 - x) * (1 - y) * (1 + x) * (1 + y)
            - (1 + x) * (1 + y) * (2 - (1 - x) * (1 - y))
            + (2 - (1 - x) * (1 - y)) ** 2
            - k * (2 - (1 - x) * (1 - y)) * (1 + x) * (1 + y)
        )
        expanded = sp.expand(sp.simplify(expression))
        coefficients = sp.Poly(expanded, y).all_coeffs()
        return tuple(sp.lambdify((x, k), coefficient, "numpy") for coefficient in coefficients)

    def boundaries(self, beta: float) -> BoundaryRegions:
        """Calculate the three integration boundaries for one beta."""
        if not 0 <= beta <= 1:
            raise ValueError("beta must be in [0, 1]")

        k_value = 1 - 2 * beta
        a_values, b_values, c_values = (
            np.asarray(function(self.x_unit, k_value)) for function in self._coefficient_functions
        )
        discriminant = b_values**2 - 4 * a_values * c_values

        y_region_1 = np.clip(
            (-b_values - np.sqrt(discriminant)) / (2 * a_values + EPSILON),
            0,
            1,
        )

        def no_intersection_boundary(values: np.ndarray) -> np.ndarray:
            return 1 / ((1 - beta) * (1 + values) + EPSILON) - 1

        y_region_2 = np.clip(
            no_intersection_boundary(self.x_unit),
            1,
            100,
        )
        y_region_3 = np.clip(
            no_intersection_boundary(self.x_tail),
            0,
            100,
        )
        return BoundaryRegions(
            x_unit=self.x_unit,
            x_tail=self.x_tail,
            y_region_1=y_region_1,
            y_region_2=y_region_2,
            y_region_3=y_region_3,
        )


def _probability(
    boundaries: BoundaryRegions,
    relative_noise: float,
    distribution: str,
) -> float:
    pdf_x, cdf_y = make_distribution(
        distr=distribution,
        sigma=relative_noise,
        a=relative_noise,
    )
    probability_1 = integrate_over_boundary_cdf(
        boundaries.x_unit,
        y_low_bnd=np.zeros_like(boundaries.y_region_1),
        y_up_bnd=boundaries.y_region_1,
        pdf_x=pdf_x,
        cdf_y=cdf_y,
    )
    probability_2 = integrate_over_boundary_cdf(
        boundaries.x_unit,
        y_low_bnd=np.ones_like(boundaries.y_region_2),
        y_up_bnd=boundaries.y_region_2,
        pdf_x=pdf_x,
        cdf_y=cdf_y,
    )
    probability_3 = integrate_over_boundary_cdf(
        boundaries.x_tail,
        y_low_bnd=np.zeros_like(boundaries.y_region_3),
        y_up_bnd=boundaries.y_region_3,
        pdf_x=pdf_x,
        cdf_y=cdf_y,
    )
    probability = probability_1 + probability_2 + probability_3
    if not np.isfinite(probability):
        raise FloatingPointError("Theoretical matching probability is non-finite")
    return float(np.clip(probability, 0.0, 1.0))


def generate_results(config: dict) -> list[Path]:
    """Compute the configured theoretical curves and write the original CSV format."""
    spec = config.get("theory")
    if not spec:
        return []
    model = BoundaryModel(spec["n_x"])
    beta_scan, gamma_scan = spec["beta_scan"], spec["gamma_scan"]
    # Iterate beta on the outside: only one potentially large integration boundary stays in RAM.
    beta_values = sorted(set(beta_scan.get("beta", [])) | set(gamma_scan.get("beta", [])))
    curves = {}
    for beta in tqdm(beta_values, desc="Theory: beta", unit="beta", dynamic_ncols=True):
        # With positive continuous noise, beta=0 allows only exact matches and beta=1 allows all matches.
        boundary = None if beta in (0.0, 1.0) else model.boundaries(beta)
        for noise in tqdm(
            spec["noise_types"], desc=f"beta={beta:g}: noise", unit="distribution", leave=False, dynamic_ncols=True
        ):
            gammas = set(beta_scan.get("gamma_percent", [])) if beta in beta_scan.get("beta", []) else set()
            if beta in gamma_scan.get("beta", []):
                gammas.update(gamma_scan.get("gamma_percent", []))
            for gamma in tqdm(
                sorted(gammas), desc=f"{noise}: gamma (%)", unit="point", leave=False, dynamic_ncols=True
            ):
                value = float(beta) if boundary is None else _probability(boundary, gamma / 100, noise)
                if beta in beta_scan.get("beta", []) and gamma in beta_scan.get("gamma_percent", []):
                    suffix = "uni" if noise == "uniform" else "gauss"
                    name = f"theor_beta_shift_{suffix}_{number_token(gamma)}.csv"
                    curves.setdefault(name, []).append({"beta": beta, "cdf": value})
                if beta in gamma_scan.get("beta", []) and gamma in gamma_scan.get("gamma_percent", []):
                    suffix = "" if noise == "uniform" else "_gauss"
                    name = f"theor_rel_shift{suffix}_{beta_token(beta)}.csv"
                    curves.setdefault(name, []).append({"rel": gamma / 100, "cdf": value})
    paths = []
    for name, records in tqdm(curves.items(), desc="Write theory CSV", unit="file", leave=False, dynamic_ncols=True):
        path = config["data_dir"] / "results" / "theor" / name
        write_csv(pd.DataFrame(records), path)
        paths.append(path)
    tqdm.write(f"theory: wrote {len(paths)} curves")
    return paths


def main(argv: list[str] | None = None) -> int:
    """Generate theoretical curves from --config YAML."""
    return run_cli(generate_results, argv)


if __name__ == "__main__":
    raise SystemExit(main())
