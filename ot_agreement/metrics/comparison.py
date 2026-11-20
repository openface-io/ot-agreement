"""Compare annotation sources for one image using any scalar pair metric."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np


def _score_matrix(
    rows: Sequence[tuple[np.ndarray, np.ndarray]],
    columns: Sequence[tuple[np.ndarray, np.ndarray]],
    metric: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
) -> np.ndarray:
    scores = np.empty((len(rows), len(columns)), dtype=float)
    for row, first in enumerate(rows):
        for column, second in enumerate(columns):
            scores[row, column] = metric(*first, *second)
    return scores


def annotators_vs_annotators(
    annotations: Mapping[str, tuple[np.ndarray, np.ndarray]],
    metric: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
) -> tuple[list[str], np.ndarray]:
    """Return names and an ``[A, A]`` matrix for one image.

    Each source is a pair of ``[N, 4]`` xyxy boxes and ``[N]`` integer labels.
    Entry ``[i, j]`` is ``metric(*annotations[names[i]], *annotations[names[j]])``.
    Every ordered pair is evaluated, including the diagonal and empty arrays;
    no symmetry, self-agreement value, or score transformation is imposed.
    Bind metric options with ``functools.partial`` before calling. For example,
    ``partial(compute_f1, return_metric_only=True)`` returns scalar F1 scores.
    """
    names = list(annotations)
    sources = [annotations[name] for name in names]
    return names, _score_matrix(sources, sources, metric)


def model_vs_annotators(
    model_annotations: tuple[np.ndarray, np.ndarray],
    annotations: Mapping[str, tuple[np.ndarray, np.ndarray]],
    metric: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
) -> tuple[list[str], np.ndarray]:
    """Return names and ``[A]`` scores, with the model as the first argument.

    The model and every annotator supply a pair ``(boxes, labels)``, with
    shapes ``[N, 4]`` and ``[N]`` and non-negative integer class IDs.
    Names follow the mapping's insertion order. Empty arrays are passed to the
    metric unchanged, exactly as in :func:`annotators_vs_annotators`.
    """
    names = list(annotations)
    sources = [annotations[name] for name in names]
    return names, _score_matrix([model_annotations], sources, metric)[0]


__all__ = ["annotators_vs_annotators", "model_vs_annotators"]
