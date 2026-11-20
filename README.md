# Quantifying Annotation Agreement in Object Detection with Optimal Transport

This paper studies an Optimal Transport metric for measuring agreement between
object detection annotations, accounting for differences in object presence,
bounding box location, and class labels.

Accepted to ACCV 2026. Read the [preprint on arXiv](https://arxiv.org/abs/2610.19912).

## Installation

Requires Python 3.13 and Poetry. From the repository root, create and activate
a virtual environment, then run:

```bash
poetry install
```

## Using the OT metric

Compare two sets of annotations for the same image. Boxes use
`[x_min, y_min, x_max, y_max]` coordinates; labels are integer class IDs.

```python
import numpy as np
from ot_agreement.metrics import compute_ot_cost

boxes_a = np.array([[10, 10, 50, 50], [60, 20, 90, 60]], dtype=float)
labels_a = np.array([0, 0])
boxes_b = np.array([[12, 10, 52, 50]], dtype=float)
labels_b = np.array([0])

cost = compute_ot_cost(
    boxes_a, labels_a, boxes_b, labels_b,
    lam=0.5, beta=1.0, return_metric_only=True,
)
print(f"OT cost: {cost:.4f}")
```

Lower cost indicates closer agreement; identical annotations have zero cost.
See [`ot_agreement/metrics/ot_metric.py`](ot_agreement/metrics/ot_metric.py) for details.

## Generating figures

Run both commands from a fresh checkout, in the order shown. The plotter
discovers result CSVs in the data directory, including files from earlier runs.

```bash
# Unpack raw annotations and generate the experiment data.
poetry run python -m ot_agreement.prepare_data.prepare_all_data

# Create the paper figures.
poetry run python -m ot_agreement.create_paper_figures
```

- Experiment settings and dataset paths are defined in [`configs/paper_plots_repro.yaml`](configs/paper_plots_repro.yaml).
- Missing raw files are automatically extracted from `ot_agreement/data/raw/raw_annotations.zip`. See the
[dataset README](ot_agreement/data/README.md) for data sources.
- Generated figures are saved as PDF and JPG files in `ot_agreement/data/paper_plots/`.
- The plotting command covers the score bounds, matching curves, annotator comparisons, discrepancy curves, and face
  example. Main-text Figure 5 requires a separate analysis.

## Citation

```bibtex
@inproceedings{prokhorov2026quantifying,
  title={Quantifying Annotation Agreement in Object Detection with Optimal Transport},
  author={Prokhorov, Kirill and Kalinin, Alexandr A},
  booktitle={Asian Conference on Computer Vision (ACCV)},
  year={2026}
}
```
