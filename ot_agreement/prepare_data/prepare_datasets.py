#!/usr/bin/env python3
"""Convert raw annotations to CSV and prepare NumPy inputs for metrics."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from shutil import copyfileobj
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

COMMON_COLUMNS = [
    "annotation_id",
    "image_id",
    "file_name",
    "image_width",
    "image_height",
    "class_name",
    "class_id",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
]

TEXBIG_COLUMNS = [
    *COMMON_COLUMNS[:5],
    "annotator",
    *COMMON_COLUMNS[5:],
]

ACNE04_CLASS_MAP = {
    "fore": ("acne", 0),
    "acne": ("acne", 0),
}

ACNESCU_CLASS_MAP = {
    "atrophic_scar": 0,
    "closed_comedo": 1,
    "hypertrophic_scar": 2,
    "melasma": 3,
    "nevus": 4,
    "nodule": 5,
    "open_comedo": 6,
    "other": 7,
    "papule": 8,
    "pustule": 9,
}

BOX_COLUMNS = ["x_min", "y_min", "x_max", "y_max"]


def _transform_xml_dataset(
    xml_dir: Path,
    resolve_class: Callable[[str], tuple[str, int]],
) -> pd.DataFrame:
    """Read Pascal VOC boxes in filename order, preserving source coordinates."""
    rows = []
    for image_id, path in enumerate(sorted(xml_dir.glob("*.xml"))):
        root = ET.parse(path).getroot()
        image = {
            "image_id": image_id,
            "file_name": root.findtext("filename").strip(),
            "image_width": int(root.findtext("size/width")),
            "image_height": int(root.findtext("size/height")),
        }
        for obj in root.findall("object"):
            class_name, class_id = resolve_class(obj.findtext("name").strip())
            rows.append(
                {
                    "annotation_id": len(rows),
                    **image,
                    "class_name": class_name,
                    "class_id": class_id,
                    **{
                        column: int(obj.findtext(f"bndbox/{tag}"))
                        for column, tag in zip(BOX_COLUMNS, ("xmin", "ymin", "xmax", "ymax"), strict=True)
                    },
                }
            )
    return pd.DataFrame(rows, columns=COMMON_COLUMNS)


def transform_acne04_orig(xml_dir: Path | str) -> pd.DataFrame:
    """Read original Acne04 XML annotations with a single acne class."""
    return _transform_xml_dataset(Path(xml_dir), lambda name: ACNE04_CLASS_MAP[name])


def transform_acnescu(xml_dir: Path | str) -> pd.DataFrame:
    """Read ACNESCU XML annotations with the existing class encoding."""
    return _transform_xml_dataset(Path(xml_dir), lambda name: (name, ACNESCU_CLASS_MAP[name]))


def transform_acne04_v2(json_path: Path | str) -> pd.DataFrame:
    """Convert circle annotations to xyxy boxes without clipping."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    images = {image["id"]: image for image in data["images"]}
    rows = []
    for annotation in data["annotations"]:
        image = images[annotation["image_id"]]
        cx, cy = annotation["coordinates"]
        radius = annotation["radius"]
        class_name, class_id = ACNE04_CLASS_MAP[annotation["class_name"]]
        rows.append(
            {
                "annotation_id": annotation["id"],
                "image_id": annotation["image_id"],
                "file_name": image["file_name"],
                "image_width": image["width"],
                "image_height": image["height"],
                "class_name": class_name,
                "class_id": class_id,
                "x_min": cx - radius,
                "y_min": cy - radius,
                "x_max": cx + radius,
                "y_max": cy + radius,
            }
        )
    return pd.DataFrame(rows, columns=COMMON_COLUMNS).sort_values("annotation_id", kind="stable").reset_index(drop=True)


def transform_texbig_v1(json_path: Path | str) -> pd.DataFrame:
    """Convert COCO boxes to xyxy and retain TexBiG annotator metadata."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    images = {image["id"]: image for image in data["images"]}
    categories = {category["id"]: category["name"] for category in data["categories"]}
    rows = []
    for annotation in data["annotations"]:
        image = images[annotation["image_id"]]
        x, y, width, height = annotation["bbox"]
        rows.append(
            {
                "annotation_id": annotation["id"],
                "image_id": annotation["image_id"],
                "file_name": image["file_name"],
                "image_width": image["width"],
                "image_height": image["height"],
                "annotator": image["annotator"],
                "class_name": categories[annotation["category_id"]],
                "class_id": annotation["category_id"],
                "x_min": x,
                "y_min": y,
                "x_max": x + width,
                "y_max": y + height,
            }
        )
    return pd.DataFrame(rows, columns=TEXBIG_COLUMNS).sort_values("annotation_id", kind="stable").reset_index(drop=True)


def to_image_annotations(
    frame: pd.DataFrame,
    label_map: Mapping[object, int],
    normalize: bool = False,
) -> dict[object, tuple[np.ndarray, np.ndarray]]:
    """Group by file_name and return separate xyxy boxes and integer labels.

    Optional normalization uses each image's dimensions. Neither the frame nor
    its coordinates are modified. All compared sources must share label_map.
    """
    annotations = {}
    for image_id, group in frame.groupby("file_name", sort=False):
        boxes = group[BOX_COLUMNS].to_numpy(dtype=float, copy=True)
        if normalize:
            dimensions = group.iloc[0][["image_width", "image_height"]].to_numpy(dtype=float)
            if (dimensions <= 0).any():
                raise ValueError("Image dimensions must be positive")
            boxes /= np.tile(dimensions, 2)
        labels = group["class_id"].map(label_map).to_numpy(dtype=np.int64)
        annotations[image_id] = (boxes, labels)
    return annotations


TRANSFORMERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    "acne04_orig": transform_acne04_orig,
    "acne04_v2": transform_acne04_v2,
    "acnescu": transform_acnescu,
    "texbig_v1": transform_texbig_v1,
}


@contextmanager
def _staged_file(destination: Path) -> Iterator[Path]:
    staged = destination.with_name(f".annotations-{uuid4().hex}.tmp")
    staged.touch(exist_ok=False)
    try:
        yield staged
    finally:
        staged.unlink(missing_ok=True)


def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Write a DataFrame atomically to avoid leaving an incomplete CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _staged_file(output_path) as staged:
        df.to_csv(staged, index=False)
        staged.replace(output_path)


def unpack_raw_annotations(raw_dir: Path) -> None:
    """Extract missing raw files from the bundled archive, preserving existing files."""
    archive_path = raw_dir / "raw_annotations.zip"
    if not archive_path.is_file():
        return
    try:
        with ZipFile(archive_path) as archive:
            members = archive.infolist()
            root = raw_dir.resolve()
            for member in members:
                if not (raw_dir / member.filename).resolve().is_relative_to(root):
                    raise ValueError(f"Unsafe path in raw annotations archive: {member.filename}")
            extracted = 0
            for member in members:
                destination = raw_dir / member.filename
                if member.is_dir() or destination.exists() or destination.is_symlink():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with _staged_file(destination) as staged:
                    with archive.open(member) as source, staged.open("wb") as target:
                        copyfileobj(source, target)
                    try:
                        destination.hardlink_to(staged)
                    except FileExistsError:
                        continue
                    extracted += 1
    except BadZipFile as exc:
        raise ValueError(f"Invalid raw annotations archive: {archive_path}") from exc
    if extracted:
        tqdm.write(f"Extracted {extracted} raw annotation files from {archive_path}")


def generate_results(config: dict) -> list[Path]:
    """Convert the datasets selected in the YAML prepare section directly to CSV."""
    if config.get("prepare"):
        unpack_raw_annotations(config["data_dir"] / "raw")
    paths = []
    for dataset in tqdm(config.get("prepare", []), desc="Prepare datasets", unit="dataset", dynamic_ncols=True):
        source = config["datasets"][dataset]["source"]
        if not source.exists():
            raise FileNotFoundError(f"Missing raw annotations: {source}")
        frame = TRANSFORMERS[dataset](source)
        if frame.empty:
            raise ValueError(f"{dataset}: source contains no annotated objects")
        numeric = frame[[*BOX_COLUMNS, "image_width", "image_height", "class_id"]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or (frame[["image_width", "image_height"]] <= 0).any().any():
            raise ValueError(f"{dataset}: invalid annotation values or image dimensions")
        path = config["data_dir"] / "prepared" / f"{dataset}.csv"
        write_csv(frame, path)
        paths.append(path)
        tqdm.write(f"{dataset}: wrote {len(frame)} annotations to {path}")
    return paths


def main(argv: list[str] | None = None) -> int:
    """Prepare selected datasets from --config YAML."""
    from ot_agreement.prepare_data.config import run_cli

    return run_cli(generate_results, argv)


if __name__ == "__main__":
    raise SystemExit(main())
