import numpy as np

from ot_agreement.metrics.annotations import validate_annotations


def compute_miou(
    boxes_a: np.ndarray,
    labels_a: np.ndarray,
    boxes_b: np.ndarray,
    labels_b: np.ndarray,
    img_shape: tuple[float, float],
    num_classes: int,
    *,
    return_metric_only: bool = False,
) -> float | tuple[float, list[float]]:
    """Compute mean and per-class IoU from continuous rectangle areas.

    Each source supplies xyxy boxes of shape ``[N, 4]`` and integer class IDs
    of shape ``[N]``, with labels in ``[0, num_classes)``.
    Average over the union of class labels in the inputs, before clipping boxes
    to the image. Absent classes do not reduce the mean. The per-class list still
    contains ``num_classes`` entries. Two empty inputs have mean IoU 0.
    With ``return_metric_only=True``, return only mean IoU as a float;
    otherwise return ``(mean_iou, per_class_iou)``.
    """
    num_classes = _validate_num_classes(num_classes)
    image_shape = _validate_image_shape(img_shape)
    boxes_a, labels_a = validate_annotations(boxes_a, labels_a, "annotations_a")
    boxes_b, labels_b = validate_annotations(boxes_b, labels_b, "annotations_b")
    presented_classes = np.union1d(labels_a, labels_b)
    boxes_1, labels_1 = _prepare_annotations(
        boxes_a,
        labels_a,
        "annotations_a",
        image_shape,
        num_classes,
    )
    boxes_2, labels_2 = _prepare_annotations(
        boxes_b,
        labels_b,
        "annotations_b",
        image_shape,
        num_classes,
    )

    per_class_iou = []
    for class_id in range(num_classes):
        class_boxes_1 = boxes_1[labels_1 == class_id]
        class_boxes_2 = boxes_2[labels_2 == class_id]
        if len(class_boxes_1) == 0 or len(class_boxes_2) == 0:
            per_class_iou.append(0.0)
            continue

        area_1 = _rectangle_union_area(class_boxes_1)
        area_2 = _rectangle_union_area(class_boxes_2)
        union_area = _rectangle_union_area(np.concatenate([class_boxes_1, class_boxes_2], axis=0))
        if union_area == 0:
            per_class_iou.append(0.0)
            continue

        intersection_area = area_1 + area_2 - union_area
        intersection_area = min(
            max(intersection_area, 0.0),
            area_1,
            area_2,
        )
        per_class_iou.append(intersection_area / union_area)

    mean_iou = float(np.sum(per_class_iou) / len(presented_classes)) if len(presented_classes) else 0.0
    if return_metric_only:
        return mean_iou
    return mean_iou, per_class_iou


def _validate_image_shape(img_shape):
    try:
        image_shape = np.asarray(img_shape[:2], dtype=np.float64)
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("img_shape must contain height and width") from exc

    if image_shape.shape != (2,):
        raise ValueError("img_shape must contain height and width")
    if not np.isfinite(image_shape).all() or (image_shape <= 0).any():
        raise ValueError("image height and width must be finite and positive")
    return float(image_shape[0]), float(image_shape[1])


def _validate_num_classes(num_classes):
    if isinstance(num_classes, (bool, np.bool_)) or not isinstance(num_classes, (int, np.integer)) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    return int(num_classes)


def _prepare_annotations(
    boxes: np.ndarray,
    labels: np.ndarray,
    name: str,
    image_shape: tuple[float, float],
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    if (labels >= num_classes).any():
        raise ValueError(f"{name} labels must be integers in [0, {num_classes})")

    height, width = image_shape
    boxes = boxes.copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, height)

    valid_boxes = (boxes[:, 0] < boxes[:, 2]) & (boxes[:, 1] < boxes[:, 3])
    return boxes[valid_boxes], labels[valid_boxes].astype(np.intp)


def _rectangle_union_area(rectangles):
    """Return the continuous union area of axis-aligned rectangles."""
    if len(rectangles) == 0:
        return 0.0

    x_edges = np.unique(rectangles[:, [0, 2]])
    total_area = 0.0

    for x_left, x_right in zip(x_edges[:-1], x_edges[1:], strict=False):
        active = rectangles[(rectangles[:, 0] < x_right) & (rectangles[:, 2] > x_left)]
        if len(active) == 0:
            continue

        intervals = active[:, [1, 3]]
        intervals = intervals[np.argsort(intervals[:, 0], kind="stable")]
        current_start, current_end = intervals[0]
        covered_y = 0.0

        for start, end in intervals[1:]:
            if start > current_end:
                covered_y += current_end - current_start
                current_start, current_end = start, end
            else:
                current_end = max(current_end, end)

        covered_y += current_end - current_start
        total_area += (x_right - x_left) * covered_y

    return float(total_area)


__all__ = ["compute_miou"]
