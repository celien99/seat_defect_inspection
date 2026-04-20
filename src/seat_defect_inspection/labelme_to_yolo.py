"""Convert LabelMe annotations into YOLO detection labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}


@dataclass(slots=True)
class ConversionSummary:
    split: str
    json_count: int
    txt_count: int
    object_count: int


def convert_labelme_split(
    image_dir: str | Path,
    label_dir: str | Path,
    *,
    class_name_to_id: dict[str, int],
    allowed_shape_types: set[str] | None = None,
) -> ConversionSummary:
    image_root = Path(image_dir)
    label_root = Path(label_dir)
    if not image_root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_root}")

    label_root.mkdir(parents=True, exist_ok=True)
    normalized_shape_types = {item.strip().lower() for item in (allowed_shape_types or {"rectangle", "linestrip"})}

    json_paths = sorted(image_root.glob("*.json")) + sorted(image_root.glob("*.JSON"))
    object_count = 0
    txt_count = 0
    for json_path in json_paths:
        lines = _convert_one_labelme_file(
            json_path,
            class_name_to_id=class_name_to_id,
            allowed_shape_types=normalized_shape_types,
        )
        txt_path = label_root / f"{json_path.stem}.txt"
        txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        txt_count += 1
        object_count += len(lines)

    return ConversionSummary(
        split=image_root.name,
        json_count=len(json_paths),
        txt_count=txt_count,
        object_count=object_count,
    )


def _convert_one_labelme_file(
    json_path: Path,
    *,
    class_name_to_id: dict[str, int],
    allowed_shape_types: set[str],
) -> list[str]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    image_width = float(payload["imageWidth"])
    image_height = float(payload["imageHeight"])
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image size in {json_path}")

    rows: list[str] = []
    for shape in payload.get("shapes", []):
        label_name = str(shape.get("label", "")).strip()
        if not label_name:
            continue
        if label_name not in class_name_to_id:
            raise ValueError(f"Unsupported label `{label_name}` in {json_path}")

        shape_type = str(shape.get("shape_type", "rectangle")).strip().lower()
        if shape_type not in allowed_shape_types:
            continue

        x1, y1, x2, y2 = _shape_to_box(shape.get("points") or [], json_path=json_path)
        x1 = min(max(x1, 0.0), image_width)
        y1 = min(max(y1, 0.0), image_height)
        x2 = min(max(x2, 0.0), image_width)
        y2 = min(max(y2, 0.0), image_height)
        if x2 <= x1 or y2 <= y1:
            continue

        x_center = ((x1 + x2) / 2.0) / image_width
        y_center = ((y1 + y2) / 2.0) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        rows.append(
            f"{class_name_to_id[label_name]} "
            f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
    return rows


def _shape_to_box(points: list[list[float]], *, json_path: Path) -> tuple[float, float, float, float]:
    if len(points) < 2:
        raise ValueError(f"Shape has fewer than 2 points in {json_path}")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)
