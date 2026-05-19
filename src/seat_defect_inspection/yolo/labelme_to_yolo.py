"""Convert LabelMe annotations into YOLO segmentation labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union


@dataclass(slots=True)
class ConversionSummary:
    split: str
    json_count: int
    txt_count: int
    object_count: int


def convert_labelme_split(
    image_dir: Union[str, Path],
    label_dir: Union[str, Path],
    *,
    class_name_to_id: Dict[str, int],
    allowed_shape_types: Optional[Set[str]] = None,
) -> ConversionSummary:
    image_root = Path(image_dir)
    label_root = Path(label_dir)
    if not image_root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_root}")

    label_root.mkdir(parents=True, exist_ok=True)
    normalized_shape_types = {
        item.strip().lower() for item in (allowed_shape_types or {"polygon", "linestrip"})
    }

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
    class_name_to_id: Dict[str, int],
    allowed_shape_types: Set[str],
) -> List[str]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    image_width = float(payload["imageWidth"])
    image_height = float(payload["imageHeight"])
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image size in {json_path}")

    rows: List[str] = []
    for shape in payload.get("shapes", []):
        label_name = str(shape.get("label", "")).strip()
        if not label_name:
            continue
        if label_name not in class_name_to_id:
            raise ValueError(f"Unsupported label `{label_name}` in {json_path}")

        shape_type = str(shape.get("shape_type", "polygon")).strip().lower()
        if shape_type not in allowed_shape_types:
            continue

        polygon = _shape_to_polygon(shape.get("points") or [], json_path=json_path)
        normalized_points: List[str] = []
        for x, y in polygon:
            normalized_x = min(max(float(x) / image_width, 0.0), 1.0)
            normalized_y = min(max(float(y) / image_height, 0.0), 1.0)
            normalized_points.append(f"{normalized_x:.6f}")
            normalized_points.append(f"{normalized_y:.6f}")

        rows.append(f"{class_name_to_id[label_name]} {' '.join(normalized_points)}")
    return rows


def _shape_to_polygon(points: List[List[float]], *, json_path: Path) -> List[Tuple[float, float]]:
    if len(points) < 3:
        raise ValueError(f"Shape has fewer than 3 points in {json_path}")

    polygon = [(float(point[0]), float(point[1])) for point in points]
    if polygon[0] == polygon[-1]:
        polygon = polygon[:-1]
    if len(polygon) < 3:
        raise ValueError(f"Shape has fewer than 3 unique points in {json_path}")
    return polygon
