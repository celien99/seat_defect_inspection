from __future__ import annotations

import json
from pathlib import Path

from seat_defect_inspection.labelme_to_yolo import convert_labelme_split


def test_convert_labelme_split_writes_segmentation_polygon_labels(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True)

    (image_dir / "sample.json").write_text(
        json.dumps(
            {
                "imageWidth": 100,
                "imageHeight": 50,
                "shapes": [
                    {
                        "label": "seat",
                        "shape_type": "linestrip",
                        "points": [
                            [10, 5],
                            [80, 5],
                            [90, 40],
                            [20, 45],
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = convert_labelme_split(
        image_dir,
        label_dir,
        class_name_to_id={"seat": 0},
    )

    assert summary.json_count == 1
    assert summary.txt_count == 1
    assert summary.object_count == 1
    assert (label_dir / "sample.txt").read_text(encoding="utf-8") == (
        "0 0.100000 0.100000 0.800000 0.100000 0.900000 0.800000 0.200000 0.900000\n"
    )
