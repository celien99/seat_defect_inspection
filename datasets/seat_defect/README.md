# YOLO Segmentation Dataset Scaffold

This directory follows the standard Ultralytics YOLO segmentation layout:

```text
datasets/seat_defect/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Class mapping:

- `0`: `seat`

The current labels are generated from the LabelMe JSON files under `images/train` and
`images/val`, and each `.txt` uses YOLO segmentation polygon format:

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

All coordinates are normalized into `[0, 1]`.
