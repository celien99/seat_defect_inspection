# YOLO Dataset Scaffold

This directory follows the standard Ultralytics YOLO detection layout:

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

- `0`: `seat_main`
- `1`: `tooling`
- `2`: `worker_hand`
- `3`: `wire`
- `4`: `foreign_object`

The initial samples in this scaffold are synthetic images used only to verify that the local
training pipeline, dataset YAML, path resolution, and output directories work end-to-end.
They are not suitable for production model quality evaluation.
