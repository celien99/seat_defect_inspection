# Seat Defect Inspection

The main inspection architecture is `seat_defect_core`. It owns the only inspect runtime. CLI, capture, training, and offline-folder workflows are engineering tools that feed images into the core flow.

For image-pipeline details, see [IMAGE_PIPELINE_DETAILS_ZH.md](./IMAGE_PIPELINE_DETAILS_ZH.md). For architecture details, see [PROJECT_ARCHITECTURE_ZH.md](./PROJECT_ARCHITECTURE_ZH.md).

## Quick Start

```bash
cd seat_defect_inspection
conda create -n seat-defect-inspection python=3.10 -y
conda activate seat-defect-inspection
pip install -e .
seat-defect-inspection --help
```

## Engineering Tool Commands

These commands belong to the engineering tool layer, not the main inspection architecture. External systems should call `seat_defect_core` directly.

```bash
seat-defect-inspection capture --config configs/seat_defect_inspection.mvs.json --part-id seat_000001
seat-defect-inspection train-patchcore --config configs/seat_defect_inspection.mvs.json
seat-defect-inspection train-yolo --config configs/seat_defect_inspection.mvs.json
seat-defect-inspection inspect --config configs/seat_defect_inspection.mvs.json --part-id seat_000001
seat-defect-inspection inspect-folder --config configs/seat_defect_inspection.mvs.json --input-dir offline_samples
```

## Python Runtime API

External projects should use `seat_defect_core`. The core runtime does not capture images; callers pass one image per configured camera.

```python
import cv2
from seat_defect_core import InspectionFrame, SeatDefectInspector

inspector = SeatDefectInspector("configs/seat_defect_inspection.mvs.json")
response = inspector.inspect(
    frames=[
        InspectionFrame(camera_id="cam_0", image=cv2.imread("cam_0.png")),
        InspectionFrame(camera_id="cam_1", image=cv2.imread("cam_1.png")),
    ],
    part_id="seat_000001",
)

print(response.status, response.decision_reason)
print(response.report_path)
print(response.archive_report_path)
```

## Engineering Tool Reference Workflow

1. Capture normal samples with `capture`.
2. Save them into each camera `train_good_dir`.
3. Run `train-patchcore`.
4. Prepare a YOLO segmentation dataset and run `train-yolo`.
5. Configure the resulting weights and run live `inspect` or offline `inspect-folder`.

`train-patchcore` is offline: it reads raw images from `train_good_dir` and replays the production YOLO, ROI, mask, and PatchCore input pipeline before fitting.

`inspect-folder` reuses the same production detection chain, but replaces live camera sources with local images.

## Offline Layouts

Single sample directory:

```text
offline_samples/
├── cam_0.jpg
└── cam_1.jpg
```

Sample directories:

```text
offline_samples/
├── sample_001/
│   ├── cam_0.jpg
│   └── cam_1.jpg
└── sample_002/
    ├── cam_0.jpg
    └── cam_1.jpg
```

Camera directories:

```text
offline_samples/
├── cam_0/
│   ├── sample_001.jpg
│   └── sample_002.jpg
└── cam_1/
    ├── sample_001.jpg
    └── sample_002.jpg
```

## Code Layout

```text
src/
├── seat_defect_core/        # inspect runtime source of truth
├── seat_defect_inspection/  # engineering tools: CLI, capture, training, offline workflows
├── media_inputs/            # tool-layer input abstraction
└── mvsCamera/               # tool-layer Hikvision MVS adapter
```

Main inspection behavior belongs in `seat_defect_core`: external-frame normalization, YOLO inference, ROI/masks, regions, PatchCore, color branch, fusion, debug artifacts, and inspection reports.

Engineering tool behavior belongs in `seat_defect_inspection`: CLI commands, config extensions, acquisition, capture manifest, offline-folder discovery, PatchCore training orchestration, YOLO training, and LabelMe conversion. Its inspect command captures images and then delegates to `seat_defect_core`.

Legacy runtime imports under `seat_defect_inspection` are intentionally removed. Import runtime APIs directly from `seat_defect_core`.
