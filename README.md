# Seat Defect Inspection

`seat_defect_inspection` is the engineering CLI project for automotive seat defect inspection. The detection runtime now lives in `seat_defect_core`; the Python SDK facade lives in `seat_defect_sdk`.

For image-pipeline details, see [IMAGE_PIPELINE_DETAILS_ZH.md](./IMAGE_PIPELINE_DETAILS_ZH.md). For architecture details, see [PROJECT_ARCHITECTURE_ZH.md](./PROJECT_ARCHITECTURE_ZH.md).

## Quick Start

```bash
cd seat_defect_inspection
conda create -n seat-defect-inspection python=3.10 -y
conda activate seat-defect-inspection
pip install -e .
seat-defect-inspection --help
```

## Main Commands

```bash
seat-defect-inspection capture --config configs/seat_defect_inspection.mvs.json --part-id seat_000001
seat-defect-inspection train-patchcore --config configs/seat_defect_inspection.mvs.json
seat-defect-inspection train-yolo --config configs/seat_defect_inspection.mvs.json
seat-defect-inspection inspect --config configs/seat_defect_inspection.mvs.json --part-id seat_000001
seat-defect-inspection inspect-folder --config configs/seat_defect_inspection.mvs.json --input-dir offline_samples
```

## Python SDK

External projects should use `seat_defect_sdk`. The SDK does not capture images; callers pass one image per configured camera.

```python
import cv2
from seat_defect_sdk import CameraFrame, SeatDefectInspector

inspector = SeatDefectInspector("configs/seat_defect_inspection.mvs.json")
response = inspector.inspect(
    frames=[
        CameraFrame(camera_id="cam_0", image=cv2.imread("cam_0.png")),
        CameraFrame(camera_id="cam_1", image=cv2.imread("cam_1.png")),
    ],
    part_id="seat_000001",
)

print(response.status, response.decision_reason)
print(response.report_path)
print(response.archive_report_path)
```

## Recommended Workflow

1. Capture normal samples with `capture`.
2. Save them into each camera `train_good_dir`.
3. Run `train-patchcore`.
4. Prepare a YOLO segmentation dataset and run `train-yolo`.
5. Configure the resulting weights and run live `inspect` or offline `inspect-folder`.

`train-patchcore` is offline: it reads raw images from `train_good_dir` and replays the production preprocessing, YOLO, ROI, mask, and PatchCore input pipeline before fitting.

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
├── seat_defect_core/        # single runtime source of truth
├── seat_defect_sdk/         # external-image SDK facade
├── seat_defect_inspection/  # CLI, capture, training, offline workflows
├── media_inputs/            # image/video/camera source abstraction
└── mvsCamera/               # Hikvision MVS adapter
```

Runtime behavior belongs in `seat_defect_core`: preprocessing, YOLO inference, ROI/masks, PatchCore, color branch, fusion, debug artifacts, and inspection reports.

Engineering behavior belongs in `seat_defect_inspection`: CLI commands, config extensions, acquisition, capture manifest, offline-folder discovery, PatchCore training orchestration, YOLO training, and LabelMe conversion.

Legacy runtime imports under `seat_defect_inspection` are intentionally removed. Import runtime APIs directly from `seat_defect_core`.
