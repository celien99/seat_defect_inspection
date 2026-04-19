# Seat Defect Inspection

`seat_defect_inspection` is a standalone subproject for automotive seat defect inspection.

It provides:

- multi-camera image capture
- one PatchCore model per camera
- YOLO seat localization training
- bundled `media_inputs` and `mvsCamera` support for `mvs://...` sources

The MVS SDK path is already wired in code. Hardware validation against real cameras is intentionally left for later on-site testing.

## Quick Start

```bash
cd seat_defect_inspection
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
seat-defect-inspection --help
```

## Main Commands

```bash
seat-defect-inspection capture --config configs/seat_defect_inspection.mvs.json --part-id seat_000001
seat-defect-inspection train-patchcore --config configs/seat_defect_inspection.mvs.json
seat-defect-inspection train-yolo --config configs/seat_defect_inspection.mvs.json
seat-defect-inspection inspect --config configs/seat_defect_inspection.mvs.json --part-id seat_000001
```

## Recommended Workflow

1. Capture normal samples with `capture`.
2. Save or copy them into each camera `train_good_dir`.
3. Run `train-patchcore`.
4. Prepare a YOLO dataset and run `train-yolo`.
5. Configure the resulting weights and run `inspect`.

## Key Paths

- `data/seat_defect_inspection/<camera_id>/train/good`
- `models/seat_defect_inspection/<camera_id>_patchcore.npz`
- `outputs/seat_defect_inspection/capture`
- `outputs/seat_defect_inspection/debug`
- `outputs/seat_defect_inspection/yolo_training`
