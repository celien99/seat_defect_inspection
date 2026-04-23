# Seat Defect Inspection

`seat_defect_inspection` is a standalone subproject for automotive seat defect inspection.

It currently provides:

- multi-camera image capture
- one PatchCore model per camera
- YOLO seat localization training
- bundled `media_inputs` and `mvsCamera` support for `mvs://...` sources

The MVS SDK path is already wired in code. Hardware validation against real cameras is intentionally left for later on-site testing.

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

## Recommended Workflow

1. Capture normal samples with `capture`.
2. Save or copy them into each camera `train_good_dir`.
   Note: `train_good_dir` stores raw camera images, but `train-patchcore` still replays the production preprocessing and mask-driven ROI preparation pipeline before fitting PatchCore.
3. Run `train-patchcore`.
4. Prepare a YOLO dataset and run `train-yolo`.
5. Configure the resulting weights and run `inspect` for live cameras, or `inspect-folder` for offline batch verification.

The project now standardizes on `yolo11m-seg.pt`. YOLO segmentation masks are consumed directly to crop the target ROI and build valid/ignore masks. The current ROI layer keeps only the lightweight crop, resize, and mask-cleanup steps instead of the older heavy local enhancement chain.

`train-patchcore` is already an offline workflow. As long as each camera `train_good_dir` points to a local image folder, PatchCore training does not require any real camera device.

`inspect-folder` is the new offline verification path. It reuses the same preprocess, YOLO, ROI, PatchCore, fusion, report, and debug-artifact chain as production `inspect`, but swaps live camera sources for local files.

Supported `inspect-folder` input layouts:

```text
offline_samples/
├── sample_001/
│   ├── cam_0.jpg
│   └── cam_1.jpg
└── sample_002/
    ├── cam_0.jpg
    └── cam_1.jpg
```

```text
offline_samples/
├── cam_0/
│   ├── sample_001.jpg
│   └── sample_002.jpg
└── cam_1/
    ├── sample_001.jpg
    └── sample_002.jpg
```

## Key Paths

- `data/seat_defect_inspection/<camera_id>/train/good`
- `models/seat_defect_inspection/<camera_id>_patchcore.npz`
- `outputs/seat_defect_inspection/capture`
- `outputs/seat_defect_inspection/debug`
- `<output_json_path sibling>/<output_json_path.stem>_history`
- `outputs/seat_defect_inspection/yolo_training`

## Code Layout

The project has been split by responsibility rather than kept in a few oversized files.

```text
src/seat_defect_inspection/
├── cli.py
├── cli_commands/
│   ├── common.py
│   ├── capture.py
│   ├── inspect.py
│   ├── inspect_folder.py
│   ├── train_patchcore.py
│   └── train_yolo.py
├── acquisition.py
├── config.py
├── debug_artifacts.py
├── fusion.py
├── reporting.py
├── runtime_config.py
├── runtime_config_parsers.py
├── runtime_config_camera_parsers.py
├── runtime_config_values.py
├── schemas.py
├── util.py
├── cvops/
│   ├── quality.py
│   ├── roi.py
│   ├── roi_geometry.py
│   └── debug_artifacts.py
├── preprocess/
│   └── engine.py
├── patchcore/
│   ├── engine.py
│   ├── features.py
│   ├── scoring.py
│   └── color_branch.py
├── service/
│   ├── __init__.py
│   ├── core.py
│   ├── capture.py
│   ├── inspection.py
│   ├── inspection_camera.py
│   ├── offline_inspection.py
│   └── training.py
└── yolo/
    ├── __init__.py
    ├── detection.py
    ├── training.py
    ├── dataset_validation.py
    └── labelme_to_yolo.py
```

## Module Responsibilities

- `cli.py`: CLI bootstrap only. It assembles subcommands and keeps the entry thin.
- `cli_commands/`: one command per file, including argument registration and routing to the matching business flow.
- `runtime_config.py`: config file entry and top-level validation.
- `runtime_config_parsers.py`: inspection-level and seat-model-level parsing.
- `runtime_config_camera_parsers.py`: camera sub-config parsing for preprocess, ROI, PatchCore, and YOLO detection.
- `runtime_config_values.py`: shared parsing helpers and path resolution.
- `service/__init__.py`: thin public routing layer.
- `service/core.py`: shared service context, caches, and `_CameraPipeline`.
- `service/capture.py`: capture workflow.
- `service/inspection.py`: multi-camera inspection orchestration and early-stop handling.
- `service/inspection_camera.py`: single-camera inspection details.
- `service/offline_inspection.py`: offline batch inspection from local image folders.
- `service/training.py`: PatchCore training workflow.
- `cvops/`: OpenCV middle layer for quality gating, ROI refinement, geometry helpers, texture preparation, and debug artifacts.
- `preprocess/engine.py`: image preprocessing chain before YOLO and ROI refinement.
- `patchcore/engine.py`: PatchCore lifecycle orchestration.
- `patchcore/features.py`: handcrafted and full-backbone feature extraction.
- `patchcore/scoring.py`: memory bank sampling, distance scoring, evidence analysis, and final decision rules.
- `patchcore/color_branch.py`: LAB-based color consistency branch.
- `yolo/detection.py`: YOLO inference and fallback box handling.
- `yolo/training.py`: YOLO training entry.
- `yolo/dataset_validation.py`: dataset preflight and label validation.

## Main Runtime Flow

`inspect` now follows a deliberately thin orchestration path:

1. `cli.py` assembles the parser, and `cli_commands/inspect.py` loads config and routes to `service.run_inspection`.
2. `service/__init__.py` creates `InspectionService`.
3. `service/inspection.py` loops over enabled cameras and handles fail-fast decisions.
4. `service/inspection_camera.py` runs one camera through `_CameraPipeline`, PatchCore, optional color branch, and debug artifact export.
5. `fusion.py` merges per-camera results.
6. `reporting.py` writes the final report.

For a more detailed architecture and flow breakdown, see [PROJECT_ARCHITECTURE_ZH.md](./PROJECT_ARCHITECTURE_ZH.md).
