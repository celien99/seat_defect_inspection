# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode
pip install -e .

# Run the CLI
python -m seat_defect_inspection --help

# Run tests (pytest)
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_core_api.py -v

# Run tests matching a pattern
python -m pytest tests/ -v -k "PatchCore"
```

## Architecture

The project has a strict two-layer architecture:

- **`seat_defect_core`** — The single source of truth for the inspection runtime. It accepts external images, runs YOLO detection, ROI refinement, quality gating, PatchCore texture analysis, color branch, region-based inspection, multi-camera fusion, debug artifacts, and reporting. It never captures images, never trains models, and never traverses folders.
- **`seat_defect_inspection`** — Engineering tool layer. Provides CLI commands (`capture`, `train-patchcore`, `train-yolo`, `inspect`, `inspect-folder`, `benchmark`), camera acquisition, offline folder discovery, and training orchestration. Its `inspect` command captures images then delegates to `seat_defect_core` for the actual inspection.

`media_inputs/` and `mvsCamera/` are tool-layer dependencies for image acquisition (MVS camera SDK bindings). They do not belong to core.

**Python 3.8.5 constraint**: `seat_defect_core` must remain compatible with Python 3.8.5 for LabVIEW integration on production machines. Do not use language features or stdlib APIs introduced after 3.8. No `match`/`case`, no `str.removeprefix`, no `list[Type]` generics (use `List[Type]` from `typing`).

### Core package structure

```
src/seat_defect_core/
├── api.py                   # Public API: SeatDefectInspector, inspect_once, frames_from_paths
├── config.py                # Dataclass config models (CameraConfig, PatchCoreConfig, etc.)
├── config_file.py           # JSON/INI file loading
├── runtime_config.py        # Config loading + validation entry point
├── runtime_config_parsers.py # Config parsing from raw dicts
├── service/
│   ├── core.py              # InspectionService runtime context, CameraPipeline, model/pipeline caching
│   ├── frames.py            # Normalize external frames into FramePacket
│   ├── inspection.py        # Main inspection orchestration (inspect_frames)
│   ├── inspection_camera.py # Per-camera inspection: detection→ROI→PatchCore→color→artifacts
│   └── response.py          # Build InspectionResponse, error/reject helpers
├── yolo/                    # YOLO detection/segmentation inference (ultralytics-based)
├── cvops/                   # ROI refinement, mask ops, quality guard, region splitting, debug artifacts
├── patchcore/               # PatchCore runtime + color consistency branch
├── fusion.py                # Multi-camera result fusion (any/majority/all strategies)
├── reporting.py             # JSON report output
├── serialization.py         # Result-to-dict conversion
├── types/
│   ├── geometry.py          # BoundingBox
│   ├── input.py             # InspectionFrame, FramePacket
│   ├── pipeline.py          # DetectionResult, RoiRefineResult, ImageQualityDecision
│   └── results.py           # InspectionResult, InspectionResponse, CameraInspectionResult, etc.
└── util.py
```

### Public API entry points

```python
# Reusable inspector (caches pipelines and models)
from seat_defect_core import SeatDefectInspector, InspectionFrame
inspector = SeatDefectInspector("configs/seat_defect_inspection.mvs.json")
response = inspector.inspect(frames=[...], part_id="seat_001")
inspector.warmup()  # preload YOLO + PatchCore models

# One-shot convenience
from seat_defect_core import inspect_once
response = inspect_once("config.json", frames=[{"camera_id": "cam_0", "image": img}])
```

## Inspection pipeline flow

```
External frames
  → normalize_inspection_frames (validate camera IDs, build FramePacket)
  → YOLO detection (batched across cameras with same model config)
  → ROI refine (crop, mask erode, edge ignore, alignment resize)
  → Quality guard (Laplacian variance, brightness, over/underexposure)
  → PatchCore texture analysis
       ├── Full-ROI mode (no regions configured): single PatchCore model per camera
       └── Region mode (regions configured): split ROI into sub-regions, each with own PatchCore model
  → Color consistency branch (optional, per-camera)
  → Per-camera result: OK / NG / REJECT
  → Multi-camera fusion (any/majority/all NG strategy)
  → Debug artifacts + JSON report
```

## Configuration model

Config is loaded from JSON (or INI) files. The file has a top-level `seat_defect_inspection` key. `InspectionConfig` is a flat dataclass hierarchy: `InspectionConfig` → `CameraConfig` → `DetectionConfig`, `RoiRefineConfig`, `PatchCoreConfig`, `ColorBranchConfig`, `RegionConfig[]`, `QualityGuardConfig`.

Config validation happens eagerly at load time (`load_config` → `validate_inspection_config`) and checks:
- Duplicate `camera_id` within each camera list
- Duplicate `region_id` within each camera
- `default_seat_model_id` exists in `seat_models` (if both are configured)
- PatchCore `backend` is `"full"` (only supported backend)
- `backbone_pretrained` or `backbone_weights_path` is set when backend is `"full"`

## Key design decisions

- **Lazy imports**: Heavy dependencies (YOLO via ultralytics, PatchCore, the `SeatDefectInspector` class) use `__getattr__`-based lazy loading in package `__init__.py` files to keep import time fast.
- **PatchCore batching**: Cameras with identical full-backend feature extractor configs share a single `_TorchPatchFeatureExtractor` instance. Regions within and across cameras are batched together via `PatchCorePredictorPool`.
- **Pipeline caching**: `CameraPipeline` instances are cached per `seat_model_id`. Model bundles (`LoadedModelBundle`) are cached by a compound key of model path mtime, pipeline signature, and seat model ID.
- **Pipeline signature**: A SHA-256 hash of the detection/ROI/quality config ensures models trained under different pipeline parameters aren't accidentally reused.
- **Regions vs full-ROI**: When any region is `enabled` on a camera, the camera uses region mode — the full-ROI PatchCore model is not used for texture, but the color branch still loads the camera-level model for its color profile.

## Engineering tool layer

```
src/seat_defect_inspection/
├── cli.py                    # argparse entry point, routes to subcommands
├── cli_commands/             # capture, inspect, inspect_folder, train_patchcore, train_yolo, benchmark
├── acquisition.py            # Unified image source acquisition (file, MVS camera, video)
├── service/                  # CaptureService, InspectionService (thin wrapper), OfflineInspectionService, TrainingService
├── patchcore/training.py     # PatchCore training orchestration (replays core pipeline on training images)
├── yolo/                     # YOLO training, dataset validation, LabelMe→YOLO conversion
├── config.py / runtime_config.py  # Extended config with tool-layer fields (capture_dir, yolo_training, train_good_dir)
└── schemas.py                # Tool-layer JSON schema generation
```

## Rules for modifying this project

1. All inspection logic changes go in `seat_defect_core`, never in the tool layer.
2. Public API exports come from `seat_defect_core`, not `seat_defect_inspection`.
3. The core config parser must reject unknown fields rather than silently ignoring them.
4. When adding a new config field, update `config.py`, `runtime_config_parsers.py`, and `types/` together.
