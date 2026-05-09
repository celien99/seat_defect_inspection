from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import seat_defect_core.patchcore.engine as patchcore_engine
import seat_defect_inspection.patchcore.training as patchcore_training
from seat_defect_core.config import AlignmentConfig, ColorBranchConfig, DetectionConfig, FusionConfig, PatchCoreConfig, QualityGuardConfig, RoiRefineConfig
from seat_defect_core.cvops import ImageQualityGuard, RoiRefineEngine
from seat_defect_core.fusion import fuse_camera_results, should_early_stop_on_ng
from seat_defect_core.patchcore import PatchCoreService, _decide_patchcore_anomaly
from seat_defect_core.patchcore.features import _PatchBatch, _prepare_feature_image
from seat_defect_core.patchcore.scoring import _analyze_patch_evidence, normalize_map_against_threshold
from seat_defect_core.service.inspection_camera import inspect_one_camera
from seat_defect_core.types import BoundingBox, CameraInspectionResult, DetectionResult, DetectionObject, FramePacket, ImageQualityDecision, ImageQualityMetrics, RoiRefineResult, TextureAnomalyResult
from seat_defect_core.yolo import DetectionService
from seat_defect_inspection.config import CameraConfig
from seat_defect_inspection.patchcore import PatchCoreTrainer
from seat_defect_inspection.runtime_config import load_yolo_training_config
from seat_defect_inspection.service import CameraPipeline, InspectionService, PreparedCameraSample
from seat_defect_inspection.service.inspection import run_inspection as run_online_inspection
from seat_defect_inspection.service.offline_inspection import inspect_image_folder
from seat_defect_inspection.service.training import train_patchcore_models


def _install_stubbed_full_patchcore(monkeypatch) -> None:
    """Avoid torch/backbone dependencies while keeping tests on the full backend contract."""

    def fake_get_torch_feature_extractor(self):
        return object()

    def fake_extract_patch_embeddings(
        image,
        _config,
        *,
        target_mask=None,
        ignore_mask=None,
        feature_extractor=None,
    ):
        mean_value = float(np.asarray(image)[:, :, :3].mean())
        base = 0.0 if mean_value < 128.0 else 10.0
        embeddings = np.asarray(
            [
                [base, base + 0.1],
                [base + 0.2, base + 0.3],
            ],
            dtype=np.float32,
        )
        batch = _PatchBatch(
            grid_shape=(1, 2),
            valid_indices=np.asarray([0, 1], dtype=np.int64),
            valid_patch_count=2,
            total_patch_count=2,
        )
        return embeddings, batch

    monkeypatch.setattr(
        PatchCoreService,
        "_get_torch_feature_extractor",
        fake_get_torch_feature_extractor,
    )
    monkeypatch.setattr(
        PatchCoreTrainer,
        "_get_torch_feature_extractor",
        fake_get_torch_feature_extractor,
    )
    monkeypatch.setattr(
        patchcore_engine,
        "extract_patch_embeddings",
        fake_extract_patch_embeddings,
    )
    monkeypatch.setattr(
        patchcore_training,
        "extract_patch_embeddings",
        fake_extract_patch_embeddings,
    )


def _camera_result(camera_id: str, status: str) -> CameraInspectionResult:
    return CameraInspectionResult(
        camera_id=camera_id,
        frame_id=f"{camera_id}_frame",
        source=f"{camera_id}.png",
        source_kind="image",
        status=status,
        reason=status.lower(),
    )


def test_patchcore_normal_rule_triggers_on_balanced_evidence() -> None:
    config = PatchCoreConfig()
    evidence = {
        "peak_patch_score": 1.50,
        "strong_patch_count": 4,
        "largest_component_patch_count": 3,
        "strong_patch_ratio": 0.04,
        "largest_component_patch_ratio": 0.03,
    }

    is_anomaly, decision_mode = _decide_patchcore_anomaly(
        score=1.12,
        threshold=1.0,
        evidence=evidence,
        config=config,
    )

    assert is_anomaly is True
    assert decision_mode in {"normal_rule", "normal_and_critical"}


def test_patchcore_critical_rule_triggers_for_strong_local_defect() -> None:
    config = PatchCoreConfig()
    evidence = {
        "peak_patch_score": 1.60,
        "strong_patch_count": 2,
        "largest_component_patch_count": 2,
        "strong_patch_ratio": 0.008,
        "largest_component_patch_ratio": 0.008,
    }

    is_anomaly, decision_mode = _decide_patchcore_anomaly(
        score=1.40,
        threshold=1.0,
        evidence=evidence,
        config=config,
    )

    assert is_anomaly is True
    assert decision_mode == "critical_rule"


def test_patchcore_rejects_isolated_small_response() -> None:
    config = PatchCoreConfig()
    evidence = {
        "peak_patch_score": 1.20,
        "strong_patch_count": 1,
        "largest_component_patch_count": 1,
        "strong_patch_ratio": 0.005,
        "largest_component_patch_ratio": 0.005,
    }

    is_anomaly, decision_mode = _decide_patchcore_anomaly(
        score=1.10,
        threshold=1.0,
        evidence=evidence,
        config=config,
    )

    assert is_anomaly is False
    assert decision_mode == "none"


def test_patchcore_peak_rule_triggers_for_small_local_defect() -> None:
    config = PatchCoreConfig()
    evidence = {
        "peak_patch_score": 1.20,
        "strong_patch_count": 2,
        "largest_component_patch_count": 2,
        "strong_patch_ratio": 0.008,
        "largest_component_patch_ratio": 0.008,
        "decision_patch_count": 2,
        "largest_decision_component_patch_count": 2,
    }

    is_anomaly, decision_mode = _decide_patchcore_anomaly(
        score=0.72,
        threshold=1.0,
        evidence=evidence,
        config=config,
    )

    assert is_anomaly is True
    assert decision_mode == "peak_rule"


def test_patchcore_peak_rule_rejects_single_patch_noise() -> None:
    config = PatchCoreConfig(critical_min_component_patch_count=1)
    evidence = {
        "peak_patch_score": 1.60,
        "strong_patch_count": 1,
        "largest_component_patch_count": 1,
        "strong_patch_ratio": 0.01,
        "largest_component_patch_ratio": 0.01,
        "decision_patch_count": 1,
        "largest_decision_component_patch_count": 1,
    }

    is_anomaly, decision_mode = _decide_patchcore_anomaly(
        score=0.72,
        threshold=1.0,
        evidence=evidence,
        config=config,
    )

    assert is_anomaly is True
    assert decision_mode == "peak_rule"


def test_patchcore_peak_rule_triggers_for_cam0_like_visible_defect() -> None:
    config = PatchCoreConfig(
        decision_score_margin=1.05,
        strong_patch_score_ratio=0.85,
        min_strong_patch_count=2,
        min_strong_component_count=2,
        min_strong_patch_ratio=0.006,
        min_strong_component_ratio=0.004,
        critical_score_margin=1.1,
        critical_peak_score_margin=1.15,
        critical_min_component_patch_count=2,
    )
    evidence = {
        "peak_patch_score": 6.05,
        "strong_patch_count": 6,
        "largest_component_patch_count": 3,
        "strong_patch_ratio": 0.0049504950495,
        "largest_component_patch_ratio": 0.0024752475247,
        "decision_patch_count": 3,
        "largest_decision_component_patch_count": 2,
    }

    is_anomaly, decision_mode = _decide_patchcore_anomaly(
        score=4.2155,
        threshold=5.3269,
        evidence=evidence,
        config=config,
    )

    assert is_anomaly is True
    assert decision_mode == "peak_rule"


def test_patchcore_peak_rule_rejects_cam1_like_small_hotspot() -> None:
    config = PatchCoreConfig(
        decision_score_margin=1.05,
        strong_patch_score_ratio=0.85,
        min_strong_patch_count=2,
        min_strong_component_count=2,
        min_strong_patch_ratio=0.006,
        min_strong_component_ratio=0.004,
        critical_score_margin=1.1,
        critical_peak_score_margin=1.15,
        critical_min_component_patch_count=2,
    )
    evidence = {
        "peak_patch_score": 6.3773,
        "strong_patch_count": 3,
        "largest_component_patch_count": 2,
        "strong_patch_ratio": 0.0031678986272,
        "largest_component_patch_ratio": 0.0021119324181,
        "decision_patch_count": 1,
        "largest_decision_component_patch_count": 1,
    }

    is_anomaly, decision_mode = _decide_patchcore_anomaly(
        score=4.5177,
        threshold=5.8787,
        evidence=evidence,
        config=config,
    )

    assert is_anomaly is True
    assert decision_mode == "peak_rule"


def test_patchcore_strong_patch_floor_respects_ratio() -> None:
    config = PatchCoreConfig(strong_patch_score_ratio=0.8)
    patch_map = np.asarray(
        [
            [0.0, 3.2],
            [3.3, 0.0],
        ],
        dtype=np.float32,
    )

    evidence = _analyze_patch_evidence(
        patch_map,
        score=1.0,
        threshold=4.0,
        valid_patch_count=4,
        config=config,
    )

    assert int(evidence["strong_patch_count"]) == 2
    assert int(evidence["largest_component_patch_count"]) == 2


def test_patchcore_evidence_tracks_decision_threshold_components() -> None:
    config = PatchCoreConfig(
        decision_score_margin=1.0,
        strong_patch_score_ratio=0.8,
    )
    patch_map = np.asarray(
        [
            [0.0, 1.3, 1.35],
            [0.0, 0.9, 0.2],
        ],
        dtype=np.float32,
    )

    evidence = _analyze_patch_evidence(
        patch_map,
        score=0.95,
        threshold=1.0,
        valid_patch_count=6,
        config=config,
    )

    assert int(evidence["decision_patch_count"]) == 2
    assert int(evidence["largest_decision_component_patch_count"]) == 2


def test_threshold_normalized_heatmap_stays_cool_below_threshold() -> None:
    heatmap = np.asarray(
        [
            [20.0, 40.0],
            [45.0, 49.0],
        ],
        dtype=np.float32,
    )

    normalized = normalize_map_against_threshold(heatmap, threshold=100.0)

    assert float(normalized.max()) < 0.01


def test_full_patchcore_threshold_uses_sample_exclusive_calibration(monkeypatch) -> None:
    _install_stubbed_full_patchcore(monkeypatch)
    config = PatchCoreConfig(
        backend="full",
        image_size=4,
        patch_size=4,
        stride=4,
        max_memory=2,
        texture_input="gray",
        coreset_sampling_ratio=1.0,
    )
    service = PatchCoreTrainer(config)

    target_mask = np.ones((4, 4), dtype=np.uint8)
    ignore_mask = np.zeros((4, 4), dtype=np.uint8)
    dark = np.zeros((4, 4, 3), dtype=np.uint8)
    bright = np.full((4, 4, 3), 255, dtype=np.uint8)

    summary = service.fit(
        [
            (dark, target_mask, ignore_mask),
            (bright, target_mask, ignore_mask),
        ]
    )

    assert float(summary["threshold"]) > 1e-3
    dark_score = service.predict(dark, target_mask, ignore_mask).score
    bright_score = service.predict(bright, target_mask, ignore_mask).score
    assert float(summary["threshold"]) > max(dark_score, bright_score)


def test_fusion_ng_can_override_reject() -> None:
    fusion = FusionConfig(
        reject_on_any_reject=True,
        ng_strategy="any",
        defect_overrides_reject=True,
    )
    camera_results = [_camera_result("cam_0", "REJECT"), _camera_result("cam_1", "NG")]

    result = fuse_camera_results(
        part_id="seat_001",
        frame_id="frame_001",
        timestamp="2026-04-19T12:00:00+08:00",
        camera_results=camera_results,
        fusion_config=fusion,
    )

    assert result.status == "NG"
    assert "override_reject" in result.decision_reason


def test_fail_fast_any_ng_stops_remaining_cameras() -> None:
    fusion = FusionConfig(
        reject_on_any_reject=True,
        ng_strategy="any",
        early_stop_on_ng=True,
        defect_overrides_reject=True,
    )
    camera_results = [_camera_result("cam_0", "NG")]

    assert should_early_stop_on_ng(
        camera_results=camera_results,
        total_camera_count=3,
        fusion_config=fusion,
    )


def test_fail_fast_disabled_when_reject_must_override() -> None:
    fusion = FusionConfig(
        reject_on_any_reject=True,
        ng_strategy="any",
        early_stop_on_ng=True,
        defect_overrides_reject=False,
    )
    camera_results = [_camera_result("cam_0", "NG")]

    assert not should_early_stop_on_ng(
        camera_results=camera_results,
        total_camera_count=3,
        fusion_config=fusion,
    )


def test_run_inspection_captures_all_cameras_before_processing(monkeypatch) -> None:
    cameras = [
        CameraConfig(camera_id="cam_0", source="0", patchcore_model_path="model_0.npz"),
        CameraConfig(camera_id="cam_1", source="1", patchcore_model_path="model_1.npz"),
    ]
    events: list[str] = []
    started: set[str] = set()
    all_captures_started = threading.Event()
    lock = threading.Lock()

    class _FakeAcquisition:
        def capture(self, camera_id: str, source: str, part_id: str) -> FramePacket:
            with lock:
                events.append(f"capture_start:{camera_id}")
                started.add(camera_id)
                if len(started) == len(cameras):
                    all_captures_started.set()
            if not all_captures_started.wait(timeout=1.0):
                raise AssertionError("captures did not run concurrently")
            with lock:
                events.append(f"capture_done:{camera_id}")
            return FramePacket(
                camera_id=camera_id,
                frame_id=f"{camera_id}_frame",
                part_id=part_id,
                source=source,
                source_kind="camera_index",
                timestamp=f"2026-04-28T00:00:0{camera_id[-1]}+08:00",
                image=np.zeros((8, 8, 3), dtype=np.uint8),
            )

    service = SimpleNamespace(
        config=SimpleNamespace(
            part_id="seat_demo",
            fusion=FusionConfig(),
        ),
        acquisition=_FakeAcquisition(),
        resolve_context=lambda _seat_model_id: SimpleNamespace(
            cameras=cameras,
            pipelines={camera.camera_id: object() for camera in cameras},
            seat_model_id=None,
        ),
    )

    def fake_inspect_frames(_service, frames, *, part_id=None, seat_model_id=None):
        for frame in frames:
            events.append(f"inspect:{frame.camera_id}")
        return fuse_camera_results(
            part_id=part_id or "seat_demo",
            frame_id=frames[0].frame_id or "",
            timestamp=frames[0].timestamp or "",
            camera_results=[_camera_result(frame.camera_id, "OK") for frame in frames],
            fusion_config=FusionConfig(),
        )

    monkeypatch.setattr(
        "seat_defect_inspection.service.inspection.inspect_frames",
        fake_inspect_frames,
    )

    result = run_online_inspection(service)

    first_inspect_index = next(
        index
        for index, event in enumerate(events)
        if event.startswith("inspect:")
    )
    assert result.status == "OK"
    assert all(
        events.index(f"capture_done:{camera.camera_id}") < first_inspect_index
        for camera in cameras
    )
    assert set(event for event in events if event.startswith("inspect:")) == {
        "inspect:cam_0",
        "inspect:cam_1",
    }


def test_run_inspection_processes_cameras_concurrently_when_fail_fast_is_disabled(monkeypatch) -> None:
    cameras = [
        CameraConfig(camera_id="cam_0", source="0", patchcore_model_path="model_0.npz"),
        CameraConfig(camera_id="cam_1", source="1", patchcore_model_path="model_1.npz"),
    ]
    events: list[str] = []
    started: set[str] = set()
    all_inspections_started = threading.Event()
    lock = threading.Lock()

    class _FakeAcquisition:
        def capture(self, camera_id: str, source: str, part_id: str) -> FramePacket:
            events.append(f"capture_done:{camera_id}")
            return FramePacket(
                camera_id=camera_id,
                frame_id=f"{camera_id}_frame",
                part_id=part_id,
                source=source,
                source_kind="camera_index",
                timestamp=f"2026-04-28T00:00:0{camera_id[-1]}+08:00",
                image=np.zeros((8, 8, 3), dtype=np.uint8),
            )

    service = SimpleNamespace(
        config=SimpleNamespace(
            part_id="seat_demo",
            fusion=FusionConfig(early_stop_on_ng=False),
        ),
        acquisition=_FakeAcquisition(),
        resolve_context=lambda _seat_model_id: SimpleNamespace(
            cameras=cameras,
            pipelines={camera.camera_id: object() for camera in cameras},
            seat_model_id=None,
        ),
    )

    def fake_inspect_frames(_service, frames, *, part_id=None, seat_model_id=None):
        for frame in frames:
            events.append(f"inspect:{frame.camera_id}")
        return fuse_camera_results(
            part_id=part_id or "seat_demo",
            frame_id=frames[0].frame_id or "",
            timestamp=frames[0].timestamp or "",
            camera_results=[_camera_result(frame.camera_id, "OK") for frame in frames],
            fusion_config=FusionConfig(early_stop_on_ng=False),
        )

    monkeypatch.setattr(
        "seat_defect_inspection.service.inspection.inspect_frames",
        fake_inspect_frames,
    )

    result = run_online_inspection(service)

    first_inspect_index = next(
        index
        for index, event in enumerate(events)
        if event.startswith("inspect:")
    )
    assert result.status == "OK"
    assert [item.camera_id for item in result.camera_results] == ["cam_0", "cam_1"]
    assert all(
        events.index(f"capture_done:{camera.camera_id}") < first_inspect_index
        for camera in cameras
    )


def test_wrapped_top_level_yolo_training_is_loaded(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    dataset_path = tmp_path / "dataset.yaml"
    dataset_path.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: seat_main\n",
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            {
                "seat_defect_inspection": {
                    "cameras": [
                        {
                            "camera_id": "cam_0",
                            "source": "0",
                            "patchcore_model_path": "model.npz",
                        }
                    ],
                    "yolo_training": {
                        "model_path": "yolo11m-seg.pt",
                        "data_config_path": "dataset.yaml",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_yolo_training_config(str(config_path))

    assert config.model_path == "yolo11m-seg.pt"
    assert config.data_config_path == str(dataset_path.resolve())


def test_yolo_training_alias_is_preserved_for_example_config() -> None:
    config = load_yolo_training_config(
        "configs/seat_defect_inspection.multimodel.example.json",
        seat_model_id="seat_model_a",
    )

    assert config.model_path == "yolo11m-seg.pt"


def test_detection_does_not_use_fallback_when_yolo_misses() -> None:
    class _EmptyBoxes:
        xyxy = None

    class _EmptyResult:
        boxes = _EmptyBoxes()
        names = {}

    class _FakeModel:
        def predict(self, *_args, **_kwargs):
            return [_EmptyResult()]

    service = DetectionService(
        DetectionConfig(
            model_path="dummy.pt",
            fallback_box=BoundingBox(1.0, 2.0, 30.0, 40.0),
        )
    )
    service._model = _FakeModel()

    result = service.detect(np.zeros((64, 64, 3), dtype=np.uint8))

    assert result.target is None
    assert result.all_objects == []


def test_legacy_patchcore_bundle_without_color_profile_loads(tmp_path) -> None:
    model_path = tmp_path / "legacy_patchcore.npz"
    np.savez_compressed(
        model_path,
        memory_bank=np.zeros((4, 8), dtype=np.float32),
        feature_mean=np.zeros((8,), dtype=np.float32),
        feature_std=np.ones((8,), dtype=np.float32),
        meta_json=np.array(
            json.dumps(
                {
                    "backend": "full",
                    "image_size": 256,
                    "patch_size": 32,
                    "stride": 16,
                    "max_memory": 128,
                    "threshold_quantile": 0.99,
                    "threshold": 1.0,
                }
            )
        ),
    )

    bundle = PatchCoreService.load_bundle(model_path)

    assert bundle.color_profile is None
    assert bundle.patchcore.threshold == 1.0


def test_load_bundle_applies_runtime_patchcore_overrides(tmp_path) -> None:
    model_path = tmp_path / "runtime_override_patchcore.npz"
    np.savez_compressed(
        model_path,
        memory_bank=np.zeros((4, 8), dtype=np.float32),
        feature_mean=np.zeros((8,), dtype=np.float32),
        feature_std=np.ones((8,), dtype=np.float32),
        meta_json=np.array(
            json.dumps(
                {
                    "backend": "full",
                    "image_size": 256,
                    "patch_size": 32,
                    "stride": 16,
                    "max_memory": 128,
                    "threshold_quantile": 0.99,
                    "texture_input": "lab_l",
                    "min_target_coverage": 0.5,
                    "max_ignore_overlap": 0.1,
                    "min_valid_patch_ratio": 0.3,
                    "decision_score_margin": 0.9,
                    "strong_patch_score_ratio": 0.8,
                    "min_strong_patch_count": 2,
                    "min_strong_component_count": 1,
                    "min_strong_patch_ratio": 0.005,
                    "min_strong_component_ratio": 0.003,
                    "critical_score_margin": 0.8,
                    "critical_peak_score_margin": 0.9,
                    "critical_min_component_patch_count": 1,
                    "threshold": 1.0,
                }
            )
        ),
        color_profile_json=np.array(""),
    )

    runtime_config = PatchCoreConfig(
        image_size=320,
        patch_size=16,
        stride=8,
        min_target_coverage=0.7,
        max_ignore_overlap=0.05,
        min_valid_patch_ratio=0.55,
        decision_score_margin=1.05,
        critical_score_margin=1.15,
        critical_peak_score_margin=1.25,
        critical_min_component_patch_count=2,
    )

    bundle = PatchCoreService.load_bundle(model_path, runtime_config=runtime_config)

    assert bundle.patchcore.config.image_size == 256
    assert bundle.patchcore.config.patch_size == 32
    assert bundle.patchcore.config.stride == 16
    assert bundle.patchcore.config.min_target_coverage == 0.7
    assert bundle.patchcore.config.max_ignore_overlap == 0.05
    assert bundle.patchcore.config.min_valid_patch_ratio == runtime_config.min_valid_patch_ratio
    assert bundle.patchcore.config.decision_score_margin == runtime_config.decision_score_margin
    assert bundle.patchcore.config.critical_peak_score_margin == runtime_config.critical_peak_score_margin
    assert bundle.patchcore.config.critical_min_component_patch_count == 2


def test_load_bundle_rejects_runtime_backend_mismatch(tmp_path) -> None:
    model_path = tmp_path / "backend_mismatch_patchcore.npz"
    np.savez_compressed(
        model_path,
        memory_bank=np.zeros((4, 8), dtype=np.float32),
        feature_mean=np.zeros((8,), dtype=np.float32),
        feature_std=np.ones((8,), dtype=np.float32),
        meta_json=np.array(
            json.dumps(
                {
                    "backend": "handcrafted",
                    "image_size": 256,
                    "patch_size": 32,
                    "stride": 16,
                    "max_memory": 128,
                    "threshold_quantile": 0.99,
                    "threshold": 1.0,
                }
            )
        ),
        color_profile_json=np.array(""),
    )

    try:
        PatchCoreService.load_bundle(model_path, runtime_config=PatchCoreConfig())
    except RuntimeError as exc:
        assert "backend" in str(exc)
        assert "Please retrain" in str(exc)
        return
    raise AssertionError("expected RuntimeError for PatchCore backend mismatch")


def test_load_bundle_rejects_pipeline_signature_mismatch(tmp_path) -> None:
    model_path = tmp_path / "pipeline_signature_patchcore.npz"
    np.savez_compressed(
        model_path,
        memory_bank=np.zeros((4, 8), dtype=np.float32),
        feature_mean=np.zeros((8,), dtype=np.float32),
        feature_std=np.ones((8,), dtype=np.float32),
        meta_json=np.array(
            json.dumps(
                {
                    "backend": "full",
                    "image_size": 256,
                    "patch_size": 32,
                    "stride": 16,
                    "max_memory": 128,
                    "threshold_quantile": 0.99,
                    "threshold": 1.0,
                    "pipeline_signature": "sig_train",
                    "pipeline_context": {"signature_version": 1},
                }
            )
        ),
        color_profile_json=np.array(""),
    )

    try:
        PatchCoreService.load_bundle(
            model_path,
            expected_pipeline_signature="sig_runtime",
        )
    except RuntimeError as exc:
        assert "Please retrain" in str(exc)
        return
    raise AssertionError("expected RuntimeError for mismatched PatchCore pipeline signature")


def test_inspection_service_rejects_missing_color_profile_when_color_branch_enabled(tmp_path) -> None:
    model_path = tmp_path / "legacy_patchcore.npz"
    np.savez_compressed(
        model_path,
        memory_bank=np.zeros((4, 8), dtype=np.float32),
        feature_mean=np.zeros((8,), dtype=np.float32),
        feature_std=np.ones((8,), dtype=np.float32),
        meta_json=np.array(
            json.dumps(
                {
                    "backend": "full",
                    "image_size": 256,
                    "patch_size": 32,
                    "stride": 16,
                    "max_memory": 128,
                    "threshold_quantile": 0.99,
                    "threshold": 1.0,
                }
            )
        ),
    )

    camera = CameraConfig(
        camera_id="cam_0",
        source="0",
        patchcore_model_path=str(model_path),
        color_branch=ColorBranchConfig(enabled=True),
    )
    service = InspectionService(
        SimpleNamespace(
            cameras=[camera],
            seat_models=[],
            default_seat_model_id=None,
            output_json_path="results.json",
            debug_dir=str(tmp_path / "debug"),
            capture_dir="capture",
            capture_retries=1,
            part_id="seat_demo",
            fusion=FusionConfig(),
        )
    )

    service.build_patchcore_pipeline_signature = lambda _camera: None  # type: ignore[method-assign]

    try:
        service.load_model_bundle(camera, None)
    except RuntimeError as exc:
        assert "颜色分支" in str(exc)
        assert "train-patchcore" in str(exc)
        return
    raise AssertionError("expected RuntimeError for missing color profile")


def test_inspection_service_shares_matching_patchcore_feature_extractors(tmp_path, monkeypatch) -> None:
    created_configs: list[PatchCoreConfig] = []

    class _FakeFeatureExtractor:
        def __init__(self, config: PatchCoreConfig) -> None:
            created_configs.append(config)

    monkeypatch.setattr(
        "seat_defect_core.service.core._TorchPatchFeatureExtractor",
        _FakeFeatureExtractor,
    )

    config = PatchCoreConfig(
        backend="full",
        image_size=128,
        backbone_pretrained=True,
        backbone_device="cpu",
    )
    service = InspectionService(
        SimpleNamespace(
            cameras=[],
            seat_models=[],
            default_seat_model_id=None,
            output_json_path="results.json",
            debug_dir=str(tmp_path / "debug"),
            capture_dir="capture",
            capture_retries=1,
            part_id="seat_demo",
            fusion=FusionConfig(),
        )
    )
    first = PatchCoreService(
        config,
        memory_bank=np.zeros((1, 2), dtype=np.float32),
        feature_mean=np.zeros((2,), dtype=np.float32),
        feature_std=np.ones((2,), dtype=np.float32),
        threshold=1.0,
    )
    second = PatchCoreService(
        config,
        memory_bank=np.zeros((1, 2), dtype=np.float32),
        feature_mean=np.zeros((2,), dtype=np.float32),
        feature_std=np.ones((2,), dtype=np.float32),
        threshold=1.0,
    )

    service.prepare_patchcore_for_predict(first)
    service.prepare_patchcore_for_predict(second)

    assert len(created_configs) == 1
    assert first._get_torch_feature_extractor() is second._get_torch_feature_extractor()


def test_inspection_service_batches_matching_region_patchcores(tmp_path, monkeypatch) -> None:
    extract_many_calls: list[int] = []

    class _FakeFeatureExtractor:
        def __init__(self, _config: PatchCoreConfig) -> None:
            pass

        def extract_many(self, samples):
            extract_many_calls.append(len(samples))
            return [
                (
                    np.asarray([[0.0, 0.0]], dtype=np.float32),
                    _PatchBatch(
                        grid_shape=(1, 1),
                        valid_indices=np.asarray([0], dtype=np.int32),
                        valid_patch_count=1,
                        total_patch_count=1,
                    ),
                )
                for _sample in samples
            ]

    monkeypatch.setattr(
        "seat_defect_core.service.core._TorchPatchFeatureExtractor",
        _FakeFeatureExtractor,
    )

    config = PatchCoreConfig(
        backend="full",
        image_size=32,
        backbone_pretrained=True,
        backbone_device="cpu",
    )
    service = InspectionService(
        SimpleNamespace(
            cameras=[],
            seat_models=[],
            default_seat_model_id=None,
            output_json_path="results.json",
            debug_dir=str(tmp_path / "debug"),
            capture_dir="capture",
            capture_retries=1,
            part_id="seat_demo",
            fusion=FusionConfig(),
        )
    )
    first = PatchCoreService(
        config,
        memory_bank=np.zeros((1, 2), dtype=np.float32),
        feature_mean=np.zeros((2,), dtype=np.float32),
        feature_std=np.ones((2,), dtype=np.float32),
        threshold=1.0,
    )
    second = PatchCoreService(
        config,
        memory_bank=np.zeros((1, 2), dtype=np.float32),
        feature_mean=np.zeros((2,), dtype=np.float32),
        feature_std=np.ones((2,), dtype=np.float32),
        threshold=1.0,
    )
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    target_mask = np.ones((16, 16), dtype=np.uint8)
    ignore_mask = np.zeros((16, 16), dtype=np.uint8)

    results = service.predict_patchcore_batch(
        [
            (first, image, target_mask, ignore_mask),
            (second, image, target_mask, ignore_mask),
        ]
    )

    assert extract_many_calls == [2]
    assert len(results) == 2
    assert all(result.valid_patch_count == 1 for result in results)


def test_inspect_image_folder_restores_caller_state(tmp_path: Path, monkeypatch) -> None:
    input_root = tmp_path / "offline"
    expected_sources: dict[str, str] = {}
    for camera_id in ("cam_0", "cam_1"):
        camera_dir = input_root / camera_id
        camera_dir.mkdir(parents=True, exist_ok=True)
        image_path = camera_dir / f"{camera_id}.png"
        image_path.write_bytes(b"placeholder")
        expected_sources[camera_id] = str(image_path)

    cameras = [
        SimpleNamespace(camera_id="cam_0", source="mvs://cam_0"),
        SimpleNamespace(camera_id="cam_1", source="mvs://cam_1"),
    ]
    context = SimpleNamespace(cameras=cameras, seat_model_id=None)
    service = SimpleNamespace(
        config=SimpleNamespace(
            output_json_path="original_results.json",
            debug_dir="original_debug",
        ),
        resolve_context=lambda _seat_model_id: context,
    )
    observed: dict[str, object] = {}

    def fake_run_inspection(service_arg, part_id=None, *, seat_model_id=None):
        observed["part_id"] = part_id
        observed["output_json_path"] = service_arg.config.output_json_path
        observed["debug_dir"] = service_arg.config.debug_dir
        observed["camera_sources"] = {
            camera.camera_id: camera.source
            for camera in cameras
        }
        return SimpleNamespace(status="OK", decision_reason="all_checks_passed")

    monkeypatch.setattr(
        "seat_defect_inspection.service.offline_inspection.run_inspection",
        fake_run_inspection,
    )
    monkeypatch.setattr(
        "seat_defect_inspection.service.offline_inspection.resolve_inspection_archive_path",
        lambda latest_output_path, _result: latest_output_path.parent / "archived.json",
    )

    summary = inspect_image_folder(
        service,
        str(input_root),
        output_dir=str(tmp_path / "outputs"),
        part_id="sample_001",
    )

    assert observed["part_id"] == "sample_001"
    assert Path(observed["output_json_path"]).name == "latest.json"
    assert Path(observed["debug_dir"]).name == "debug"
    assert observed["camera_sources"] == expected_sources
    assert service.config.output_json_path == "original_results.json"
    assert service.config.debug_dir == "original_debug"
    assert {camera.camera_id: camera.source for camera in cameras} == {
        "cam_0": "mvs://cam_0",
        "cam_1": "mvs://cam_1",
    }
    assert summary["ok_count"] == 1


def test_camera_pipeline_quality_uses_roi_instead_of_full_frame() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    checker = ((np.indices((60, 60)).sum(axis=0) % 2) * 255).astype(np.uint8)
    image[70:130, 70:130] = np.stack([checker, checker, checker], axis=-1)

    full_frame_quality = QualityGuardConfig(
        min_laplacian_variance=20.0,
        min_brightness_mean=30.0,
    )
    full_frame_decision = ImageQualityGuard(full_frame_quality).evaluate(image)
    camera = CameraConfig(
        camera_id="cam_0",
        source="0",
        patchcore_model_path="model.npz",
        quality=full_frame_quality,
        detection=DetectionConfig(
            model_path=None,
            fallback_box=BoundingBox(70.0, 70.0, 130.0, 130.0),
        ),
        roi=RoiRefineConfig(
            edge_ignore_pixels=0,
        ),
    )

    pipeline = CameraPipeline(camera)
    prepared = pipeline.prepare_image(image)

    assert full_frame_decision.accepted is False
    assert full_frame_decision.reason == "underexposed"
    assert prepared.quality is not None
    assert prepared.quality.accepted is True
    assert prepared.rejection_reason is None


def test_train_patchcore_writes_per_image_audit_artifacts(tmp_path: Path, monkeypatch) -> None:
    train_dir = tmp_path / "train_good"
    train_dir.mkdir()
    image_path = train_dir / "sample.png"

    cv2.imwrite(str(image_path), np.full((24, 24, 3), 120, dtype=np.uint8))

    class _FakePatchCore:
        def fit(self, samples):
            self.samples = list(samples)
            return {
                "backend": "full",
                "train_sample_count": len(self.samples),
                "patch_count": 4,
                "memory_bank_size": 2,
                "threshold": 1.0,
            }

        def save(self, model_path, **_kwargs):
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            Path(model_path).write_bytes(b"fake")

    class _FakePipeline:
        def prepare_image(self, _image):
            target_mask = np.ones((16, 16), dtype=np.uint8)
            roi = RoiRefineResult(
                crop_box=BoundingBox(1.0, 2.0, 17.0, 18.0),
                roi_image=np.full((16, 16, 3), 80, dtype=np.uint8),
                aligned_roi_image=np.full((16, 16, 3), 90, dtype=np.uint8),
                texture_ready_image=np.dstack(
                    [
                        np.full((16, 16, 3), 100, dtype=np.uint8),
                        target_mask * 255,
                    ]
                ),
                target_mask=target_mask,
                valid_mask=target_mask,
                ignore_mask=np.zeros((16, 16), dtype=np.uint8),
                foreground_weight=None,
            )
            return PreparedCameraSample(
                quality=ImageQualityDecision(
                    accepted=True,
                    reason=None,
                    metrics=ImageQualityMetrics(
                        laplacian_variance=100.0,
                        brightness_mean=90.0,
                        overexposed_ratio=0.0,
                        underexposed_ratio=0.0,
                        is_black_frame=False,
                        is_white_frame=False,
                    ),
                ),
                detection=DetectionResult(
                    target=DetectionObject(
                        label="seat",
                        confidence=1.0,
                        bounding_box=BoundingBox(1.0, 2.0, 17.0, 18.0),
                    ),
                    used_fallback=True,
                ),
                roi=roi,
                rejection_reason=None,
            )

    camera = CameraConfig(
        camera_id="cam_0",
        source="0",
        patchcore_model_path=str(tmp_path / "models" / "cam_0_patchcore.npz"),
        train_good_dir=str(train_dir),
    )
    service = SimpleNamespace(
        config=SimpleNamespace(seat_models=[]),
        resolve_context=lambda _seat_model_id: SimpleNamespace(
            seat_model_id=None,
            cameras=[camera],
            pipelines={"cam_0": _FakePipeline()},
        ),
        resolve_patchcore_config=lambda camera, region=None: camera.patchcore if region is None else region.patchcore or camera.patchcore,
        build_patchcore_pipeline_context=lambda _camera: {"pipeline": "fake"},
        build_patchcore_pipeline_signature=lambda _camera: "signature",
    )

    monkeypatch.setattr(
        "seat_defect_inspection.service.training.PatchCoreTrainer",
        lambda _config: _FakePatchCore(),
    )

    summaries = train_patchcore_models(service)

    summary = summaries[0]
    assert summary["accepted_image_count"] == 1
    assert summary["skipped_image_count"] == 0
    summary_path = Path(camera.patchcore_model_path).with_suffix(".summary.json")
    assert summary_path.is_file()


def test_full_patchcore_fit_predict_and_reload(tmp_path, monkeypatch) -> None:
    _install_stubbed_full_patchcore(monkeypatch)
    config = PatchCoreConfig(
        backend="full",
        image_size=64,
        max_memory=32,
        texture_input="lab_l",
        coreset_sampling_ratio=0.5,
    )
    service = PatchCoreTrainer(config)

    rng = np.random.default_rng(42)
    samples = []
    for _ in range(3):
        image = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        target_mask = np.ones((64, 64), dtype=np.uint8) * 255
        ignore_mask = np.zeros((64, 64), dtype=np.uint8)
        samples.append((image, target_mask, ignore_mask))

    summary = service.fit(samples)
    assert summary["backend"] == "full"
    assert int(summary["train_sample_count"]) == 3
    assert int(summary["memory_bank_size"]) > 0

    result = service.predict(*samples[0])
    assert result.heatmap.shape == samples[0][0].shape[:2]
    assert result.total_patch_count >= result.valid_patch_count > 0

    model_path = tmp_path / "full_patchcore_test.npz"
    service.save(model_path)
    loaded = PatchCoreService.load_bundle(model_path).patchcore
    reloaded_result = loaded.predict(*samples[1])

    assert reloaded_result.heatmap.shape == samples[1][0].shape[:2]
    assert reloaded_result.total_patch_count >= reloaded_result.valid_patch_count > 0


def test_roi_valid_mask_respects_edge_ignore_pixels() -> None:
    image = np.full((64, 64, 3), 127, dtype=np.uint8)
    segmentation_mask = np.zeros((64, 64), dtype=np.uint8)
    segmentation_mask[8:56, 8:56] = 1
    detection = DetectionResult(
        target=DetectionObject(
            label="seat",
            confidence=1.0,
            bounding_box=BoundingBox(8.0, 8.0, 56.0, 56.0),
            segmentation_mask=segmentation_mask,
        )
    )
    engine = RoiRefineEngine(
        RoiRefineConfig(
            crop_expand_ratio=0.0,
            edge_ignore_pixels=7,
            alignment=AlignmentConfig(
                output_width=64,
                output_height=64,
            ),
        )
    )

    roi = engine.refine(image, detection)

    assert roi.valid_mask.sum() > 0
    assert roi.valid_mask.sum() < roi.target_mask.sum()
    assert roi.ignore_mask.sum() == roi.target_mask.sum() - roi.valid_mask.sum()


def test_roi_rejects_missing_segmentation_mask_without_fallback() -> None:
    image = np.full((64, 64, 3), 127, dtype=np.uint8)
    detection = DetectionResult(
        target=DetectionObject(
            label="seat",
            confidence=1.0,
            bounding_box=BoundingBox(8.0, 8.0, 56.0, 56.0),
        ),
        used_fallback=False,
    )
    engine = RoiRefineEngine(
        RoiRefineConfig(
            alignment=AlignmentConfig(
                output_width=64,
                output_height=64,
            ),
        )
    )

    try:
        engine.refine(image, detection)
    except ValueError as exc:
        assert str(exc) == "target_mask_missing"
        return
    raise AssertionError("expected ValueError for missing segmentation mask")


def test_roi_crop_prefers_segmentation_mask_bounds() -> None:
    image = np.full((80, 80, 3), 127, dtype=np.uint8)
    segmentation_mask = np.zeros((80, 80), dtype=np.uint8)
    segmentation_mask[18:62, 26:54] = 1
    detection = DetectionResult(
        target=DetectionObject(
            label="seat",
            confidence=1.0,
            bounding_box=BoundingBox(8.0, 8.0, 72.0, 72.0),
            segmentation_mask=segmentation_mask,
        )
    )
    engine = RoiRefineEngine(
        RoiRefineConfig(
            crop_expand_ratio=0.0,
            crop_shrink_ratio=0.0,
            edge_ignore_pixels=0,
            alignment=AlignmentConfig(
                output_width=64,
                output_height=64,
            ),
        )
    )

    roi = engine.refine(image, detection)

    assert roi.crop_box.x1 == 26.0
    assert roi.crop_box.y1 == 18.0
    assert roi.crop_box.x2 == 54.0
    assert roi.crop_box.y2 == 62.0


def test_roi_patchcore_input_uses_transparent_background() -> None:
    image = np.full((80, 80, 3), 127, dtype=np.uint8)
    segmentation_mask = np.zeros((80, 80), dtype=np.uint8)
    segmentation_mask[20:60, 30:50] = 1
    detection = DetectionResult(
        target=DetectionObject(
            label="seat",
            confidence=1.0,
            bounding_box=BoundingBox(0.0, 0.0, 80.0, 80.0),
            segmentation_mask=segmentation_mask,
        )
    )
    engine = RoiRefineEngine(
        RoiRefineConfig(
            crop_expand_ratio=0.0,
            crop_shrink_ratio=0.0,
            edge_ignore_pixels=0,
            alignment=AlignmentConfig(
                output_width=64,
                output_height=64,
            ),
        )
    )

    roi = engine.refine(image, detection)

    assert roi.texture_ready_image.shape == (64, 64, 4)
    assert np.array_equal(roi.texture_ready_image[:, :, 3], roi.target_mask * 255)
    assert np.count_nonzero(roi.texture_ready_image[:, :, 3] == 0) > 0


def test_patchcore_feature_image_does_not_turn_transparent_background_black() -> None:
    image = np.zeros((4, 4, 4), dtype=np.uint8)
    image[1:3, 1:3, :3] = np.asarray([20, 80, 140], dtype=np.uint8)
    image[1:3, 1:3, 3] = 255

    prepared = _prepare_feature_image(image)

    assert prepared.shape == (4, 4, 3)
    assert np.array_equal(prepared[0, 0], np.asarray([20, 80, 140], dtype=np.uint8))
    assert np.array_equal(prepared[1, 1], np.asarray([20, 80, 140], dtype=np.uint8))


def test_patchcore_pipeline_context_records_transparent_bgra_input_mode() -> None:
    camera = CameraConfig(
        camera_id="cam_0",
        source="0",
        patchcore_model_path="model.npz",
        patchcore=PatchCoreConfig(backbone_pretrained=True),
    )
    service = InspectionService(
        SimpleNamespace(
            cameras=[camera],
            seat_models=[],
            default_seat_model_id=None,
            output_json_path="results.json",
            debug_dir="debug",
            capture_dir="capture",
            capture_retries=1,
            part_id="seat_demo",
            fusion=FusionConfig(),
        )
    )

    context = service.build_patchcore_pipeline_context(camera)
    signature = service.build_patchcore_pipeline_signature(camera)

    changed_camera = CameraConfig(
        camera_id="cam_0",
        source="0",
        patchcore_model_path="model.npz",
        patchcore=PatchCoreConfig(backbone_pretrained=True),
        roi=RoiRefineConfig(
            alignment=AlignmentConfig(output_width=512, output_height=512),
        ),
    )

    assert context["signature_version"] == 2
    assert context["patchcore_input_mode"] == "transparent_bgra"
    assert service.build_patchcore_pipeline_signature(changed_camera) != signature


def test_inspection_service_passes_target_and_ignore_masks_to_patchcore(tmp_path: Path, monkeypatch) -> None:
    class _FakePatchCore:
        def __init__(self) -> None:
            self.image = None
            self.target_mask = None
            self.ignore_mask = None

        def predict(self, image, target_mask, ignore_mask):
            self.image = image.copy()
            self.target_mask = target_mask.copy()
            self.ignore_mask = ignore_mask.copy()
            return TextureAnomalyResult(
                score=0.1,
                threshold=1.0,
                is_anomaly=False,
                heatmap=np.zeros(target_mask.shape, dtype=np.float32),
                valid_patch_ratio=1.0,
                valid_patch_count=4,
                total_patch_count=4,
            )

    class _FakePipeline:
        def __init__(self, prepared: PreparedCameraSample) -> None:
            self.prepared = prepared

        def prepare_image(self, _image):
            return self.prepared

    camera = CameraConfig(
        camera_id="cam_0",
        source="0",
        patchcore_model_path="model.npz",
    )
    service = InspectionService(
        SimpleNamespace(
            cameras=[camera],
            seat_models=[],
            default_seat_model_id=None,
            output_json_path="results.json",
            debug_dir=str(tmp_path / "debug"),
            capture_dir="capture",
            capture_retries=1,
            part_id="seat_demo",
            fusion=FusionConfig(),
        )
    )

    fake_patchcore = _FakePatchCore()
    service.load_model_bundle = lambda *_args, **_kwargs: SimpleNamespace(  # type: ignore[method-assign]
        patchcore=fake_patchcore,
        color_profile=None,
    )
    target_mask = np.pad(np.ones((16, 16), dtype=np.uint8), 8)
    roi = RoiRefineResult(
        crop_box=BoundingBox(0.0, 0.0, 32.0, 32.0),
        roi_image=np.zeros((32, 32, 3), dtype=np.uint8),
        aligned_roi_image=np.zeros((32, 32, 3), dtype=np.uint8),
        texture_ready_image=np.dstack(
            [
                np.zeros((32, 32, 3), dtype=np.uint8),
                target_mask * 255,
            ]
        ),
        target_mask=target_mask,
        valid_mask=target_mask,
        ignore_mask=np.zeros((32, 32), dtype=np.uint8),
        foreground_weight=None,
    )
    prepared = PreparedCameraSample(
        quality=None,
        detection=DetectionResult(
            target=DetectionObject(
                label="seat",
                confidence=1.0,
                bounding_box=BoundingBox(0.0, 0.0, 32.0, 32.0),
            )
        ),
        roi=roi,
        rejection_reason=None,
    )
    frame_packet = FramePacket(
        camera_id="cam_0",
        frame_id="frame_0",
        part_id="part_0",
        source="0",
        source_kind="image",
        timestamp="2026-04-21T00:00:00+08:00",
        image=np.zeros((32, 32, 3), dtype=np.uint8),
    )

    result = inspect_one_camera(
        service,
        frame_packet,
        camera,
        _FakePipeline(prepared),
        seat_model_id=None,
    )

    assert result.status == "OK"
    assert fake_patchcore.image is not None
    assert fake_patchcore.image.shape == (32, 32, 4)
    assert np.array_equal(fake_patchcore.target_mask, roi.target_mask)
    assert np.array_equal(fake_patchcore.ignore_mask, roi.ignore_mask)
    assert set(result.artifact_paths) >= {"heatmap", "overlay"}
    assert Path(result.artifact_paths["heatmap"]).is_file()
    assert Path(result.artifact_paths["overlay"]).is_file()


def test_quality_reject_can_still_return_ng_when_patchcore_finds_obvious_defect(tmp_path: Path, monkeypatch) -> None:
    class _AnomalousPatchCore:
        def predict(self, _image, target_mask, _ignore_mask):
            return TextureAnomalyResult(
                score=2.0,
                threshold=1.0,
                is_anomaly=True,
                heatmap=np.zeros(target_mask.shape, dtype=np.float32),
                valid_patch_ratio=1.0,
                valid_patch_count=4,
                total_patch_count=4,
                decision_mode="normal_rule",
            )

    class _FakePipeline:
        def __init__(self, prepared: PreparedCameraSample) -> None:
            self.prepared = prepared

        def prepare_image(self, _image):
            return self.prepared

    camera = CameraConfig(
        camera_id="cam_0",
        source="0",
        patchcore_model_path="model.npz",
    )
    service = InspectionService(
        SimpleNamespace(
            cameras=[camera],
            seat_models=[],
            default_seat_model_id=None,
            output_json_path="results.json",
            debug_dir=str(tmp_path / "debug"),
            capture_dir="capture",
            capture_retries=1,
            part_id="seat_demo",
            fusion=FusionConfig(),
        )
    )
    service.load_model_bundle = lambda *_args, **_kwargs: SimpleNamespace(  # type: ignore[method-assign]
        patchcore=_AnomalousPatchCore(),
        color_profile=None,
    )
    roi = RoiRefineResult(
        crop_box=BoundingBox(0.0, 0.0, 32.0, 32.0),
        roi_image=np.zeros((32, 32, 3), dtype=np.uint8),
        aligned_roi_image=np.zeros((32, 32, 3), dtype=np.uint8),
        texture_ready_image=np.zeros((32, 32, 3), dtype=np.uint8),
        target_mask=np.ones((32, 32), dtype=np.uint8),
        valid_mask=np.ones((32, 32), dtype=np.uint8),
        ignore_mask=np.zeros((32, 32), dtype=np.uint8),
        foreground_weight=None,
    )
    prepared = PreparedCameraSample(
        quality=ImageQualityDecision(
            accepted=False,
            reason="blur",
            metrics=ImageQualityMetrics(
                laplacian_variance=1.0,
                brightness_mean=80.0,
                overexposed_ratio=0.0,
                underexposed_ratio=0.0,
                is_black_frame=False,
                is_white_frame=False,
            ),
        ),
        detection=DetectionResult(
            target=DetectionObject(
                label="seat",
                confidence=1.0,
                bounding_box=BoundingBox(0.0, 0.0, 32.0, 32.0),
            )
        ),
        roi=roi,
        rejection_reason="quality_blur",
    )
    frame_packet = FramePacket(
        camera_id="cam_0",
        frame_id="frame_0",
        part_id="part_0",
        source="0",
        source_kind="image",
        timestamp="2026-04-21T00:00:00+08:00",
        image=np.zeros((32, 32, 3), dtype=np.uint8),
    )

    result = inspect_one_camera(
        service,
        frame_packet,
        camera,
        _FakePipeline(prepared),
        seat_model_id=None,
    )

    assert result.status == "NG"
    assert result.reason == "texture_anomaly_quality_override"
    assert set(result.artifact_paths) >= {"heatmap", "overlay"}
