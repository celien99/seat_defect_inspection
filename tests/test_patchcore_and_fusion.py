from __future__ import annotations

import json

import numpy as np

from seat_defect_inspection.config import CameraConfig, DetectionConfig, FusionConfig, PatchCoreConfig, QualityGuardConfig, RoiRefineConfig
from seat_defect_inspection.detection import DetectionService
from seat_defect_inspection.fusion import fuse_camera_results, should_early_stop_on_ng
from seat_defect_inspection.patchcore import PatchCoreService, _decide_patchcore_anomaly
from seat_defect_inspection.quality import ImageQualityGuard
from seat_defect_inspection.runtime_config import load_yolo_training_config
from seat_defect_inspection.schemas import BoundingBox, CameraInspectionResult
from seat_defect_inspection.service import _CameraPipeline


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
                        "model_path": "yolo11n.pt",
                        "data_config_path": "dataset.yaml",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_yolo_training_config(str(config_path))

    assert config.model_path == "yolo11n.pt"
    assert config.data_config_path == str(dataset_path.resolve())


def test_yolo_training_alias_is_preserved_for_example_config() -> None:
    config = load_yolo_training_config(
        "configs/seat_defect_inspection.multimodel.example.json",
        seat_model_id="seat_model_a",
    )

    assert config.model_path == "yolo11n.pt"


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
            mask_mode="full",
            morphology_kernel_size=1,
            ignore_dilate_kernel_size=1,
            edge_ignore_pixels=0,
        ),
    )

    pipeline = _CameraPipeline(camera)
    prepared = pipeline.prepare_image(image)

    assert full_frame_decision.accepted is False
    assert full_frame_decision.reason == "underexposed"
    assert prepared.quality is not None
    assert prepared.quality.accepted is True
    assert prepared.rejection_reason is None


def test_full_patchcore_fit_predict_and_reload(tmp_path) -> None:
    config = PatchCoreConfig(
        backend="full",
        image_size=64,
        max_memory=32,
        texture_input="lab_l",
        backbone_name="resnet18",
        feature_layers=["layer2", "layer3"],
        backbone_pretrained=False,
        backbone_device="cpu",
        feature_pool_kernel_size=3,
        coreset_sampling_ratio=0.5,
    )
    service = PatchCoreService(config)

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
