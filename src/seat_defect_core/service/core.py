"""SDK runtime context and cached camera pipelines."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ..config import CameraConfig, InspectionConfig
from ..cvops import ImageQualityGuard, RoiRefineEngine
from ..patchcore import LoadedModelBundle, PatchCoreService
from ..preprocess import PreprocessEngine
from ..schemas import DetectionResult, ImageQualityDecision, RoiRefineResult
from ..yolo import DetectionService


@dataclass(slots=True)
class PreparedCameraSample:
    """Shared intermediate data for one camera."""

    quality: ImageQualityDecision | None
    preprocessed_image: Any | None = None
    detection: DetectionResult | None = None
    roi: RoiRefineResult | None = None
    rejection_reason: str | None = None


@dataclass(slots=True)
class ResolvedInspectionContext:
    """Resolved camera set and pipelines for a model route."""

    seat_model_id: str | None
    cameras: list[CameraConfig]
    pipelines: dict[str, "CameraPipeline"]


class CameraPipeline:
    """Per-camera preprocess, detection, ROI and quality pipeline."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.quality_guard = ImageQualityGuard(config.quality)
        self.preprocess_engine = PreprocessEngine(config.preprocess)
        self.detection_service = DetectionService(config.detection)
        self.roi_refine_engine = RoiRefineEngine(config.roi)

    def prepare_image(self, image: Any) -> PreparedCameraSample:
        preprocessed = self.preprocess_engine.process(image)
        detection = self.detection_service.detect(preprocessed)
        if detection.target is None:
            return PreparedCameraSample(
                quality=None,
                preprocessed_image=preprocessed,
                detection=detection,
                rejection_reason="target_not_found",
            )

        if detection.target.segmentation_mask is None and not detection.used_fallback:
            return PreparedCameraSample(
                quality=None,
                preprocessed_image=preprocessed,
                detection=detection,
                rejection_reason="target_mask_missing",
            )

        try:
            roi = self.roi_refine_engine.refine(preprocessed, detection)
        except ValueError as exc:
            return PreparedCameraSample(
                quality=None,
                preprocessed_image=preprocessed,
                detection=detection,
                rejection_reason=str(exc),
            )
        quality = self.quality_guard.evaluate(roi.roi_image, valid_mask=roi.valid_mask)
        if not quality.accepted:
            return PreparedCameraSample(
                quality=quality,
                preprocessed_image=preprocessed,
                detection=detection,
                roi=roi,
                rejection_reason=f"quality_{quality.reason}",
            )

        return PreparedCameraSample(
            quality=quality,
            preprocessed_image=preprocessed,
            detection=detection,
            roi=roi,
        )


class InspectionService:
    """SDK inspection service without capture or training responsibilities."""

    def __init__(self, config: InspectionConfig) -> None:
        self.config = config
        self._pipeline_cache: dict[str, dict[str, CameraPipeline]] = {}
        self._model_cache: dict[tuple[str, str, str, int], LoadedModelBundle] = {}

    def resolve_context(self, seat_model_id: str | None) -> ResolvedInspectionContext:
        resolved_seat_model_id, cameras = self._resolve_active_cameras(seat_model_id)
        cache_key = resolved_seat_model_id or "__default__"
        pipelines = self._pipeline_cache.get(cache_key)
        if pipelines is None:
            pipelines = {
                camera.camera_id: CameraPipeline(camera)
                for camera in cameras
            }
            self._pipeline_cache[cache_key] = pipelines
        return ResolvedInspectionContext(
            seat_model_id=resolved_seat_model_id,
            cameras=cameras,
            pipelines=pipelines,
        )

    def _resolve_active_cameras(self, seat_model_id: str | None) -> tuple[str | None, list[CameraConfig]]:
        if self.config.seat_models:
            resolved_seat_model_id = (
                seat_model_id
                or self.config.default_seat_model_id
                or self.config.seat_models[0].seat_model_id
            )
            for seat_model in self.config.seat_models:
                if seat_model.seat_model_id == resolved_seat_model_id:
                    return (
                        resolved_seat_model_id,
                        [camera for camera in seat_model.cameras if camera.enabled],
                    )
            available = ", ".join(item.seat_model_id for item in self.config.seat_models)
            raise ValueError(f"未知 seat_model_id `{resolved_seat_model_id}`，可选值：{available}")

        resolved_seat_model_id = seat_model_id or self.config.default_seat_model_id
        return resolved_seat_model_id, [camera for camera in self.config.cameras if camera.enabled]

    def build_patchcore_pipeline_context(self, camera: CameraConfig) -> dict[str, Any]:
        fallback_box = (
            asdict(camera.detection.fallback_box)
            if camera.detection.fallback_box is not None
            else None
        )
        return {
            "signature_version": 2,
            "patchcore_input_mode": "transparent_bgra",
            "color_insensitive_mode": bool(camera.color_insensitive_mode),
            "quality": asdict(camera.quality),
            "preprocess": asdict(camera.preprocess),
            "detection": {
                "model_path": camera.detection.model_path,
                "target_class": camera.detection.target_class,
                "confidence": float(camera.detection.confidence),
                "iou": float(camera.detection.iou),
                "fallback_box": fallback_box,
            },
            "roi": asdict(camera.roi),
        }

    def build_patchcore_pipeline_signature(self, camera: CameraConfig) -> str:
        payload = json.dumps(
            self.build_patchcore_pipeline_context(camera),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def build_patchcore_service(self, camera: CameraConfig) -> PatchCoreService:
        patchcore_config = camera.patchcore
        if (
            camera.color_insensitive_mode
            and patchcore_config.texture_input.strip().lower() not in {"gray", "lab_l"}
        ):
            patchcore_config = replace(patchcore_config, texture_input="lab_l")
        return PatchCoreService(patchcore_config)

    def load_model_bundle(
        self,
        camera: CameraConfig,
        seat_model_id: str | None,
    ) -> LoadedModelBundle:
        pipeline_signature = self.build_patchcore_pipeline_signature(camera)
        model_mtime_ns = Path(camera.patchcore_model_path).stat().st_mtime_ns
        cache_key = (
            seat_model_id or "__default__",
            camera.camera_id,
            pipeline_signature,
            model_mtime_ns,
        )
        bundle = self._model_cache.get(cache_key)
        if bundle is not None:
            return bundle

        loaded = PatchCoreService.load_bundle(
            camera.patchcore_model_path,
            runtime_config=camera.patchcore,
            expected_pipeline_signature=pipeline_signature,
        )
        if (
            camera.color_branch.enabled
            and not camera.color_insensitive_mode
            and loaded.color_profile is None
        ):
            raise RuntimeError(
                f"机位 `{camera.camera_id}` 已启用颜色分支，但模型包缺少颜色参考分布。"
                " 请重新执行 train-patchcore，或关闭颜色分支 / 启用 color_insensitive_mode。"
        )
        self._model_cache[cache_key] = loaded
        return loaded
