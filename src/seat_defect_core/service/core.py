"""Core runtime context and cached camera pipelines."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..config import CameraConfig, InspectionConfig, PatchCoreConfig, RegionConfig
from ..cvops import ImageQualityGuard, RoiRefineEngine
from ..patchcore.features import _TorchPatchFeatureExtractor
from ..patchcore import LoadedModelBundle, PatchCoreService
from ..types import DetectionResult, ImageQualityDecision, RoiRefineResult, TextureAnomalyResult
from ..yolo import DetectionService


@dataclass(slots=True)
class PreparedCameraSample:
    """Shared intermediate data for one camera."""

    quality: ImageQualityDecision | None
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
    """Per-camera detection, ROI and quality pipeline."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.quality_guard = ImageQualityGuard(config.quality)
        self.detection_service = DetectionService(config.detection)
        self.roi_refine_engine = RoiRefineEngine(config.roi)

    def prepare_image(self, image: Any) -> PreparedCameraSample:
        detection = self.detection_service.detect(image)
        return self.prepare_from_detection(image, detection)

    def prepare_from_detection(
        self,
        image: Any,
        detection: DetectionResult,
    ) -> PreparedCameraSample:
        """Run ROI refinement and quality checks from a precomputed detection."""
        if detection.target is None:
            return PreparedCameraSample(
                quality=None,
                detection=detection,
                rejection_reason="target_not_found",
            )

        if detection.target.segmentation_mask is None:
            return PreparedCameraSample(
                quality=None,
                detection=detection,
                rejection_reason="target_mask_missing",
            )

        try:
            roi = self.roi_refine_engine.refine(image, detection)
        except ValueError as exc:
            return PreparedCameraSample(
                quality=None,
                detection=detection,
                rejection_reason=str(exc),
            )
        quality = self.quality_guard.evaluate(
            roi.aligned_roi_image,
            valid_mask=roi.valid_mask,
        )
        if not quality.accepted:
            return PreparedCameraSample(
                quality=quality,
                detection=detection,
                roi=roi,
                rejection_reason=f"quality_{quality.reason}",
            )

        return PreparedCameraSample(
            quality=quality,
            detection=detection,
            roi=roi,
        )


class InspectionService:
    """Core inspection service without capture or training responsibilities."""

    def __init__(self, config: InspectionConfig) -> None:
        self.config = config
        self._pipeline_cache: dict[str, dict[str, CameraPipeline]] = {}
        self._model_cache = ModelBundleCache(self)
        self._patchcore_predictor = PatchCorePredictorPool()

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
        return {
            "signature_version": 2,
            "patchcore_input_mode": "transparent_bgra",
            "color_insensitive_mode": bool(camera.color_insensitive_mode),
            "quality": asdict(camera.quality),
            "detection": {
                "model_path": camera.detection.model_path,
                "target_class": camera.detection.target_class,
                "confidence": float(camera.detection.confidence),
                "iou": float(camera.detection.iou),
            },
            "roi": asdict(camera.roi),
        }

    def build_region_patchcore_pipeline_context(
        self,
        camera: CameraConfig,
        region: RegionConfig,
    ) -> dict[str, Any]:
        context = self.build_patchcore_pipeline_context(camera)
        context["region"] = {
            "region_id": region.region_id,
            "box": [float(value) for value in region.box],
            "patchcore_input_mode": "transparent_bgra_region",
        }
        return context

    def build_patchcore_pipeline_signature(self, camera: CameraConfig) -> str:
        payload = json.dumps(
            self.build_patchcore_pipeline_context(camera),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def build_region_patchcore_pipeline_signature(
        self,
        camera: CameraConfig,
        region: RegionConfig,
    ) -> str:
        payload = json.dumps(
            self.build_region_patchcore_pipeline_context(camera, region),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def resolve_patchcore_config(
        self,
        camera: CameraConfig,
        region: RegionConfig | None = None,
    ) -> PatchCoreConfig:
        patchcore_config = region.patchcore if region is not None and region.patchcore is not None else camera.patchcore
        if (
            camera.color_insensitive_mode
            and patchcore_config.texture_input.strip().lower() not in {"gray", "lab_l"}
        ):
            patchcore_config = replace(patchcore_config, texture_input="lab_l")
        return patchcore_config

    def load_model_bundle(
        self,
        camera: CameraConfig,
        seat_model_id: str | None,
    ) -> LoadedModelBundle:
        return self._model_cache.load_camera_bundle(camera, seat_model_id)

    def load_region_model_bundle(
        self,
        camera: CameraConfig,
        region: RegionConfig,
        seat_model_id: str | None,
    ) -> LoadedModelBundle:
        return self._model_cache.load_region_bundle(camera, region, seat_model_id)

    def prepare_patchcore_for_predict(self, patchcore: Any) -> None:
        """Attach shared runtime resources just before PatchCore prediction."""
        self._patchcore_predictor.prepare(patchcore)

    def predict_patchcore_batch(
        self,
        items: list[tuple[PatchCoreService, Any, Any, Any]],
    ) -> list[TextureAnomalyResult]:
        """Predict PatchCore results, batching full-backend items with matching features."""
        return self._patchcore_predictor.predict_batch(items)

    def warmup(self, seat_model_id: str | None = None) -> None:
        """Preload active YOLO, PatchCore bundles, and full-backend backbones."""
        context = self.resolve_context(seat_model_id)
        patchcore_items: list[tuple[PatchCoreService, Any, Any, Any]] = []
        for camera in context.cameras:
            pipeline = context.pipelines[camera.camera_id]
            pipeline.detection_service.warmup()
            active_regions = [region for region in camera.regions if region.enabled]
            if active_regions:
                for region in active_regions:
                    model_bundle = self.load_region_model_bundle(
                        camera,
                        region,
                        context.seat_model_id,
                    )
                    patchcore_config = self.resolve_patchcore_config(camera, region)
                    patchcore_items.append(
                        (
                            model_bundle.patchcore,
                            *_dummy_patchcore_sample(patchcore_config),
                        )
                    )
                if camera.color_branch.enabled and not camera.color_insensitive_mode:
                    self.load_model_bundle(camera, context.seat_model_id)
                continue

            model_bundle = self.load_model_bundle(camera, context.seat_model_id)
            patchcore_config = self.resolve_patchcore_config(camera)
            patchcore_items.append(
                (
                    model_bundle.patchcore,
                    *_dummy_patchcore_sample(patchcore_config),
                )
            )

        if patchcore_items:
            self.predict_patchcore_batch(patchcore_items)


class ModelBundleCache:
    """Load and cache PatchCore bundles by model file and pipeline signature."""

    def __init__(self, service: InspectionService) -> None:
        self._service = service
        self._cache: dict[tuple[str, str, str, str, int], LoadedModelBundle] = {}

    def load_camera_bundle(
        self,
        camera: CameraConfig,
        seat_model_id: str | None,
    ) -> LoadedModelBundle:
        pipeline_signature = self._service.build_patchcore_pipeline_signature(camera)
        cache_key = self._cache_key(
            seat_model_id=seat_model_id,
            camera_id=camera.camera_id,
            model_id="__full__",
            model_path=camera.patchcore_model_path,
            pipeline_signature=pipeline_signature,
        )
        bundle = self._cache.get(cache_key)
        if bundle is not None:
            return bundle

        loaded = PatchCoreService.load_bundle(
            camera.patchcore_model_path,
            runtime_config=self._service.resolve_patchcore_config(camera),
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
        self._cache[cache_key] = loaded
        return loaded

    def load_region_bundle(
        self,
        camera: CameraConfig,
        region: RegionConfig,
        seat_model_id: str | None,
    ) -> LoadedModelBundle:
        pipeline_signature = self._service.build_region_patchcore_pipeline_signature(camera, region)
        cache_key = self._cache_key(
            seat_model_id=seat_model_id,
            camera_id=camera.camera_id,
            model_id=region.region_id,
            model_path=region.patchcore_model_path,
            pipeline_signature=pipeline_signature,
        )
        bundle = self._cache.get(cache_key)
        if bundle is not None:
            return bundle

        loaded = PatchCoreService.load_bundle(
            region.patchcore_model_path,
            runtime_config=self._service.resolve_patchcore_config(camera, region),
            expected_pipeline_signature=pipeline_signature,
        )
        self._cache[cache_key] = loaded
        return loaded

    @staticmethod
    def _cache_key(
        *,
        seat_model_id: str | None,
        camera_id: str,
        model_id: str,
        model_path: str,
        pipeline_signature: str,
    ) -> tuple[str, str, str, str, int]:
        model_mtime_ns = Path(model_path).stat().st_mtime_ns
        return (
            seat_model_id or "__default__",
            camera_id,
            model_id,
            pipeline_signature,
            model_mtime_ns,
        )


class PatchCorePredictorPool:
    """Share full-backend feature extractors and batch compatible predictions."""

    def __init__(self) -> None:
        self._feature_extractor_cache: dict[str, _TorchPatchFeatureExtractor] = {}

    def prepare(self, patchcore: Any) -> None:
        if not isinstance(patchcore, PatchCoreService):
            return
        self._attach_shared_feature_extractor(patchcore)

    def predict_batch(
        self,
        items: list[tuple[PatchCoreService, Any, Any, Any]],
    ) -> list[TextureAnomalyResult]:
        results: list[TextureAnomalyResult | None] = [None] * len(items)
        batch_groups: dict[str, list[tuple[int, PatchCoreService, Any, Any, Any]]] = {}
        for index, (patchcore, image, target_mask, ignore_mask) in enumerate(items):
            if not isinstance(patchcore, PatchCoreService):
                results[index] = patchcore.predict(image, target_mask, ignore_mask)
                continue
            self.prepare(patchcore)
            if patchcore.config.backend.strip().lower() != "full":
                results[index] = patchcore.predict(image, target_mask, ignore_mask)
                continue
            cache_key = _feature_extractor_cache_key(patchcore.config)
            batch_groups.setdefault(cache_key, []).append(
                (index, patchcore, image, target_mask, ignore_mask),
            )

        for group in batch_groups.values():
            if len(group) == 1:
                index, patchcore, image, target_mask, ignore_mask = group[0]
                results[index] = patchcore.predict(image, target_mask, ignore_mask)
                continue
            feature_extractor = group[0][1]._get_torch_feature_extractor()
            if feature_extractor is None:
                for index, patchcore, image, target_mask, ignore_mask in group:
                    results[index] = patchcore.predict(image, target_mask, ignore_mask)
                continue
            extracted = feature_extractor.extract_many(
                [
                    (image, target_mask, ignore_mask)
                    for _index, _patchcore, image, target_mask, ignore_mask in group
                ]
            )
            for (index, patchcore, image, target_mask, _ignore_mask), (embeddings, batch) in zip(group, extracted):
                results[index] = patchcore.predict_from_embeddings(
                    image_shape=image.shape[:2],
                    target_mask=target_mask,
                    embeddings=embeddings,
                    batch=batch,
                )

        missing_results = [index for index, result in enumerate(results) if result is None]
        if missing_results:
            raise RuntimeError(f"PatchCore batch prediction missed results: {missing_results}")
        return [result for result in results if result is not None]

    def _attach_shared_feature_extractor(self, patchcore: PatchCoreService) -> None:
        """Share identical full-backend backbones across camera and region models."""
        if patchcore.config.backend.strip().lower() != "full":
            return
        cache_key = _feature_extractor_cache_key(patchcore.config)
        feature_extractor = self._feature_extractor_cache.get(cache_key)
        if feature_extractor is None:
            feature_extractor = _TorchPatchFeatureExtractor(patchcore.config)
            self._feature_extractor_cache[cache_key] = feature_extractor
        patchcore.set_feature_extractor(feature_extractor)


def _feature_extractor_cache_key(config: PatchCoreConfig) -> str:
    """Key only the settings that affect full-backend feature extraction."""
    payload = {
        "backend": config.backend.strip().lower(),
        "image_size": int(config.image_size),
        "texture_input": config.texture_input.strip().lower(),
        "backbone_name": config.backbone_name.strip().lower(),
        "feature_layers": [layer.strip() for layer in config.feature_layers if layer.strip()],
        "backbone_pretrained": bool(config.backbone_pretrained),
        "backbone_weights_path": str(Path(config.backbone_weights_path).resolve())
        if config.backbone_weights_path
        else None,
        "backbone_device": config.backbone_device.strip().lower(),
        "feature_pool_kernel_size": int(config.feature_pool_kernel_size),
        "min_target_coverage": float(config.min_target_coverage),
        "max_ignore_overlap": float(config.max_ignore_overlap),
    }
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _dummy_patchcore_sample(config: PatchCoreConfig) -> tuple[Any, Any, Any]:
    image_size = max(1, int(config.image_size))
    image = np.zeros((image_size, image_size, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    target_mask = np.ones((image_size, image_size), dtype=np.uint8)
    ignore_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    return image, target_mask, ignore_mask
