"""主流程共享上下文与服务骨架。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..acquisition import AcquisitionService
from ..config import CameraConfig, InspectionConfig
from ..cvops import ImageQualityGuard, RoiRefineEngine
from ..patchcore import LoadedModelBundle, PatchCoreService
from ..preprocess import PreprocessEngine
from ..schemas import DetectionResult, ImageQualityDecision, RoiRefineResult
from ..yolo import DetectionService


@dataclass(slots=True)
class PreparedCameraSample:
    """单机位共享中间结果。"""

    quality: ImageQualityDecision | None
    preprocessed_image: Any | None = None
    detection: DetectionResult | None = None
    roi: RoiRefineResult | None = None
    rejection_reason: str | None = None


@dataclass(slots=True)
class _ResolvedInspectionContext:
    """运行时解析出的型号上下文。"""

    seat_model_id: str | None
    cameras: list[CameraConfig]
    pipelines: dict[str, "_CameraPipeline"]


class _CameraPipeline:
    """单机位的预处理、检测与 ROI 精修链。"""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.quality_guard = ImageQualityGuard(config.quality)  # ROI 质量门控
        self.preprocess_engine = PreprocessEngine(config.preprocess)  # OpenCV 预处理
        self.detection_service = DetectionService(config.detection)  # YOLO 检测
        self.roi_refine_engine = RoiRefineEngine(config.roi)  # ROI 精修

    def prepare_image(self, image: Any) -> PreparedCameraSample:
        """完成预处理、检测、ROI 精修和 ROI 质量检查。"""
        # 训练和推理共用同一条图像链路，避免线上线下输入分布漂移。
        preprocessed = self.preprocess_engine.process(image)
        detection = self.detection_service.detect(preprocessed)
        if detection.target is None:
            return PreparedCameraSample(
                quality=None,
                preprocessed_image=preprocessed,
                detection=detection,
                rejection_reason="target_not_found",
            )

        roi = self.roi_refine_engine.refine(preprocessed, detection)
        quality = self.quality_guard.evaluate(
            roi.aligned_roi_image,
            valid_mask=roi.valid_mask,
        )
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
    """缺陷检测总编排层。"""

    def __init__(self, config: InspectionConfig) -> None:
        self.config = config
        self.acquisition = AcquisitionService(config.capture_retries)
        # 同一型号下的机位流程对象复用，避免重复构造 YOLO / ROI / 质量门控。
        self._pipeline_cache: dict[str, dict[str, _CameraPipeline]] = {}
        # 模型按“型号 + 机位”缓存，避免重复读盘。
        self._model_cache: dict[tuple[str, str], LoadedModelBundle] = {}

    def train_patchcore_models(self, seat_model_id: str | None = None) -> list[dict[str, Any]]:
        """按机位训练 PatchCore 模型。"""
        from .training import train_patchcore_models as _train_patchcore_models

        return _train_patchcore_models(self, seat_model_id=seat_model_id)

    def capture(
        self,
        part_id: str | None = None,
        *,
        output_dir: str | None = None,
        seat_model_id: str | None = None,
        save_to_train_good_dir: bool = False,
        count: int = 1,
        interval_ms: int = 0,
    ):
        """采图并落盘。"""
        from .capture import capture_samples as _capture_samples

        return _capture_samples(
            self,
            part_id=part_id,
            output_dir=output_dir,
            seat_model_id=seat_model_id,
            save_to_train_good_dir=save_to_train_good_dir,
            count=count,
            interval_ms=interval_ms,
        )

    def run_inspection(
        self,
        part_id: str | None = None,
        *,
        seat_model_id: str | None = None,
    ):
        """执行一次完整检测。"""
        from .inspection import run_inspection as _run_inspection

        return _run_inspection(
            self,
            part_id=part_id,
            seat_model_id=seat_model_id,
        )

    def _resolve_training_scope(self, seat_model_id: str | None) -> list[str | None]:
        """解析本次训练要覆盖的型号范围。"""
        if not self.config.seat_models:
            return [seat_model_id or self.config.default_seat_model_id]
        if seat_model_id is not None:
            return [seat_model_id]
        return [item.seat_model_id for item in self.config.seat_models]

    def _resolve_context(self, seat_model_id: str | None) -> _ResolvedInspectionContext:
        """解析当前应使用的型号、启用机位与机位流程缓存。"""
        resolved_seat_model_id, cameras = self._resolve_active_cameras(seat_model_id)
        cache_key = resolved_seat_model_id or "__default__"
        pipelines = self._pipeline_cache.get(cache_key)
        if pipelines is None:
            pipelines = {
                camera.camera_id: _CameraPipeline(camera)
                for camera in cameras
            }
            self._pipeline_cache[cache_key] = pipelines
        return _ResolvedInspectionContext(
            seat_model_id=resolved_seat_model_id,
            cameras=cameras,
            pipelines=pipelines,
        )

    def _resolve_active_cameras(self, seat_model_id: str | None) -> tuple[str | None, list[CameraConfig]]:
        """根据单型号或多型号配置，解析当前启用的机位列表。"""
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

    def _build_patchcore_service(self, camera: CameraConfig) -> PatchCoreService:
        """根据机位配置创建 PatchCore 服务。"""
        patchcore_config = camera.patchcore
        if (
            camera.color_insensitive_mode
            and patchcore_config.texture_input.strip().lower() not in {"gray", "lab_l"}
        ):
            # 颜色不敏感模式下，把纹理输入收敛到亮度主导模式，避免颜色抖动放大。
            patchcore_config = replace(patchcore_config, texture_input="lab_l")
        return PatchCoreService(patchcore_config)

    def _load_model_bundle(
        self,
        camera: CameraConfig,
        seat_model_id: str | None,
    ) -> LoadedModelBundle:
        """从缓存或磁盘加载当前型号/机位对应的模型包。"""
        cache_key = (seat_model_id or "__default__", camera.camera_id)
        bundle = self._model_cache.get(cache_key)
        if bundle is not None:
            return bundle

        loaded = PatchCoreService.load_bundle(camera.patchcore_model_path)
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
