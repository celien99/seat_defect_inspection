"""座椅缺陷检测项目主服务。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from media_inputs import infer_source_kind

from .acquisition import AcquisitionService
from .color_branch import ColorConsistencyService
from .config import CameraConfig, InspectionConfig
from .detection import DetectionService
from .fusion import fuse_camera_results
from .patchcore import LoadedModelBundle, PatchCoreService, list_images
from .preprocess import PreprocessEngine
from .quality import ImageQualityGuard
from .reporting import export_capture_manifest, export_inspection_report
from .roi import RoiRefineEngine
from .schemas import (
    CameraInspectionResult,
    CaptureRecord,
    CaptureSummary,
    DetectionResult,
    FramePacket,
    ImageQualityDecision,
    InspectionResult,
    RoiRefineResult,
)


@dataclass(slots=True)
class PreparedCameraSample:
    """单机位共享中间结果。"""

    quality: ImageQualityDecision
    preprocessed_image: Any | None = None
    detection: DetectionResult | None = None
    roi: RoiRefineResult | None = None
    rejection_reason: str | None = None


class _CameraPipeline:
    """单机位预处理、检测与 ROI 精修流程。"""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.quality_guard = ImageQualityGuard(config.quality)
        self.preprocess_engine = PreprocessEngine(config.preprocess)
        self.detection_service = DetectionService(config.detection)
        self.roi_refine_engine = RoiRefineEngine(config.roi)

    def prepare_image(self, image: Any) -> PreparedCameraSample:
        """完成质量检查、预处理、检测和 ROI 精修。"""
        quality = self.quality_guard.evaluate(image)
        if not quality.accepted:
            return PreparedCameraSample(
                quality=quality,
                rejection_reason=f"quality_{quality.reason}",
            )

        preprocessed = self.preprocess_engine.process(image)
        detection = self.detection_service.detect(preprocessed)
        if detection.target is None:
            return PreparedCameraSample(
                quality=quality,
                preprocessed_image=preprocessed,
                detection=detection,
                rejection_reason="target_not_found",
            )

        roi = self.roi_refine_engine.refine(preprocessed, detection)
        return PreparedCameraSample(
            quality=quality,
            preprocessed_image=preprocessed,
            detection=detection,
            roi=roi,
            rejection_reason=None,
        )


class InspectionService:
    """缺陷检测主服务，负责采图、训练和推理。"""

    def __init__(self, config: InspectionConfig) -> None:
        self.config = config
        self.acquisition = AcquisitionService(config.capture_retries)
        self._pipelines = {
            camera.camera_id: _CameraPipeline(camera)
            for camera in config.cameras
            if camera.enabled
        }
        self._model_cache: dict[str, LoadedModelBundle] = {}

    def train_patchcore_models(self) -> list[dict[str, Any]]:
        """按机位训练 PatchCore 模型。"""
        summaries: list[dict[str, Any]] = []
        for camera in [item for item in self.config.cameras if item.enabled]:
            if not camera.train_good_dir:
                raise ValueError(f"机位 `{camera.camera_id}` 缺少 `train_good_dir` 配置")

            train_dir = Path(camera.train_good_dir)
            image_paths = list_images(train_dir)
            if not image_paths:
                raise FileNotFoundError(f"训练目录中没有图像：{train_dir}")

            pipeline = self._pipelines[camera.camera_id]
            patchcore_samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            color_samples: list[tuple[np.ndarray, np.ndarray]] = []
            skipped_images: list[str] = []

            for image_path in image_paths:
                image = cv2.imread(str(image_path))
                if image is None:
                    skipped_images.append(str(image_path))
                    continue

                prepared = pipeline.prepare_image(image)
                if prepared.rejection_reason is not None or prepared.roi is None:
                    skipped_images.append(str(image_path))
                    continue

                patchcore_samples.append(
                    (
                        prepared.roi.aligned_roi_image,
                        prepared.roi.target_mask,
                        prepared.roi.ignore_mask,
                    )
                )
                color_samples.append(
                    (
                        prepared.roi.aligned_roi_image,
                        prepared.roi.valid_mask,
                    )
                )

            patchcore = PatchCoreService(camera.patchcore)
            patchcore_summary = patchcore.fit(patchcore_samples)

            color_profile = None
            color_summary: dict[str, Any] | None = None
            if camera.color_branch.enabled:
                color_service = ColorConsistencyService(camera.color_branch)
                color_summary = color_service.fit(color_samples)
                color_profile = color_service.profile

            patchcore.save(camera.patchcore_model_path, color_profile=color_profile)
            summary = {
                "camera_id": camera.camera_id,
                "model_path": camera.patchcore_model_path,
                "patchcore": patchcore_summary,
                "color_branch": color_summary,
                "skipped_image_count": len(skipped_images),
            }
            summary_path = Path(camera.patchcore_model_path).with_suffix(".summary.json")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            summaries.append(summary)
        return summaries

    def capture(
        self,
        part_id: str | None = None,
        *,
        output_dir: str | None = None,
        save_to_train_good_dir: bool = False,
    ) -> CaptureSummary:
        """每个启用机位抓取一帧并落盘。"""
        resolved_part_id = part_id or self.config.part_id
        active_cameras = [camera for camera in self.config.cameras if camera.enabled]
        run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        capture_root = Path(output_dir or self.config.capture_dir) / resolved_part_id / run_id
        capture_root.mkdir(parents=True, exist_ok=True)

        records: list[CaptureRecord] = []
        for camera in active_cameras:
            try:
                frame_packet = self.acquisition.capture(
                    camera.camera_id,
                    camera.source,
                    resolved_part_id,
                )
                output_path = self._save_captured_frame(capture_root, frame_packet)
                train_good_path = None
                if save_to_train_good_dir:
                    train_good_path = self._save_train_good_frame(camera, frame_packet)
                records.append(
                    CaptureRecord(
                        camera_id=frame_packet.camera_id,
                        frame_id=frame_packet.frame_id,
                        part_id=frame_packet.part_id,
                        source=frame_packet.source,
                        source_kind=frame_packet.source_kind,
                        timestamp=frame_packet.timestamp,
                        status="OK",
                        output_path=output_path,
                        train_good_path=train_good_path,
                    )
                )
            except Exception as exc:
                records.append(
                    CaptureRecord(
                        camera_id=camera.camera_id,
                        frame_id="",
                        part_id=resolved_part_id,
                        source=camera.source,
                        source_kind=infer_source_kind(camera.source),
                        timestamp=datetime.now().astimezone().isoformat(),
                        status="ERROR",
                        reason=str(exc),
                    )
                )

        summary = CaptureSummary(
            part_id=resolved_part_id,
            run_id=run_id,
            output_dir=str(capture_root),
            manifest_path=str(capture_root / "manifest.json"),
            records=records,
        )
        export_capture_manifest(summary)
        return summary

    def run_inspection(self, part_id: str | None = None) -> InspectionResult:
        """抓取各机位图像，执行检测并输出融合结果。"""
        resolved_part_id = part_id or self.config.part_id
        active_cameras = [camera for camera in self.config.cameras if camera.enabled]
        if not active_cameras:
            result = InspectionResult(
                part_id=resolved_part_id,
                frame_id="",
                timestamp="",
                status="REJECT",
                decision_reason="no_enabled_cameras",
                camera_results=[],
            )
            export_inspection_report(result, self.config.output_json_path)
            return result

        frame_id = ""
        timestamp = ""
        camera_results: list[CameraInspectionResult] = []
        for camera in active_cameras:
            try:
                frame_packet = self.acquisition.capture(
                    camera.camera_id,
                    camera.source,
                    resolved_part_id,
                )
            except Exception as exc:
                camera_results.append(
                    CameraInspectionResult(
                        camera_id=camera.camera_id,
                        frame_id=frame_id,
                        source=camera.source,
                        source_kind=infer_source_kind(camera.source),
                        status="REJECT",
                        reason=f"capture_failed:{exc}",
                    )
                )
                continue

            if not frame_id:
                frame_id = frame_packet.frame_id
                timestamp = frame_packet.timestamp
            camera_results.append(self._inspect_one_camera(frame_packet))

        fused = fuse_camera_results(
            part_id=resolved_part_id,
            frame_id=frame_id,
            timestamp=timestamp,
            camera_results=camera_results,
            fusion_config=self.config.fusion,
        )
        export_inspection_report(fused, self.config.output_json_path)
        return fused

    def _inspect_one_camera(self, frame_packet: FramePacket) -> CameraInspectionResult:
        camera = next(
            item
            for item in self.config.cameras
            if item.enabled and item.camera_id == frame_packet.camera_id
        )
        pipeline = self._pipelines[camera.camera_id]
        prepared = pipeline.prepare_image(frame_packet.image)
        if prepared.rejection_reason is not None or prepared.roi is None:
            result = CameraInspectionResult(
                camera_id=frame_packet.camera_id,
                frame_id=frame_packet.frame_id,
                source=frame_packet.source,
                source_kind=frame_packet.source_kind,
                status="REJECT",
                reason=prepared.rejection_reason or "camera_prepare_failed",
                quality=prepared.quality,
                detection=prepared.detection,
                crop_box=(prepared.roi.crop_box if prepared.roi is not None else None),
            )
            result.artifact_paths = self._save_artifacts(frame_packet, prepared, None)
            return result

        model_bundle = self._load_model_bundle(camera)
        texture_result = model_bundle.patchcore.predict(
            prepared.roi.aligned_roi_image,
            prepared.roi.target_mask,
            prepared.roi.ignore_mask,
        )
        if texture_result.valid_patch_ratio < camera.patchcore.min_valid_patch_ratio:
            result = CameraInspectionResult(
                camera_id=frame_packet.camera_id,
                frame_id=frame_packet.frame_id,
                source=frame_packet.source,
                source_kind=frame_packet.source_kind,
                status="REJECT",
                reason="low_valid_patch_ratio",
                quality=prepared.quality,
                detection=prepared.detection,
                texture_result=texture_result,
                crop_box=prepared.roi.crop_box,
            )
            result.artifact_paths = self._save_artifacts(frame_packet, prepared, texture_result)
            return result

        color_result = None
        if camera.color_branch.enabled and model_bundle.color_profile is not None:
            color_service = ColorConsistencyService(
                camera.color_branch,
                profile=model_bundle.color_profile,
            )
            color_result = color_service.predict(
                prepared.roi.aligned_roi_image,
                prepared.roi.valid_mask,
            )

        status = "OK"
        reason = "all_checks_passed"
        if texture_result.is_anomaly:
            status = "NG"
            reason = "texture_anomaly"
        if color_result is not None and color_result.is_anomaly:
            status = "NG"
            reason = "color_anomaly" if status == "OK" else "texture_and_color_anomaly"

        result = CameraInspectionResult(
            camera_id=frame_packet.camera_id,
            frame_id=frame_packet.frame_id,
            source=frame_packet.source,
            source_kind=frame_packet.source_kind,
            status=status,
            reason=reason,
            quality=prepared.quality,
            detection=prepared.detection,
            texture_result=texture_result,
            color_result=color_result,
            crop_box=prepared.roi.crop_box,
        )
        result.artifact_paths = self._save_artifacts(frame_packet, prepared, texture_result)
        return result

    def _load_model_bundle(self, camera: CameraConfig) -> LoadedModelBundle:
        bundle = self._model_cache.get(camera.camera_id)
        if bundle is not None:
            return bundle
        loaded = PatchCoreService.load_bundle(camera.patchcore_model_path)
        self._model_cache[camera.camera_id] = loaded
        return loaded

    def _save_captured_frame(self, capture_root: Path, frame_packet: FramePacket) -> str:
        camera_dir = capture_root / frame_packet.camera_id
        camera_dir.mkdir(parents=True, exist_ok=True)
        output_path = camera_dir / f"{frame_packet.frame_id}.png"
        _write_image(output_path, frame_packet.image)
        return str(output_path)

    def _save_train_good_frame(self, camera: CameraConfig, frame_packet: FramePacket) -> str:
        if not camera.train_good_dir:
            raise ValueError(f"机位 `{camera.camera_id}` 未配置 `train_good_dir`")
        train_good_dir = Path(camera.train_good_dir)
        train_good_dir.mkdir(parents=True, exist_ok=True)
        output_path = train_good_dir / f"{frame_packet.part_id}_{frame_packet.camera_id}_{frame_packet.frame_id}.png"
        _write_image(output_path, frame_packet.image)
        return str(output_path)

    def _save_artifacts(
        self,
        frame_packet: FramePacket,
        prepared: PreparedCameraSample,
        texture_result,
    ) -> dict[str, str]:
        if not self.config.save_debug_artifacts:
            return {}

        camera_dir = (
            Path(self.config.debug_dir)
            / frame_packet.part_id
            / frame_packet.camera_id
            / frame_packet.frame_id
        )
        camera_dir.mkdir(parents=True, exist_ok=True)
        artifact_paths: dict[str, str] = {}

        raw_path = camera_dir / "raw.png"
        _write_image(raw_path, frame_packet.image)
        artifact_paths["raw"] = str(raw_path)

        if prepared.preprocessed_image is not None:
            preprocessed_path = camera_dir / "preprocessed.png"
            _write_image(preprocessed_path, prepared.preprocessed_image)
            artifact_paths["preprocessed"] = str(preprocessed_path)

            detection_overlay_path = camera_dir / "detections.png"
            _write_image(
                detection_overlay_path,
                _render_detections(prepared.preprocessed_image, prepared.detection),
            )
            artifact_paths["detections"] = str(detection_overlay_path)

        if prepared.roi is not None:
            roi_path = camera_dir / "roi.png"
            target_mask_path = camera_dir / "target_mask.png"
            ignore_mask_path = camera_dir / "ignore_mask.png"
            valid_mask_path = camera_dir / "valid_mask.png"
            _write_image(roi_path, prepared.roi.aligned_roi_image)
            _write_mask(target_mask_path, prepared.roi.target_mask)
            _write_mask(ignore_mask_path, prepared.roi.ignore_mask)
            _write_mask(valid_mask_path, prepared.roi.valid_mask)
            artifact_paths["roi"] = str(roi_path)
            artifact_paths["target_mask"] = str(target_mask_path)
            artifact_paths["ignore_mask"] = str(ignore_mask_path)
            artifact_paths["valid_mask"] = str(valid_mask_path)

        if texture_result is not None and prepared.roi is not None:
            heatmap_path = camera_dir / "heatmap.png"
            overlay_path = camera_dir / "overlay.png"
            _write_mask(heatmap_path, texture_result.heatmap)
            _write_image(
                overlay_path,
                _overlay_heatmap(prepared.roi.aligned_roi_image, texture_result.heatmap),
            )
            artifact_paths["heatmap"] = str(heatmap_path)
            artifact_paths["overlay"] = str(overlay_path)

        return artifact_paths


def train_patchcore_models(config: InspectionConfig) -> list[dict[str, Any]]:
    """训练全部机位的 PatchCore 模型。"""
    return InspectionService(config).train_patchcore_models()


def capture_samples(
    config: InspectionConfig,
    part_id: str | None = None,
    *,
    output_dir: str | None = None,
    save_to_train_good_dir: bool = False,
) -> CaptureSummary:
    """抓取并落盘全部启用机位的图像。"""
    return InspectionService(config).capture(
        part_id=part_id,
        output_dir=output_dir,
        save_to_train_good_dir=save_to_train_good_dir,
    )


def run_inspection(
    config: InspectionConfig,
    part_id: str | None = None,
) -> InspectionResult:
    """执行一次完整检测。"""
    return InspectionService(config).run_inspection(part_id=part_id)


def _write_image(path: Path, image: Any) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write image: {path}")


def _write_mask(path: Path, mask: np.ndarray) -> None:
    if mask.dtype != np.uint8:
        normalized = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    else:
        normalized = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(path), normalized):
        raise OSError(f"Failed to write mask: {path}")


def _render_detections(image: Any, detection) -> Any:
    if detection is None:
        return image.copy()
    canvas = image.copy()
    if detection.target is not None:
        _draw_box(canvas, detection.target.bounding_box, (0, 255, 0), detection.target.label)
    for item in detection.ignores:
        _draw_box(canvas, item.bounding_box, (0, 0, 255), item.label)
    return canvas


def _draw_box(image: Any, box, color: tuple[int, int, int], label: str) -> None:
    x1 = int(round(box.x1))
    y1 = int(round(box.y1))
    x2 = int(round(box.x2))
    y2 = int(round(box.y2))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def _overlay_heatmap(image: Any, heatmap: np.ndarray) -> Any:
    color_map = cv2.applyColorMap(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.65, color_map, 0.35, 0.0)
