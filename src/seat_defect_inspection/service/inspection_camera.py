"""单机位检测细节。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from media_inputs import infer_source_kind

from ..config import CameraConfig
from ..cvops import save_debug_artifacts
from ..patchcore import ColorConsistencyService
from ..schemas import CameraInspectionResult, FramePacket
from ..util import select_patchcore_input

if TYPE_CHECKING:
    from .core import InspectionService, _CameraPipeline


def _inspect_one_camera(
    service: "InspectionService",
    frame_packet: FramePacket,
    camera: CameraConfig,
    pipeline: "_CameraPipeline",
    seat_model_id: str | None,
) -> CameraInspectionResult:
    """执行单机位完整检测。"""
    prepared = pipeline.prepare_image(frame_packet.image)
    shared_result_fields = {
        "camera_id": frame_packet.camera_id,
        "frame_id": frame_packet.frame_id,
        "source": frame_packet.source,
        "source_kind": frame_packet.source_kind,
        "seat_model_id": seat_model_id,
        "quality": prepared.quality,
        "detection": prepared.detection,
    }

    quality_rejected = (
        prepared.rejection_reason is not None
        and prepared.rejection_reason.startswith("quality_")
    )
    if prepared.roi is None or (prepared.rejection_reason is not None and not quality_rejected):
        result = CameraInspectionResult(
            status="REJECT",
            reason=prepared.rejection_reason or "camera_prepare_failed",
            crop_box=(prepared.roi.crop_box if prepared.roi is not None else None),
            **shared_result_fields,
        )
        return _attach_debug_artifacts(service, frame_packet, prepared, seat_model_id, result)

    model_bundle = service._load_model_bundle(camera, seat_model_id)
    texture_input = select_patchcore_input(prepared.roi)
    texture_result = model_bundle.patchcore.predict(
        texture_input,
        prepared.roi.target_mask,
        prepared.roi.ignore_mask,
    )
    if texture_result.valid_patch_ratio < camera.patchcore.min_valid_patch_ratio:
        result = CameraInspectionResult(
            status="REJECT",
            reason="low_valid_patch_ratio",
            texture_result=texture_result,
            crop_box=prepared.roi.crop_box,
            **shared_result_fields,
        )
        return _attach_debug_artifacts(
            service,
            frame_packet,
            prepared,
            seat_model_id,
            result,
            texture_result,
        )

    color_result = None
    if (
        camera.color_branch.enabled
        and not camera.color_insensitive_mode
        and model_bundle.color_profile is not None
    ):
        color_service = ColorConsistencyService(
            camera.color_branch,
            profile=model_bundle.color_profile,
        )
        color_result = color_service.predict(
            prepared.roi.aligned_roi_image,
            prepared.roi.valid_mask,
        )

    if texture_result.is_anomaly and color_result is not None and color_result.is_anomaly:
        status = "NG"
        reason = (
            "texture_and_color_anomaly_quality_override"
            if quality_rejected
            else "texture_and_color_anomaly"
        )
    elif texture_result.is_anomaly:
        status = "NG"
        reason = "texture_anomaly_quality_override" if quality_rejected else "texture_anomaly"
    elif color_result is not None and color_result.is_anomaly:
        status = "NG"
        reason = "color_anomaly_quality_override" if quality_rejected else "color_anomaly"
    elif quality_rejected:
        status = "REJECT"
        reason = prepared.rejection_reason or "quality_reject"
    else:
        status = "OK"
        reason = "all_checks_passed"

    result = CameraInspectionResult(
        status=status,
        reason=reason,
        texture_result=texture_result,
        color_result=color_result,
        crop_box=prepared.roi.crop_box,
        **shared_result_fields,
    )
    return _attach_debug_artifacts(
        service,
        frame_packet,
        prepared,
        seat_model_id,
        result,
        texture_result,
    )


def _attach_debug_artifacts(
    service: "InspectionService",
    frame_packet: FramePacket,
    prepared,
    seat_model_id: str | None,
    result: CameraInspectionResult,
    texture_result=None,
) -> CameraInspectionResult:
    """把调试产物挂到结果对象后返回。"""
    result.artifact_paths = save_debug_artifacts(
        debug_dir=service.config.debug_dir,
        frame_packet=frame_packet,
        prepared=prepared,
        texture_result=texture_result,
        seat_model_id=seat_model_id,
    )
    return result


def _build_reject_result(
    *,
    camera_id: str,
    frame_id: str,
    source: str,
    source_kind: str,
    reason: str,
    seat_model_id: str | None,
) -> CameraInspectionResult:
    """统一构造单机位 REJECT 结果。"""
    return CameraInspectionResult(
        camera_id=camera_id,
        frame_id=frame_id,
        source=source,
        source_kind=source_kind,
        status="REJECT",
        reason=reason,
        seat_model_id=seat_model_id,
    )


def _build_capture_failed_result(
    camera: CameraConfig,
    *,
    reason: str,
    seat_model_id: str | None,
) -> CameraInspectionResult:
    """采图失败时，补齐最基础的机位返回字段。"""
    return _build_reject_result(
        camera_id=camera.camera_id,
        frame_id="",
        source=camera.source,
        source_kind=infer_source_kind(camera.source),
        reason=reason,
        seat_model_id=seat_model_id,
    )
