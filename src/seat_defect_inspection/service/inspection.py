"""在线检测流程。"""

from __future__ import annotations

import numpy as np
from media_inputs import infer_source_kind

from ..config import CameraConfig
from ..cvops import save_debug_artifacts
from ..fusion import fuse_camera_results, should_early_stop_on_ng
from ..patchcore import ColorConsistencyService
from ..reporting import export_inspection_report
from ..schemas import CameraInspectionResult, FramePacket, InspectionResult
from ..util import select_patchcore_input
from .core import InspectionService, _CameraPipeline


def run_inspection(
    service: InspectionService,
    part_id: str | None = None,
    *,
    seat_model_id: str | None = None,
) -> InspectionResult:
    """抓取各机位图像，执行检测并输出融合结果。"""
    context = service._resolve_context(seat_model_id)
    resolved_part_id = part_id or service.config.part_id

    if not context.cameras:
        return _export_result(
            service,
            InspectionResult(
                part_id=resolved_part_id,
                frame_id="",
                timestamp="",
                status="REJECT",
                decision_reason="no_enabled_cameras",
                seat_model_id=context.seat_model_id,
                camera_results=[],
            ),
        )

    frame_id = ""
    timestamp = ""
    camera_results: list[CameraInspectionResult] = []
    total_camera_count = len(context.cameras)

    for camera in context.cameras:
        try:
            frame_packet = service.acquisition.capture(
                camera.camera_id,
                camera.source,
                resolved_part_id,
            )
        except Exception as exc:
            camera_results.append(
                _build_reject_result(
                    camera_id=camera.camera_id,
                    frame_id="",
                    source=camera.source,
                    source_kind=infer_source_kind(camera.source),
                    reason=f"capture_failed:{exc}",
                    seat_model_id=context.seat_model_id,
                )
            )
            if should_early_stop_on_ng(
                camera_results=camera_results,
                total_camera_count=total_camera_count,
                fusion_config=service.config.fusion,
            ):
                return _export_result(
                    service,
                    _build_early_stop_result(
                        part_id=resolved_part_id,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        seat_model_id=context.seat_model_id,
                        camera_results=camera_results,
                    ),
                )
            continue

        if not frame_id:
            frame_id = frame_packet.frame_id
            timestamp = frame_packet.timestamp

        try:
            camera_results.append(
                _inspect_one_camera(
                    service,
                    frame_packet,
                    camera,
                    context.pipelines[camera.camera_id],
                    context.seat_model_id,
                )
            )
        except Exception as exc:
            camera_results.append(
                _build_reject_result(
                    camera_id=frame_packet.camera_id,
                    frame_id=frame_packet.frame_id,
                    source=frame_packet.source,
                    source_kind=frame_packet.source_kind,
                    reason=f"pipeline_failed:{exc}",
                    seat_model_id=context.seat_model_id,
                )
            )

        if should_early_stop_on_ng(
            camera_results=camera_results,
            total_camera_count=total_camera_count,
            fusion_config=service.config.fusion,
        ):
            return _export_result(
                service,
                _build_early_stop_result(
                    part_id=resolved_part_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    seat_model_id=context.seat_model_id,
                    camera_results=camera_results,
                ),
            )

    fused = fuse_camera_results(
        part_id=resolved_part_id,
        frame_id=frame_id,
        timestamp=timestamp,
        camera_results=camera_results,
        fusion_config=service.config.fusion,
    )
    fused.seat_model_id = context.seat_model_id
    return _export_result(service, fused)


def _inspect_one_camera(
    service: InspectionService,
    frame_packet: FramePacket,
    camera: CameraConfig,
    pipeline: _CameraPipeline,
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

    if prepared.rejection_reason is not None or prepared.roi is None:
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
        prepared.roi.valid_mask,
        np.zeros_like(prepared.roi.valid_mask, dtype=np.uint8),
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
        reason = "texture_and_color_anomaly"
    elif texture_result.is_anomaly:
        status = "NG"
        reason = "texture_anomaly"
    elif color_result is not None and color_result.is_anomaly:
        status = "NG"
        reason = "color_anomaly"
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
    service: InspectionService,
    frame_packet: FramePacket,
    prepared,
    seat_model_id: str | None,
    result: CameraInspectionResult,
    texture_result=None,
) -> CameraInspectionResult:
    """把调试产物挂到结果对象后返回。"""
    result.artifact_paths = save_debug_artifacts(
        enabled=service.config.save_debug_artifacts,
        debug_dir=service.config.debug_dir,
        debug_artifact_mode=service.config.debug_artifact_mode,
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


def _build_early_stop_result(
    *,
    part_id: str,
    frame_id: str,
    timestamp: str,
    seat_model_id: str | None,
    camera_results: list[CameraInspectionResult],
) -> InspectionResult:
    """基于当前累计机位结果构造提前终止结果。"""
    return InspectionResult(
        part_id=part_id,
        frame_id=frame_id,
        timestamp=timestamp,
        status="NG",
        decision_reason=_build_early_stop_reason(camera_results),
        seat_model_id=seat_model_id,
        camera_results=camera_results,
    )


def _build_early_stop_reason(camera_results: list[CameraInspectionResult]) -> str:
    """生成 fail-fast 对应的汇总原因。"""
    ng_cameras = [result.camera_id for result in camera_results if result.status == "NG"]
    if not ng_cameras:
        return "early_stop_without_ng"
    return f"early_stop_ng_from_{','.join(ng_cameras)}"


def _export_result(service: InspectionService, result: InspectionResult) -> InspectionResult:
    """统一落盘检测结果，避免不同出口漏写报告。"""
    export_inspection_report(result, service.config.output_json_path)
    return result
