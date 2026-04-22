"""在线检测流程。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..fusion import fuse_camera_results, should_early_stop_on_ng
from ..reporting import export_inspection_report
from ..schemas import CameraInspectionResult, InspectionResult
from .inspection_camera import (
    _build_capture_failed_result,
    _build_reject_result,
    _inspect_one_camera,
)

if TYPE_CHECKING:
    from .core import InspectionService


def run_inspection(
    service: "InspectionService",
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
                _build_capture_failed_result(
                    camera,
                    reason=f"capture_failed:{exc}",
                    seat_model_id=context.seat_model_id,
                )
            )
            early_result = _build_exported_early_stop_result(
                service,
                part_id=resolved_part_id,
                frame_id=frame_id,
                timestamp=timestamp,
                seat_model_id=context.seat_model_id,
                camera_results=camera_results,
                total_camera_count=total_camera_count,
            )
            if early_result is not None:
                return early_result
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

        early_result = _build_exported_early_stop_result(
            service,
            part_id=resolved_part_id,
            frame_id=frame_id,
            timestamp=timestamp,
            seat_model_id=context.seat_model_id,
            camera_results=camera_results,
            total_camera_count=total_camera_count,
        )
        if early_result is not None:
            return early_result

    fused = fuse_camera_results(
        part_id=resolved_part_id,
        frame_id=frame_id,
        timestamp=timestamp,
        camera_results=camera_results,
        fusion_config=service.config.fusion,
    )
    fused.seat_model_id = context.seat_model_id
    return _export_result(service, fused)


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


def _build_exported_early_stop_result(
    service: "InspectionService",
    *,
    part_id: str,
    frame_id: str,
    timestamp: str,
    seat_model_id: str | None,
    camera_results: list[CameraInspectionResult],
    total_camera_count: int,
) -> InspectionResult | None:
    """在满足 fail-fast 条件时直接返回已落盘结果。"""
    if not should_early_stop_on_ng(
        camera_results=camera_results,
        total_camera_count=total_camera_count,
        fusion_config=service.config.fusion,
    ):
        return None
    return _export_result(
        service,
        _build_early_stop_result(
            part_id=part_id,
            frame_id=frame_id,
            timestamp=timestamp,
            seat_model_id=seat_model_id,
            camera_results=camera_results,
        ),
    )


def _export_result(service: "InspectionService", result: InspectionResult) -> InspectionResult:
    """统一落盘检测结果，避免不同出口漏写报告。"""
    export_inspection_report(result, service.config.output_json_path)
    return result
