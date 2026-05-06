"""在线检测流程。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config import CameraConfig
from ..fusion import fuse_camera_results, should_early_stop_on_ng
from ..reporting import export_inspection_report
from ..schemas import CameraInspectionResult, FramePacket, InspectionResult
from .inspection_camera import (
    _build_capture_failed_result,
    _build_reject_result,
    _inspect_one_camera,
)

if TYPE_CHECKING:
    from .core import InspectionService


@dataclass(slots=True)
class _CaptureOutcome:
    """一次在线检测中某个机位的采图结果。"""

    camera: CameraConfig
    frame_packet: FramePacket | None = None
    error: Exception | None = None


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
    capture_outcomes = _capture_cameras_concurrently(
        service,
        context.cameras,
        resolved_part_id,
    )
    first_frame_packet = next(
        (
            outcome.frame_packet
            for outcome in capture_outcomes
            if outcome.frame_packet is not None
        ),
        None,
    )
    if first_frame_packet is not None:
        frame_id = first_frame_packet.frame_id
        timestamp = first_frame_packet.timestamp

    if _should_inspect_concurrently(service, total_camera_count):
        camera_results = _inspect_cameras_concurrently(
            service,
            capture_outcomes,
            context.seat_model_id,
            context.pipelines,
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

    for outcome in capture_outcomes:
        camera = outcome.camera
        if outcome.error is not None:
            camera_results.append(
                _build_capture_failed_result(
                    camera,
                    reason=f"capture_failed:{outcome.error}",
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

        frame_packet = outcome.frame_packet
        if frame_packet is None:
            camera_results.append(
                _build_capture_failed_result(
                    camera,
                    reason="capture_failed:empty_frame_packet",
                    seat_model_id=context.seat_model_id,
                )
            )
            continue

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


def _should_inspect_concurrently(
    service: "InspectionService",
    total_camera_count: int,
) -> bool:
    """Only parallelize camera inspection when it will not change fail-fast behavior."""
    if total_camera_count <= 1:
        return False
    return not service.config.fusion.early_stop_on_ng


def _capture_cameras_concurrently(
    service: "InspectionService",
    cameras: list[CameraConfig],
    part_id: str,
) -> list[_CaptureOutcome]:
    """并发抓取全部启用机位图像，并按配置中的机位顺序返回结果。"""
    indexed_outcomes: dict[int, _CaptureOutcome] = {}
    max_workers = max(1, len(cameras))
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="seat-inspect-capture",
    ) as executor:
        futures = {
            executor.submit(
                service.acquisition.capture,
                camera.camera_id,
                camera.source,
                part_id,
            ): index
            for index, camera in enumerate(cameras)
        }
        for future in as_completed(futures):
            index = futures[future]
            camera = cameras[index]
            try:
                indexed_outcomes[index] = _CaptureOutcome(
                    camera=camera,
                    frame_packet=future.result(),
                )
            except Exception as exc:
                indexed_outcomes[index] = _CaptureOutcome(
                    camera=camera,
                    error=exc,
                )
    return [indexed_outcomes[index] for index in range(len(cameras))]


def _inspect_cameras_concurrently(
    service: "InspectionService",
    capture_outcomes: list[_CaptureOutcome],
    seat_model_id: str | None,
    pipelines,
) -> list[CameraInspectionResult]:
    """Run heavy per-camera inspection in parallel and preserve config order in output."""
    indexed_results: dict[int, CameraInspectionResult] = {}
    inspect_futures = {}

    max_workers = max(1, len(capture_outcomes))
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="seat-inspect-camera",
    ) as executor:
        for index, outcome in enumerate(capture_outcomes):
            camera = outcome.camera
            if outcome.error is not None:
                indexed_results[index] = _build_capture_failed_result(
                    camera,
                    reason=f"capture_failed:{outcome.error}",
                    seat_model_id=seat_model_id,
                )
                continue

            frame_packet = outcome.frame_packet
            if frame_packet is None:
                indexed_results[index] = _build_capture_failed_result(
                    camera,
                    reason="capture_failed:empty_frame_packet",
                    seat_model_id=seat_model_id,
                )
                continue

            inspect_futures[
                executor.submit(
                    _inspect_one_captured_camera,
                    service,
                    frame_packet,
                    camera,
                    pipelines[camera.camera_id],
                    seat_model_id,
                )
            ] = index

        for future in as_completed(inspect_futures):
            indexed_results[inspect_futures[future]] = future.result()

    return [indexed_results[index] for index in range(len(capture_outcomes))]


def _inspect_one_captured_camera(
    service: "InspectionService",
    frame_packet: FramePacket,
    camera: CameraConfig,
    pipeline,
    seat_model_id: str | None,
) -> CameraInspectionResult:
    """Wrap one camera inspection so concurrent mode still returns normalized results."""
    try:
        return _inspect_one_camera(
            service,
            frame_packet,
            camera,
            pipeline,
            seat_model_id,
        )
    except Exception as exc:
        return _build_reject_result(
            camera_id=frame_packet.camera_id,
            frame_id=frame_packet.frame_id,
            source=frame_packet.source,
            source_kind=frame_packet.source_kind,
            reason=f"pipeline_failed:{exc}",
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
