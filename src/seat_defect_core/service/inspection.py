"""Inspect pipeline orchestration for normalized external frames."""

from __future__ import annotations

from time import perf_counter

from ..fusion import fuse_camera_results, should_early_stop_on_ng
from ..types import CameraInspectionResult, InspectionError, InspectionFrame, InspectionResult
from .core import InspectionService
from .frames import (
    build_frame_map,
    build_frame_packet,
    resolve_run_frame_id,
    resolve_run_timestamp,
    validate_frame_camera_ids,
)
from .inspection_camera import inspect_one_camera
from .response import (
    build_early_stop_reason,
    build_missing_frame_result,
    build_reject_result,
    export_result,
)


def inspect_frames(
    service: InspectionService,
    frames: list[InspectionFrame],
    *,
    part_id: str | None = None,
    seat_model_id: str | None = None,
) -> InspectionResult:
    """Run the clean inspect pipeline against a prepared core service."""
    started_at = perf_counter()
    context = service.resolve_context(seat_model_id)
    context_ms = _elapsed_ms(started_at)
    resolved_part_id = part_id or service.config.part_id
    if not context.cameras:
        result = InspectionResult(
            part_id=resolved_part_id,
            frame_id="",
            timestamp="",
            status="REJECT",
            decision_reason="no_enabled_cameras",
            seat_model_id=context.seat_model_id,
            camera_results=[],
            timings_ms={"context": context_ms},
        )
        _finish_result_timing(result, started_at)
        return export_result(
            service.config,
            result,
        )

    frame_started_at = perf_counter()
    frame_map = build_frame_map(frames)
    validate_frame_camera_ids(frame_map, [camera.camera_id for camera in context.cameras])
    run_frame_id = resolve_run_frame_id(frames)
    run_timestamp = resolve_run_timestamp(frames)
    frames_ms = _elapsed_ms(frame_started_at)
    camera_results: list[CameraInspectionResult] = []
    total_camera_count = len(context.cameras)

    camera_loop_started_at = perf_counter()
    for camera in context.cameras:
        external_frame = frame_map.get(camera.camera_id)
        if external_frame is None:
            camera_results.append(
                build_missing_frame_result(
                    camera,
                    frame_id=run_frame_id,
                    seat_model_id=context.seat_model_id,
                )
            )
        elif external_frame.error_reason is not None:
            camera_results.append(
                build_reject_result(
                    camera_id=camera.camera_id,
                    frame_id=external_frame.frame_id or run_frame_id,
                    source=external_frame.source or f"external://{camera.camera_id}",
                    source_kind=external_frame.source_kind,
                    reason=external_frame.error_reason,
                    seat_model_id=context.seat_model_id,
                    error=InspectionError(
                        code=_error_code_from_reason(external_frame.error_reason),
                        message=external_frame.error_reason,
                        stage="input",
                    ),
                )
            )
        else:
            frame_packet = build_frame_packet(
                external_frame,
                camera,
                part_id=resolved_part_id,
                fallback_frame_id=run_frame_id,
                fallback_timestamp=run_timestamp,
            )
            try:
                camera_results.append(
                    inspect_one_camera(
                        service,
                        frame_packet,
                        camera,
                        context.pipelines[camera.camera_id],
                        context.seat_model_id,
                    )
                )
            except Exception as exc:
                camera_results.append(
                    build_reject_result(
                        camera_id=frame_packet.camera_id,
                        frame_id=frame_packet.frame_id,
                        source=frame_packet.source,
                        source_kind=frame_packet.source_kind,
                        reason="pipeline_failed",
                        seat_model_id=context.seat_model_id,
                        error=InspectionError(
                            code="pipeline_failed",
                            message=str(exc),
                            stage="camera_pipeline",
                        ),
                    )
                )

        if should_early_stop_on_ng(
            camera_results=camera_results,
            total_camera_count=total_camera_count,
            fusion_config=service.config.fusion,
        ):
            result = InspectionResult(
                part_id=resolved_part_id,
                frame_id=run_frame_id,
                timestamp=run_timestamp,
                status="NG",
                decision_reason=build_early_stop_reason(camera_results),
                seat_model_id=context.seat_model_id,
                camera_results=camera_results,
                timings_ms={
                    "context": context_ms,
                    "frames": frames_ms,
                    "cameras": _elapsed_ms(camera_loop_started_at),
                },
            )
            _finish_result_timing(result, started_at)
            return export_result(service.config, result)

    camera_loop_ms = _elapsed_ms(camera_loop_started_at)
    fusion_started_at = perf_counter()
    fused = fuse_camera_results(
        part_id=resolved_part_id,
        frame_id=run_frame_id,
        timestamp=run_timestamp,
        camera_results=camera_results,
        fusion_config=service.config.fusion,
    )
    fusion_ms = _elapsed_ms(fusion_started_at)
    fused.seat_model_id = context.seat_model_id
    fused.timings_ms = {
        "context": context_ms,
        "frames": frames_ms,
        "cameras": camera_loop_ms,
        "fusion": fusion_ms,
    }
    _finish_result_timing(fused, started_at)
    return export_result(service.config, fused)


def _elapsed_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000.0


def _finish_result_timing(result: InspectionResult, started_at: float) -> None:
    result.timings_ms["total"] = _elapsed_ms(started_at)


def _error_code_from_reason(reason: str) -> str:
    normalized = reason.split(":", 1)[0].strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    return normalized or "input_error"


__all__ = [
    "inspect_frames",
]
