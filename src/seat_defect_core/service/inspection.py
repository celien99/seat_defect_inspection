"""Inspect pipeline orchestration for normalized external frames."""

from __future__ import annotations

from collections import defaultdict
from time import perf_counter

from ..fusion import fuse_camera_results
from ..types import CameraInspectionResult, InspectionError, InspectionFrame, InspectionResult
from .core import InspectionService
from .frames import (
    build_frame_map,
    build_frame_packet,
    resolve_run_frame_id,
    resolve_run_timestamp,
    validate_frame_camera_ids,
)
from .inspection_camera import (
    _StageTimer,
    RegionPatchCorePlan,
    finish_region_patchcore_plan,
    inspect_prepared_camera,
)
from .response import (
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
    camera_loop_started_at = perf_counter()
    camera_results_by_index: dict[int, CameraInspectionResult] = {}
    pending_cameras = []
    for index, camera in enumerate(context.cameras):
        external_frame = frame_map.get(camera.camera_id)
        if external_frame is None:
            camera_results_by_index[index] = build_missing_frame_result(
                camera,
                frame_id=run_frame_id,
                seat_model_id=context.seat_model_id,
            )
        elif external_frame.error_reason is not None:
            camera_results_by_index[index] = build_reject_result(
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
                )
            )
        else:
            pending_cameras.append(
                (
                    index,
                    camera,
                    build_frame_packet(
                        external_frame,
                        camera,
                        part_id=resolved_part_id,
                        fallback_frame_id=run_frame_id,
                        fallback_timestamp=run_timestamp,
                    ),
                )
            )

    pending_results = _inspect_pending_cameras(
        service,
        pending_cameras,
        context.pipelines,
        context.seat_model_id,
    )
    camera_results_by_index.update(pending_results)
    camera_results = [
        camera_results_by_index[index]
        for index in range(len(context.cameras))
        if index in camera_results_by_index
    ]

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


def _inspect_pending_cameras(
    service: InspectionService,
    pending_cameras,
    pipelines: dict[str, object],
    seat_model_id: str | None,
) -> dict[int, CameraInspectionResult]:
    if not pending_cameras:
        return {}

    prepared_by_index = {}
    prepared_errors: dict[int, CameraInspectionResult] = {}
    for group in _group_pending_by_detection(pending_cameras, pipelines).values():
        timers = {
            index: _StageTimer()
            for index, _camera, _frame_packet, _pipeline in group
        }
        started_at = perf_counter()
        try:
            detections = group[0][3].detection_service.detect_many(
                [frame_packet.image for _index, _camera, frame_packet, _pipeline in group]
            )
            batch_prepare_ms = _elapsed_ms(started_at)
            for (index, _camera, frame_packet, pipeline), detection in zip(group, detections):
                camera_timer = timers[index]
                try:
                    local_prepare_started_at = perf_counter()
                    prepared = pipeline.prepare_from_detection(frame_packet.image, detection)
                    local_prepare_ms = _elapsed_ms(local_prepare_started_at)
                    camera_timer.record(
                        "prepare",
                        batch_prepare_ms / max(1, len(group)) + local_prepare_ms,
                    )
                    prepared_by_index[index] = (frame_packet, _camera, prepared, camera_timer)
                except Exception as exc:
                    prepared_errors[index] = _pipeline_failed_result(
                        frame_packet,
                        seat_model_id,
                        exc,
                    )
        except Exception:
            for index, camera, frame_packet, pipeline in group:
                camera_timer = timers[index]
                try:
                    prepared = pipeline.prepare_image(frame_packet.image)
                    camera_timer.mark("prepare")
                    prepared_by_index[index] = (frame_packet, camera, prepared, camera_timer)
                except Exception as exc:
                    prepared_errors[index] = _pipeline_failed_result(
                        frame_packet,
                        seat_model_id,
                        exc,
                    )

    ordered_outputs: dict[int, CameraInspectionResult] = dict(prepared_errors)
    plans: list[tuple[int, RegionPatchCorePlan]] = []
    for index, (frame_packet, camera, prepared, camera_timer) in prepared_by_index.items():
        try:
            output = inspect_prepared_camera(
                service,
                frame_packet,
                camera,
                prepared,
                seat_model_id,
                camera_timer,
            )
            if isinstance(output, RegionPatchCorePlan):
                plans.append((index, output))
            else:
                ordered_outputs[index] = output
        except Exception as exc:
            ordered_outputs[index] = _pipeline_failed_result(
                frame_packet,
                seat_model_id,
                exc,
            )

    _finish_region_plans(service, plans, ordered_outputs)
    return ordered_outputs


def _group_pending_by_detection(pending_cameras, pipelines) -> dict[tuple, list[tuple]]:
    groups: dict[tuple, list[tuple]] = defaultdict(list)
    for index, camera, frame_packet in pending_cameras:
        pipeline = pipelines[camera.camera_id]
        detection_config = camera.detection
        key = (
            detection_config.model_path,
            detection_config.device,
            detection_config.confidence,
            detection_config.iou,
            detection_config.target_class,
            detection_config.imgsz,
        )
        groups[key].append((index, camera, frame_packet, pipeline))
    return groups


def _finish_region_plans(
    service: InspectionService,
    plans: list[tuple[int, RegionPatchCorePlan]],
    ordered_outputs: dict[int, CameraInspectionResult],
) -> None:
    if not plans:
        return

    all_items = []
    slices = []
    for index, plan in plans:
        start = len(all_items)
        all_items.extend(plan.patchcore_items)
        end = len(all_items)
        slices.append((index, plan, start, end))

    batch_started_at = perf_counter()
    try:
        texture_results = service.predict_patchcore_batch(all_items)
    except Exception as exc:
        for index, plan in plans:
            ordered_outputs[index] = _pipeline_failed_result(
                plan.frame_packet,
                plan.seat_model_id,
                exc,
            )
        return

    batch_elapsed_ms = _elapsed_ms(batch_started_at)
    per_item_ms = batch_elapsed_ms / max(1, len(texture_results))
    for index, plan, start, end in slices:
        plan_results = texture_results[start:end]
        plan.camera_timer.record("region_patchcore_batch", per_item_ms * len(plan_results))
        try:
            ordered_outputs[index] = finish_region_patchcore_plan(
                service,
                plan,
                plan_results,
                patchcore_elapsed_ms=per_item_ms * len(plan_results),
            )
        except Exception as exc:
            ordered_outputs[index] = _pipeline_failed_result(
                plan.frame_packet,
                plan.seat_model_id,
                exc,
            )


def _pipeline_failed_result(
    frame_packet,
    seat_model_id: str | None,
    exc: Exception,
) -> CameraInspectionResult:
    return build_reject_result(
        camera_id=frame_packet.camera_id,
        frame_id=frame_packet.frame_id,
        source=frame_packet.source,
        source_kind=frame_packet.source_kind,
        reason="pipeline_failed",
        seat_model_id=seat_model_id,
        error=InspectionError(
            code="pipeline_failed",
            message=str(exc),
            stage="camera_pipeline",
        ),
    )


def _error_code_from_reason(reason: str) -> str:
    normalized = reason.split(":", 1)[0].strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    return normalized or "input_error"


__all__ = [
    "inspect_frames",
]
