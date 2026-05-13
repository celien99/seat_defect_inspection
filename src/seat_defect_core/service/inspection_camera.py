"""Single-camera core inspection details."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

from ..artifacts import save_debug_artifacts
from ..config import CameraConfig
from ..cvops import split_roi_regions
from ..cvops.regions import RegionRoiSample
from ..patchcore import ColorConsistencyService
from ..types import BoundingBox, CameraInspectionResult, FramePacket, InspectionError, RegionPatchCoreResult
from ..util import select_patchcore_input

if TYPE_CHECKING:
    from .core import CameraPipeline, InspectionService


@dataclass(slots=True)
class RegionPatchCorePlan:
    """Deferred region PatchCore work for cross-camera batching."""

    frame_packet: FramePacket
    camera: CameraConfig
    prepared: Any
    seat_model_id: str | None
    shared_result_fields: dict
    quality_rejected: bool
    camera_timer: "_StageTimer"
    region_results: list[RegionPatchCoreResult]
    patchcore_items: list[tuple[Any, Any, Any, Any]]
    runnable_regions: list[tuple[Any, RegionRoiSample, Any]]


def inspect_one_camera(
    service: "InspectionService",
    frame_packet: FramePacket,
    camera: CameraConfig,
    pipeline: "CameraPipeline",
    seat_model_id: str | None,
) -> CameraInspectionResult:
    """Run one camera through detection, ROI, PatchCore and artifacts."""
    camera_timer = _StageTimer()
    prepared = pipeline.prepare_image(frame_packet.image)
    camera_timer.mark("prepare")
    outcome = inspect_prepared_camera(
        service,
        frame_packet,
        camera,
        prepared,
        seat_model_id,
        camera_timer,
    )
    if isinstance(outcome, RegionPatchCorePlan):
        texture_results = service.predict_patchcore_batch(outcome.patchcore_items)
        patchcore_elapsed_ms = camera_timer.mark("region_patchcore_batch")
        return finish_region_patchcore_plan(
            service,
            outcome,
            texture_results,
            patchcore_elapsed_ms=patchcore_elapsed_ms,
        )
    return outcome


def inspect_prepared_camera(
    service: "InspectionService",
    frame_packet: FramePacket,
    camera: CameraConfig,
    prepared,
    seat_model_id: str | None,
    camera_timer: "_StageTimer",
) -> CameraInspectionResult | RegionPatchCorePlan:
    """Finish one camera after prepare, optionally deferring region PatchCore."""
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
            error=_error_from_reason(
                prepared.rejection_reason or "camera_prepare_failed",
                stage="prepare",
            ),
            **shared_result_fields,
        )
        return _finish_camera_result(
            service,
            frame_packet,
            prepared,
            seat_model_id,
            result,
            camera_timer,
        )

    active_regions = [region for region in camera.regions if region.enabled]
    if active_regions:
        return build_region_patchcore_plan(
            service,
            frame_packet,
            camera,
            prepared,
            seat_model_id,
            shared_result_fields,
            quality_rejected,
            camera_timer,
        )

    model_bundle = service.load_model_bundle(camera, seat_model_id)
    service.prepare_patchcore_for_predict(model_bundle.patchcore)
    texture_input = select_patchcore_input(prepared.roi)
    texture_result = model_bundle.patchcore.predict(
        texture_input,
        prepared.roi.target_mask,
        prepared.roi.ignore_mask,
    )
    camera_timer.mark("patchcore")
    if texture_result.valid_patch_ratio < camera.patchcore.min_valid_patch_ratio:
        result = CameraInspectionResult(
            status="REJECT",
            reason="low_valid_patch_ratio",
            texture_result=texture_result,
            crop_box=prepared.roi.crop_box,
            error=_error_from_reason("low_valid_patch_ratio", stage="patchcore"),
            **shared_result_fields,
        )
        return _finish_camera_result(
            service,
            frame_packet,
            prepared,
            seat_model_id,
            result,
            camera_timer,
            texture_result,
        )

    color_result = _predict_color_branch(camera, model_bundle, prepared)
    camera_timer.mark("color")

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
    return _finish_camera_result(
        service,
        frame_packet,
        prepared,
        seat_model_id,
        result,
        camera_timer,
        texture_result,
    )


def build_region_patchcore_plan(
    service: "InspectionService",
    frame_packet: FramePacket,
    camera: CameraConfig,
    prepared,
    seat_model_id: str | None,
    shared_result_fields: dict,
    quality_rejected: bool,
    camera_timer: "_StageTimer",
) -> RegionPatchCorePlan:
    region_samples: dict[str, RegionRoiSample] = {
        sample.region_id: sample
        for sample in split_roi_regions(prepared.roi, camera.regions)
    }
    camera_timer.mark("split_regions")
    region_results: list[RegionPatchCoreResult] = []
    patchcore_items = []
    runnable_regions = []
    for region in camera.regions:
        if not region.enabled:
            continue
        region_sample = region_samples.get(region.region_id)
        if region_sample is None:
            region_results.append(
                RegionPatchCoreResult(
                    region_id=region.region_id,
                    status="REJECT",
                    reason="region_empty",
                    box=_region_config_box_to_roi_box(region.box, prepared.roi.aligned_roi_image.shape[:2]),
                    patchcore_model_path=region.patchcore_model_path,
                    timings_ms={},
                    error=_error_from_reason("region_empty", stage="region_prepare"),
                )
            )
            continue

        model_bundle = service.load_region_model_bundle(camera, region, seat_model_id)
        patchcore_config = service.resolve_patchcore_config(camera, region)
        patchcore_items.append(
            (
                model_bundle.patchcore,
                region_sample.image,
                region_sample.target_mask,
                region_sample.ignore_mask,
            )
        )
        runnable_regions.append((region, region_sample, patchcore_config))

    return RegionPatchCorePlan(
        frame_packet=frame_packet,
        camera=camera,
        prepared=prepared,
        seat_model_id=seat_model_id,
        shared_result_fields=shared_result_fields,
        quality_rejected=quality_rejected,
        camera_timer=camera_timer,
        region_results=region_results,
        patchcore_items=patchcore_items,
        runnable_regions=runnable_regions,
    )


def finish_region_patchcore_plan(
    service: "InspectionService",
    plan: RegionPatchCorePlan,
    texture_results,
    *,
    patchcore_elapsed_ms: float,
) -> CameraInspectionResult:
    region_results = list(plan.region_results)
    per_region_patchcore_ms = (
        patchcore_elapsed_ms / len(texture_results)
        if texture_results
        else 0.0
    )
    for (region, region_sample, patchcore_config), texture_result in zip(plan.runnable_regions, texture_results):
        if texture_result.valid_patch_ratio < patchcore_config.min_valid_patch_ratio:
            status = "REJECT"
            reason = "low_valid_patch_ratio"
            error = _error_from_reason(reason, stage="patchcore")
        elif texture_result.is_anomaly:
            status = "NG"
            reason = (
                "texture_anomaly_quality_override"
                if plan.quality_rejected
                else "texture_anomaly"
            )
            error = None
        else:
            status = "OK"
            reason = "all_checks_passed"
            error = None
        region_results.append(
            RegionPatchCoreResult(
                region_id=region.region_id,
                status=status,
                reason=reason,
                box=region_sample.box,
                texture_result=texture_result,
                patchcore_model_path=region.patchcore_model_path,
                timings_ms={"patchcore": per_region_patchcore_ms},
                error=error,
                sample=region_sample,
            )
        )

    if not region_results:
        result = CameraInspectionResult(
            status="REJECT",
            reason="no_enabled_regions",
            crop_box=plan.prepared.roi.crop_box,
            error=_error_from_reason("no_enabled_regions", stage="region_prepare"),
            **plan.shared_result_fields,
        )
        return _finish_camera_result(
            service,
            plan.frame_packet,
            plan.prepared,
            plan.seat_model_id,
            result,
            plan.camera_timer,
        )

    color_model_bundle = None
    if plan.camera.color_branch.enabled and not plan.camera.color_insensitive_mode:
        color_model_bundle = service.load_model_bundle(plan.camera, plan.seat_model_id)
    color_result = _predict_color_branch(plan.camera, color_model_bundle, plan.prepared)
    plan.camera_timer.mark("color")

    status, reason = _merge_region_status(
        region_results,
        color_result,
        plan.quality_rejected,
        plan.prepared,
    )
    result = CameraInspectionResult(
        status=status,
        reason=reason,
        region_results=region_results,
        color_result=color_result,
        crop_box=plan.prepared.roi.crop_box,
        **plan.shared_result_fields,
    )
    if status == "REJECT":
        result.error = _error_from_reason(reason, stage="region_merge")
    result = _finish_camera_result(
        service,
        plan.frame_packet,
        plan.prepared,
        plan.seat_model_id,
        result,
        plan.camera_timer,
        region_results=region_results,
    )
    return result


class _StageTimer:
    """Small monotonic stage timer for one camera."""

    def __init__(self) -> None:
        self._started_at = perf_counter()
        self._last_at = self._started_at
        self.timings_ms: dict[str, float] = {}

    def mark(self, name: str) -> float:
        now = perf_counter()
        elapsed_ms = (now - self._last_at) * 1000.0
        self.timings_ms[name] = elapsed_ms
        self._last_at = now
        return elapsed_ms

    def record(self, name: str, elapsed_ms: float) -> None:
        self.timings_ms[name] = elapsed_ms
        self._last_at = perf_counter()

    def finish(self) -> dict[str, float]:
        total_ms = (perf_counter() - self._started_at) * 1000.0
        self.timings_ms["total"] = total_ms
        return dict(self.timings_ms)


def _merge_region_status(
    region_results: list[RegionPatchCoreResult],
    color_result,
    quality_rejected: bool,
    prepared,
) -> tuple[str, str]:
    ng_regions = [item for item in region_results if item.status == "NG"]
    reject_regions = [item for item in region_results if item.status == "REJECT"]
    if ng_regions and color_result is not None and color_result.is_anomaly:
        return (
            "NG",
            "region_texture_and_color_anomaly_quality_override"
            if quality_rejected
            else "region_texture_and_color_anomaly",
        )
    if ng_regions:
        region_ids = ",".join(item.region_id for item in ng_regions)
        prefix = (
            "region_texture_anomaly_quality_override"
            if quality_rejected
            else "region_texture_anomaly"
        )
        return "NG", f"{prefix}:{region_ids}"
    if color_result is not None and color_result.is_anomaly:
        return (
            "NG",
            "color_anomaly_quality_override" if quality_rejected else "color_anomaly",
        )
    if reject_regions:
        return "REJECT", f"region_reject:{reject_regions[0].region_id}:{reject_regions[0].reason}"
    if quality_rejected:
        return "REJECT", prepared.rejection_reason or "quality_reject"
    return "OK", "all_regions_passed"


def _predict_color_branch(
    camera: CameraConfig,
    model_bundle,
    prepared,
):
    if (
        model_bundle is None
        or not camera.color_branch.enabled
        or camera.color_insensitive_mode
        or model_bundle.color_profile is None
    ):
        return None
    color_service = ColorConsistencyService(
        camera.color_branch,
        profile=model_bundle.color_profile,
    )
    return color_service.predict(
        prepared.roi.aligned_roi_image,
        prepared.roi.valid_mask,
    )


def _region_config_box_to_roi_box(
    box: list[float],
    roi_shape: tuple[int, int],
) -> BoundingBox:
    height, width = roi_shape
    return BoundingBox(
        x1=float(round(box[0] * width)),
        y1=float(round(box[1] * height)),
        x2=float(round(box[2] * width)),
        y2=float(round(box[3] * height)),
    )


def _attach_debug_artifacts(
    service: "InspectionService",
    frame_packet: FramePacket,
    prepared,
    seat_model_id: str | None,
    result: CameraInspectionResult,
    texture_result=None,
    region_results=None,
) -> CameraInspectionResult:
    if not getattr(service.config, "debug_artifacts_enabled", True):
        result.artifact_paths = {}
        return result
    result.artifact_paths = save_debug_artifacts(
        debug_dir=service.config.debug_dir,
        artifact_names=getattr(service.config, "debug_artifact_names", None),
        frame_packet=frame_packet,
        prepared=prepared,
        texture_result=texture_result,
        region_results=region_results,
        seat_model_id=seat_model_id,
    )
    return result


def _finish_camera_result(
    service: "InspectionService",
    frame_packet: FramePacket,
    prepared,
    seat_model_id: str | None,
    result: CameraInspectionResult,
    timer: _StageTimer,
    texture_result=None,
    region_results=None,
) -> CameraInspectionResult:
    before_artifacts = perf_counter()
    result = _attach_debug_artifacts(
        service,
        frame_packet,
        prepared,
        seat_model_id,
        result,
        texture_result=texture_result,
        region_results=region_results,
    )
    result.timings_ms = timer.finish()
    result.timings_ms["debug_artifacts"] = (perf_counter() - before_artifacts) * 1000.0
    return result


def _error_from_reason(reason: str, *, stage: str) -> InspectionError:
    code = _normalize_error_code(reason)
    return InspectionError(code=code, message=reason, stage=stage)


def _normalize_error_code(reason: str) -> str:
    normalized = reason.split(":", 1)[0].strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    return normalized or "unknown_error"

