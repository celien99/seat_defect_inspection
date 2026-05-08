"""Single-camera core inspection details."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import CameraConfig
from ..cvops import save_debug_artifacts, split_roi_regions
from ..patchcore import ColorConsistencyService
from ..types import BoundingBox, CameraInspectionResult, FramePacket, RegionPatchCoreResult
from ..util import select_patchcore_input

if TYPE_CHECKING:
    from .core import CameraPipeline, InspectionService


def inspect_one_camera(
    service: "InspectionService",
    frame_packet: FramePacket,
    camera: CameraConfig,
    pipeline: "CameraPipeline",
    seat_model_id: str | None,
) -> CameraInspectionResult:
    """Run one camera through detection, ROI, PatchCore and artifacts."""
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

    active_regions = [region for region in camera.regions if region.enabled]
    if active_regions:
        return _inspect_region_patchcores(
            service,
            frame_packet,
            camera,
            prepared,
            seat_model_id,
            shared_result_fields,
            quality_rejected,
        )

    model_bundle = service.load_model_bundle(camera, seat_model_id)
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


def _inspect_region_patchcores(
    service: "InspectionService",
    frame_packet: FramePacket,
    camera: CameraConfig,
    prepared,
    seat_model_id: str | None,
    shared_result_fields: dict,
    quality_rejected: bool,
) -> CameraInspectionResult:
    region_samples = {
        sample.region_id: sample
        for sample in split_roi_regions(prepared.roi, camera.regions)
    }
    region_results: list[RegionPatchCoreResult] = []
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
                )
            )
            continue

        model_bundle = service.load_region_model_bundle(camera, region, seat_model_id)
        patchcore_config = service.resolve_patchcore_config(camera, region)
        texture_result = model_bundle.patchcore.predict(
            region_sample.image,
            region_sample.target_mask,
            region_sample.ignore_mask,
        )
        if texture_result.valid_patch_ratio < patchcore_config.min_valid_patch_ratio:
            status = "REJECT"
            reason = "low_valid_patch_ratio"
        elif texture_result.is_anomaly:
            status = "NG"
            reason = "texture_anomaly_quality_override" if quality_rejected else "texture_anomaly"
        else:
            status = "OK"
            reason = "all_checks_passed"
        region_results.append(
            RegionPatchCoreResult(
                region_id=region.region_id,
                status=status,
                reason=reason,
                box=region_sample.box,
                texture_result=texture_result,
                patchcore_model_path=region.patchcore_model_path,
            )
        )

    if not region_results:
        result = CameraInspectionResult(
            status="REJECT",
            reason="no_enabled_regions",
            crop_box=prepared.roi.crop_box,
            **shared_result_fields,
        )
        return _attach_debug_artifacts(service, frame_packet, prepared, seat_model_id, result)

    color_result = None

    status, reason = _merge_region_status(region_results, color_result, quality_rejected, prepared)
    result = CameraInspectionResult(
        status=status,
        reason=reason,
        region_results=region_results,
        color_result=color_result,
        crop_box=prepared.roi.crop_box,
        **shared_result_fields,
    )
    result = _attach_debug_artifacts(
        service,
        frame_packet,
        prepared,
        seat_model_id,
        result,
        region_results=region_results,
    )
    _attach_region_artifact_paths(result)
    return result


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
    result.artifact_paths = save_debug_artifacts(
        debug_dir=service.config.debug_dir,
        frame_packet=frame_packet,
        prepared=prepared,
        texture_result=texture_result,
        region_results=region_results,
        seat_model_id=seat_model_id,
    )
    return result


def _attach_region_artifact_paths(result: CameraInspectionResult) -> None:
    for region_result in result.region_results:
        prefix = f"regions.{region_result.region_id}."
        region_result.artifact_paths = {
            key.removeprefix(prefix): value
            for key, value in result.artifact_paths.items()
            if key.startswith(prefix)
        }
