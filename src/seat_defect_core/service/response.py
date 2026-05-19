"""Inspection result builders and public response assembly."""

from __future__ import annotations

from typing import Any

from ..config import CameraConfig, InspectionConfig
from ..reporting import export_inspection_report
from ..types import CameraInspectionResult, InspectionError, InspectionResponse, InspectionResult


def build_missing_frame_result(
    camera: CameraConfig,
    *,
    frame_id: str,
    seat_model_id: str | None,
) -> CameraInspectionResult:
    """Build a normalized reject result for an enabled camera with no input frame."""
    return build_reject_result(
        camera_id=camera.camera_id,
        frame_id=frame_id,
        source=f"external://{camera.camera_id}",
        source_kind="external_image",
        reason="missing_external_frame",
        seat_model_id=seat_model_id,
        error=InspectionError(
            code="missing_external_frame",
            message="missing_external_frame",
            stage="input",
        ),
    )


def build_reject_result(
    *,
    camera_id: str,
    frame_id: str,
    source: str,
    source_kind: str,
    reason: str,
    seat_model_id: str | None,
    error: InspectionError | None = None,
) -> CameraInspectionResult:
    """Build a normalized single-camera reject result."""
    return CameraInspectionResult(
        camera_id=camera_id,
        frame_id=frame_id,
        source=source,
        source_kind=source_kind,
        status="REJECT",
        reason=reason,
        seat_model_id=seat_model_id,
        error=error,
    )


def export_result(config: InspectionConfig, result: InspectionResult) -> InspectionResult:
    """Write the latest report, then return the original result."""
    export_inspection_report(result, config.output_json_path)
    return result


def build_inspection_response(
    config: InspectionConfig,
    result: InspectionResult,
) -> InspectionResponse:
    """Build the public response wrapper returned by the core API."""
    return InspectionResponse(
        result=result,
        report_path=config.output_json_path,
        artifact_paths=collect_artifact_paths(result),
    )


def collect_camera_images(result: InspectionResult) -> dict[str, Any]:
    """Collect per-camera overlay images from the inspection result."""
    return {
        camera_result.camera_id: camera_result.overlay_image
        for camera_result in result.camera_results
        if camera_result.overlay_image is not None
    }


def collect_artifact_paths(result: InspectionResult) -> dict[str, dict[str, str]]:
    """Collect per-camera artifact paths into the public response shape."""
    return {
        camera_result.camera_id: dict(camera_result.artifact_paths)
        for camera_result in result.camera_results
        if camera_result.artifact_paths
    }


__all__ = [
    "build_inspection_response",
    "build_missing_frame_result",
    "build_reject_result",
    "collect_artifact_paths",
    "collect_camera_images",
    "export_result",
]
