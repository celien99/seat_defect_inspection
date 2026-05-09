"""Inspection result builders and public response assembly."""

from __future__ import annotations

from pathlib import Path

from ..config import CameraConfig, InspectionConfig
from ..reporting import export_inspection_report, resolve_inspection_archive_path
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


def build_early_stop_reason(camera_results: list[CameraInspectionResult]) -> str:
    """Build the fused reason used when early-stop ends the run."""
    ng_cameras = [result.camera_id for result in camera_results if result.status == "NG"]
    if not ng_cameras:
        return "early_stop_without_ng"
    return f"early_stop_ng_from_{','.join(ng_cameras)}"


def export_result(config: InspectionConfig, result: InspectionResult) -> InspectionResult:
    """Write latest and archived reports, then return the original result."""
    export_inspection_report(result, config.output_json_path)
    return result


def build_inspection_response(
    config: InspectionConfig,
    result: InspectionResult,
) -> InspectionResponse:
    """Build the public response wrapper returned by the core API."""
    report_path = Path(config.output_json_path)
    archive_report_path = resolve_inspection_archive_path(report_path, result)
    return InspectionResponse(
        result=result,
        report_path=str(report_path),
        archive_report_path=str(archive_report_path),
        artifact_paths=collect_artifact_paths(result),
    )


def collect_artifact_paths(result: InspectionResult) -> dict[str, dict[str, str]]:
    """Collect per-camera artifact paths into the public response shape."""
    return {
        camera_result.camera_id: dict(camera_result.artifact_paths)
        for camera_result in result.camera_results
        if camera_result.artifact_paths
    }


__all__ = [
    "build_early_stop_reason",
    "build_inspection_response",
    "build_missing_frame_result",
    "build_reject_result",
    "collect_artifact_paths",
    "export_result",
]
