"""Inspection result builders and public response assembly."""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from ..config import CameraConfig, InspectionConfig
from ..reporting import export_inspection_report
from ..types import CameraInspectionResult, InspectionError, InspectionResponse, InspectionResult

_logger = logging.getLogger(__name__)


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
    artifact_paths = collect_artifact_paths(result)
    defect_images = _collect_defect_images(result, artifact_paths)
    return InspectionResponse(
        result=result,
        report_path=config.output_json_path,
        artifact_paths=artifact_paths,
        defect_images=defect_images,
    )


def collect_artifact_paths(result: InspectionResult) -> dict[str, dict[str, str]]:
    """Collect per-camera artifact paths into the public response shape."""
    return {
        camera_result.camera_id: dict(camera_result.artifact_paths)
        for camera_result in result.camera_results
        if camera_result.artifact_paths
    }


def _collect_defect_images(
    result: InspectionResult,
    artifact_paths: dict[str, dict[str, str]],
) -> dict[str, str]:
    """为 NG 机位读取已保存的 overlay 调试产物并编码为 base64。

    仅在 debug_artifacts_enabled 且 overlay 文件存在时生效。
    """
    defect_images: dict[str, str] = {}
    for cam_result in result.camera_results:
        if cam_result.status != "NG":
            continue
        cam_artifacts = artifact_paths.get(cam_result.camera_id, {})
        overlay_path = cam_artifacts.get("overlay")
        if overlay_path is None:
            continue
        try:
            data = Path(overlay_path).read_bytes()
            defect_images[cam_result.camera_id] = base64.b64encode(data).decode("ascii")
        except Exception:
            _logger.warning(
                "无法读取 camera %s 的缺陷标注图: %s",
                cam_result.camera_id,
                overlay_path,
                exc_info=True,
            )
    return defect_images


__all__ = [
    "build_inspection_response",
    "build_missing_frame_result",
    "build_reject_result",
    "collect_artifact_paths",
    "export_result",
]
