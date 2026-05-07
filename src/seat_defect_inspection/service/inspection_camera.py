"""工程层单机位失败结果工具。"""

from __future__ import annotations

from media_inputs import infer_source_kind
from seat_defect_core.schemas import CameraInspectionResult

from ..config import CameraConfig


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


def _build_capture_failed_result(
    camera: CameraConfig,
    *,
    reason: str,
    seat_model_id: str | None,
) -> CameraInspectionResult:
    """采图失败时，补齐最基础的机位返回字段。"""
    return _build_reject_result(
        camera_id=camera.camera_id,
        frame_id="",
        source=camera.source,
        source_kind=infer_source_kind(camera.source),
        reason=reason,
        seat_model_id=seat_model_id,
    )


__all__ = [
    "_build_capture_failed_result",
    "_build_reject_result",
]
