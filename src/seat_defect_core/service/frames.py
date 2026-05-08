"""External frame normalization helpers for the inspect pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..config import CameraConfig
from ..types import FramePacket, InspectionFrame


def normalize_inspection_frames(
    frames: list[InspectionFrame | dict[str, Any]],
) -> list[InspectionFrame]:
    """Normalize external input frames into core InspectionFrame objects."""
    normalized = [_normalize_frame(frame) for frame in frames]
    build_frame_map(normalized)
    return normalized


def build_frame_map(frames: list[InspectionFrame]) -> dict[str, InspectionFrame]:
    """Index frames by camera_id and reject duplicate camera inputs."""
    frame_map: dict[str, InspectionFrame] = {}
    duplicates: set[str] = set()
    for frame in frames:
        if frame.camera_id in frame_map:
            duplicates.add(frame.camera_id)
        frame_map[frame.camera_id] = frame
    if duplicates:
        duplicated_ids = ", ".join(f"`{camera_id}`" for camera_id in sorted(duplicates))
        raise ValueError(f"frames 中存在重复 camera_id: {duplicated_ids}")
    return frame_map


def validate_frame_camera_ids(
    frame_map: dict[str, InspectionFrame],
    active_camera_ids: list[str],
) -> None:
    """Reject frames for cameras that are not configured or not enabled."""
    active_id_set = set(active_camera_ids)
    unknown_ids = sorted(set(frame_map) - active_id_set)
    if not unknown_ids:
        return
    unknown = ", ".join(f"`{camera_id}`" for camera_id in unknown_ids)
    available = ", ".join(f"`{camera_id}`" for camera_id in active_camera_ids) or "none"
    raise ValueError(f"frames 包含未配置或未启用的 camera_id: {unknown}；可用 camera_id: {available}")


def build_frame_packet(
    frame: InspectionFrame,
    camera: CameraConfig,
    *,
    part_id: str,
    fallback_frame_id: str,
    fallback_timestamp: str,
) -> FramePacket:
    """Build the internal per-camera packet consumed by the camera pipeline."""
    frame_id = frame.frame_id or fallback_frame_id
    timestamp = frame.timestamp or fallback_timestamp
    return FramePacket(
        camera_id=camera.camera_id,
        frame_id=frame_id,
        part_id=part_id,
        source=frame.source or f"external://{camera.camera_id}",
        source_kind=frame.source_kind,
        timestamp=timestamp,
        image=frame.image,
        image_path=frame.source,
    )


def resolve_run_frame_id(frames: list[InspectionFrame]) -> str:
    """Resolve a run-level frame id from input metadata or current time."""
    for frame in frames:
        if frame.frame_id:
            return frame.frame_id
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")


def resolve_run_timestamp(frames: list[InspectionFrame]) -> str:
    """Resolve a run-level timestamp from input metadata or current time."""
    for frame in frames:
        if frame.timestamp:
            return frame.timestamp
    return datetime.now().astimezone().isoformat()


def _normalize_frame(frame: InspectionFrame | dict[str, Any]) -> InspectionFrame:
    if isinstance(frame, InspectionFrame):
        return frame
    try:
        camera_id = frame["camera_id"]
        image = frame["image"]
    except KeyError as exc:
        raise ValueError("每个 frame 必须包含 camera_id 和 image") from exc
    return InspectionFrame(
        camera_id=str(camera_id),
        image=image,
        source=frame.get("source"),
        frame_id=frame.get("frame_id"),
        timestamp=frame.get("timestamp"),
        source_kind=str(frame.get("source_kind", "external_image")),
        error_reason=frame.get("error_reason"),
    )


__all__ = [
    "build_frame_map",
    "build_frame_packet",
    "normalize_inspection_frames",
    "resolve_run_frame_id",
    "resolve_run_timestamp",
    "validate_frame_camera_ids",
]
