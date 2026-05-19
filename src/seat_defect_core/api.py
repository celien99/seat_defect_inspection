"""Public inspect runtime API."""

from __future__ import annotations

from os import PathLike
from typing import Any, Union

import cv2

from .config import InspectionConfig
from .runtime_config import load_config
from .types import InspectionFrame, InspectionResponse

ConfigSource = Union[str, PathLike[str], InspectionConfig]


class SeatDefectInspector:
    """Reusable inspect-only runtime."""

    def __init__(self, config: ConfigSource) -> None:
        from .service.core import InspectionService

        self.config = resolve_config(config)
        self._service = InspectionService(self.config)

    def inspect(
        self,
        frames: list[InspectionFrame | dict[str, Any]],
        *,
        part_id: str | None = None,
        seat_model_id: str | None = None,
    ) -> tuple[InspectionResponse, dict[str, Any]]:
        """Run one full inspection from externally supplied camera frames.

        Returns a tuple of ``(response, camera_images)`` where *camera_images*
        maps ``camera_id`` to the BGR overlay image (frame + anomaly heatmap).
        """
        from .service.frames import normalize_inspection_frames
        from .service.inspection import inspect_frames
        from .service.response import build_inspection_response, collect_camera_images

        result = inspect_frames(
            self._service,
            normalize_inspection_frames(frames),
            part_id=part_id,
            seat_model_id=seat_model_id,
        )
        response = build_inspection_response(self.config, result)
        return response, collect_camera_images(result)

    def inspect_paths(
        self,
        image_paths: dict[str, str | PathLike[str]],
        *,
        part_id: str | None = None,
        seat_model_id: str | None = None,
        frame_id: str | None = None,
        timestamp: str | None = None,
    ) -> tuple[InspectionResponse, dict[str, Any]]:
        """Run one inspection from externally supplied image paths.

        Returns a tuple of ``(response, camera_images)``.
        """
        return self.inspect(
            frames_from_paths(
                image_paths,
                frame_id=frame_id,
                timestamp=timestamp,
            ),
            part_id=part_id,
            seat_model_id=seat_model_id,
        )

    def warmup(self, *, seat_model_id: str | None = None) -> None:
        """Preload active runtime models and run a lightweight PatchCore warmup."""
        self._service.warmup(seat_model_id=seat_model_id)


def inspect_once(
    config: ConfigSource,
    frames: list[InspectionFrame | dict[str, Any]],
    *,
    part_id: str | None = None,
    seat_model_id: str | None = None,
) -> tuple[InspectionResponse, dict[str, Any]]:
    """Load config and run one inspection from externally supplied frames.

    Returns a tuple of ``(response, camera_images)``.
    """
    return SeatDefectInspector(config).inspect(
        frames,
        part_id=part_id,
        seat_model_id=seat_model_id,
    )


def inspect_paths_once(
    config: ConfigSource,
    image_paths: dict[str, str | PathLike[str]],
    *,
    part_id: str | None = None,
    seat_model_id: str | None = None,
    frame_id: str | None = None,
    timestamp: str | None = None,
) -> tuple[InspectionResponse, dict[str, Any]]:
    """Load config and run one inspection from image paths.

    Returns a tuple of ``(response, camera_images)``.
    """
    return SeatDefectInspector(config).inspect_paths(
        image_paths,
        part_id=part_id,
        seat_model_id=seat_model_id,
        frame_id=frame_id,
        timestamp=timestamp,
    )


def frames_from_paths(
    image_paths: dict[str, str | PathLike[str]],
    *,
    frame_id: str | None = None,
    timestamp: str | None = None,
) -> list[InspectionFrame]:
    """Build InspectionFrame objects from a camera_id to image_path mapping."""
    frames: list[InspectionFrame] = []
    for camera_id, image_path in image_paths.items():
        path = str(image_path)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        error_reason = None if image is not None else "image_read_failed"
        frames.append(
            InspectionFrame(
                camera_id=str(camera_id),
                image=image,
                source=path,
                frame_id=frame_id,
                timestamp=timestamp,
                source_kind="image_path",
                error_reason=error_reason,
            )
        )
    return frames


def resolve_config(config: ConfigSource) -> InspectionConfig:
    """Resolve a config object or JSON/INI path into runtime config."""
    if isinstance(config, InspectionConfig):
        return config
    return load_config(str(config))


__all__ = [
    "ConfigSource",
    "SeatDefectInspector",
    "frames_from_paths",
    "inspect_paths_once",
    "inspect_once",
    "resolve_config",
]
