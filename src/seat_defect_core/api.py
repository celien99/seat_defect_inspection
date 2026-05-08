"""Public inspect runtime API."""

from __future__ import annotations

from os import PathLike
from typing import Any

from .config import InspectionConfig
from .runtime_config import load_config
from .types import InspectionFrame, InspectionResponse

ConfigSource = str | PathLike[str] | InspectionConfig


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
    ) -> InspectionResponse:
        """Run one full inspection from externally supplied camera frames."""
        from .service.frames import normalize_inspection_frames
        from .service.inspection import inspect_frames
        from .service.response import build_inspection_response

        result = inspect_frames(
            self._service,
            normalize_inspection_frames(frames),
            part_id=part_id,
            seat_model_id=seat_model_id,
        )
        return build_inspection_response(self.config, result)


def inspect_once(
    config: ConfigSource,
    frames: list[InspectionFrame | dict[str, Any]],
    *,
    part_id: str | None = None,
    seat_model_id: str | None = None,
) -> InspectionResponse:
    """Load config and run one inspection from externally supplied frames."""
    return SeatDefectInspector(config).inspect(
        frames,
        part_id=part_id,
        seat_model_id=seat_model_id,
    )


def resolve_config(config: ConfigSource) -> InspectionConfig:
    """Resolve a config object or JSON path into runtime config."""
    if isinstance(config, InspectionConfig):
        return config
    return load_config(str(config))


__all__ = [
    "ConfigSource",
    "SeatDefectInspector",
    "inspect_once",
    "resolve_config",
]
