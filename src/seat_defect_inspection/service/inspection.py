"""在线采图入口。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Dict, List

from seat_defect_core.service.inspection import inspect_frames
from seat_defect_core.types import FramePacket, InspectionFrame, InspectionResult

from ..config import CameraConfig

if TYPE_CHECKING:
    from .core import InspectionService


def run_inspection(
    service: "InspectionService",
    part_id: str | None = None,
    *,
    seat_model_id: str | None = None,
) -> InspectionResult:
    """采集启用机位图像后，统一交给 core 主检测流程。"""
    context = service.resolve_context(seat_model_id)
    resolved_part_id = part_id or service.config.part_id
    frames = _capture_inspection_frames(
        service,
        context.cameras,
        resolved_part_id,
    )
    return inspect_frames(
        service,
        frames,
        part_id=resolved_part_id,
        seat_model_id=context.seat_model_id,
    )


def _capture_inspection_frames(
    service: "InspectionService",
    cameras: List[CameraConfig],
    part_id: str,
) -> List[InspectionFrame]:
    """并发采集机位图片，并转换成 core 主流程输入。"""
    if not cameras:
        return []

    indexed_frames: Dict[int, InspectionFrame] = {}
    max_workers = max(1, len(cameras))
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="seat-inspect-capture",
    ) as executor:
        futures = {
            executor.submit(
                service.acquisition.capture,
                camera.camera_id,
                camera.source,
                part_id,
            ): index
            for index, camera in enumerate(cameras)
        }
        for future in as_completed(futures):
            index = futures[future]
            camera = cameras[index]
            try:
                indexed_frames[index] = _inspection_frame_from_packet(future.result())
            except Exception as exc:
                indexed_frames[index] = InspectionFrame(
                    camera_id=camera.camera_id,
                    image=None,
                    source=camera.source,
                    source_kind="capture_error",
                    frame_id="",
                    timestamp=None,
                    error_reason=f"capture_failed:{exc}",
                )

    return [
        indexed_frames[index]
        for index in range(len(cameras))
        if index in indexed_frames
    ]


def _inspection_frame_from_packet(packet: FramePacket) -> InspectionFrame:
    """把采图包转换成 core 对外输入类型。"""
    return InspectionFrame(
        camera_id=packet.camera_id,
        image=packet.image,
        source=packet.source,
        frame_id=packet.frame_id,
        timestamp=packet.timestamp,
        source_kind=packet.source_kind,
    )
