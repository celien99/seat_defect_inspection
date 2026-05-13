"""采图服务。"""

from __future__ import annotations

from datetime import datetime

from media_inputs import infer_source_kind, load_image_frame, open_frame_stream
from seat_defect_core.types import FramePacket


class AcquisitionService:
    """按机位抓取标准化单帧图像。"""

    def __init__(self, capture_retries: int = 3) -> None:
        self.capture_retries = max(1, int(capture_retries))

    def capture(self, camera_id: str, source: str, part_id: str) -> FramePacket:
        """从图片、视频、普通相机或 MVS 相机抓取一帧。"""
        source_kind = infer_source_kind(source)
        timestamp = datetime.now().astimezone()
        frame_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")

        if source_kind == "image":
            media_frame = load_image_frame(source)
            return FramePacket(
                camera_id=camera_id,
                frame_id=frame_id,
                part_id=part_id,
                source=source,
                source_kind=source_kind,
                timestamp=timestamp.isoformat(),
                image=media_frame.image,
                image_path=source,
            )

        stream = open_frame_stream(source)
        try:
            if not stream.is_opened():
                raise RuntimeError(f"无法打开输入源：{source}")

            media_frame = None
            for _ in range(self.capture_retries):
                media_frame = stream.read_frame()
                if media_frame is not None:
                    break

            if media_frame is None:
                raise RuntimeError(f"无法从输入源读取图像：{source}")

            return FramePacket(
                camera_id=camera_id,
                frame_id=frame_id,
                part_id=part_id,
                source=source,
                source_kind=source_kind,
                timestamp=timestamp.isoformat(),
                image=media_frame.image,
                image_path=None,
            )
        finally:
            stream.release()
