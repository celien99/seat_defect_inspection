"""媒体资源到标准帧数据的转换层。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2

from mvsCamera import is_mvs_source, open_mvs_capture


@dataclass(slots=True)
class MediaSourceInfo:
    """媒体源基础信息。"""

    raw_source: str
    source_kind: str


@dataclass(slots=True)
class MediaFrame:
    """标准化单帧数据。"""

    frame_index: int
    image: Any
    source: MediaSourceInfo
    width: int
    height: int
    timestamp_ms: float | None = None


class FrameStream(Protocol):
    """统一帧流协议。"""

    source_info: MediaSourceInfo

    def is_opened(self) -> bool:
        ...

    def read_frame(self) -> MediaFrame | None:
        ...

    def release(self) -> None:
        ...

    def get(self, prop_id: int) -> float:
        ...


class _CaptureFrameStream:
    """把 OpenCV/MVS 取流对象包装成统一帧流。"""

    def __init__(self, capture: Any, source: str, source_kind: str) -> None:
        self._capture = capture
        self._frame_index = 0
        self.source_info = MediaSourceInfo(raw_source=source, source_kind=source_kind)

    def is_opened(self) -> bool:
        """返回底层流是否已成功打开。"""
        return bool(self._capture.isOpened())

    def read_frame(self) -> MediaFrame | None:
        """读取并标准化一帧图像。"""
        success, image = self._capture.read()
        if not success or image is None:
            return None

        self._frame_index += 1
        height, width = image.shape[:2]
        timestamp_ms = _resolve_timestamp_ms(self._capture, self._frame_index)
        return MediaFrame(
            frame_index=self._frame_index,
            image=image,
            source=self.source_info,
            width=int(width),
            height=int(height),
            timestamp_ms=timestamp_ms,
        )

    def release(self) -> None:
        """释放底层资源。"""
        self._capture.release()

    def get(self, prop_id: int) -> float:
        """透传底层属性查询。"""
        return float(self._capture.get(prop_id))


def infer_source_kind(source: str) -> str:
    """推断输入源类型。"""
    if is_mvs_source(source):
        return "mvs_camera"
    if source.isdigit():
        return "camera_index"
    suffix = Path(source).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return "image"
    if suffix in {".mp4", ".avi", ".mov", ".mkv", ".wmv"}:
        return "video_file"
    return "generic_stream"


def resolve_capture_source(source: str) -> int | str:
    """把摄像头编号字符串转换为 OpenCV 可接受的整数。"""
    if source.isdigit():
        return int(source)
    return source


def open_frame_stream(source: str) -> FrameStream:
    """按输入类型打开标准帧流。"""
    source_kind = infer_source_kind(source)
    if source_kind == "image":
        raise ValueError(
            f"Image source '{source}' should be loaded with load_image_frame(), not open_frame_stream().",
        )
    if source_kind == "mvs_camera":
        return _CaptureFrameStream(open_mvs_capture(source), source, source_kind)
    return _CaptureFrameStream(cv2.VideoCapture(resolve_capture_source(source)), source, source_kind)


def load_image_frame(source: str) -> MediaFrame:
    """把图片资源加载为标准帧数据。"""
    image_path = Path(source)
    if not image_path.exists():
        raise ValueError(f"Image source does not exist: {source}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image source: {source}")

    height, width = image.shape[:2]
    return MediaFrame(
        frame_index=1,
        image=image,
        source=MediaSourceInfo(raw_source=source, source_kind="image"),
        width=int(width),
        height=int(height),
        timestamp_ms=None,
    )


def _resolve_timestamp_ms(capture: Any, frame_index: int) -> float | None:
    """优先读取真实时间戳，缺失时按 FPS 合成时间轴。"""
    timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp > 0:
        return timestamp

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps > 0:
        return max(0.0, (frame_index - 1) * 1000.0 / fps)
    return None
