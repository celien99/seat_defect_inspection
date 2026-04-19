"""统一媒体输入层。

职责：
- 把视频、图片、普通摄像头、MVS 工业相机统一为标准化帧数据
- 让业务层只依赖稳定的数据接口，而不是直接耦合具体 SDK
"""

from .core import (
    FrameStream,
    MediaFrame,
    MediaSourceInfo,
    infer_source_kind,
    load_image_frame,
    open_frame_stream,
    resolve_capture_source,
)

__all__ = [
    "FrameStream",
    "MediaFrame",
    "MediaSourceInfo",
    "infer_source_kind",
    "load_image_frame",
    "open_frame_stream",
    "resolve_capture_source",
]
