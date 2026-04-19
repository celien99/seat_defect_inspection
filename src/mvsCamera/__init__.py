"""MVS 工业相机 SDK 适配层。

这里统一暴露海康 MVS 相机在 Python 中的基础接入能力，包括：
- 相机源字符串解析，例如 `mvs://0?timeout_ms=1000`
- 类似 `cv2.VideoCapture` 的取流封装
- Windows 下海康 MVS SDK DLL 的定位与诊断能力

说明：
- 该包只负责相机 SDK 接入与取流
- 业务层应优先通过 `media_inputs` 获取标准化帧数据
"""

from .frame_source import (
    MvsCameraCapture,
    MvsCameraSourceConfig,
    is_mvs_source,
    open_mvs_capture,
    parse_mvs_source,
)
from .camera_controller import CameraLocator, CameraPropertyConfig, HikCamera, MvsCameraError, MvsDeviceInfo
from .sdk.sdk_loader import MvsSdkLoadError, describe_mvs_sdk_candidates

__all__ = [
    "CameraLocator",
    "CameraPropertyConfig",
    "HikCamera",
    "MvsCameraCapture",
    "MvsCameraError",
    "MvsDeviceInfo",
    "MvsSdkLoadError",
    "MvsCameraSourceConfig",
    "describe_mvs_sdk_candidates",
    "is_mvs_source",
    "open_mvs_capture",
    "parse_mvs_source",
]
