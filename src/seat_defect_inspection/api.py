"""兼容旧导入路径的 SDK 转发层。

新项目请直接使用 `seat_defect_sdk`。这里保留是为了避免已有外部代码
从 `seat_defect_inspection.api` 导入时立即失效。
"""

from seat_defect_sdk import (
    CameraFrame,
    ConfigSource,
    InspectionSdkResponse,
    SeatDefectInspector,
    inspect_once,
)

InspectionApiResponse = InspectionSdkResponse


def inspect_folder_once(config, input_dir: str, **kwargs):
    """兼容旧 API；离线目录批测仍使用原主包 service 实现。"""
    from .config import InspectionConfig
    from .runtime_config import load_config
    from .service import inspect_image_folder

    resolved_config = config if isinstance(config, InspectionConfig) else load_config(str(config))
    return inspect_image_folder(resolved_config, input_dir=input_dir, **kwargs)

__all__ = [
    "CameraFrame",
    "ConfigSource",
    "InspectionApiResponse",
    "InspectionSdkResponse",
    "SeatDefectInspector",
    "inspect_folder_once",
    "inspect_once",
]
