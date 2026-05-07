"""可嵌入外部项目的缺陷检测调用入口。"""

from os import PathLike
from typing import Any

from .config import InspectionConfig
from .runtime_config import load_config
from .schemas import InspectionResult

ConfigSource = str | PathLike[str] | InspectionConfig


class SeatDefectInspector:
    """可复用的座椅缺陷检测运行器。

    外部项目如果需要连续检测，应复用同一个实例，这样可以复用已构造的
    YOLO、PatchCore 和相机管线缓存，避免每次调用都重新初始化。
    """

    def __init__(self, config: ConfigSource) -> None:
        self.config = _resolve_config(config)
        self._service = _create_service(self.config)

    def inspect(
        self,
        part_id: str | None = None,
        *,
        seat_model_id: str | None = None,
    ) -> InspectionResult:
        """执行一次完整在线检测。"""
        return _run_online_inspection(
            self._service,
            part_id=part_id,
            seat_model_id=seat_model_id,
        )

    def inspect_folder(
        self,
        input_dir: str,
        *,
        seat_model_id: str | None = None,
        output_dir: str | None = None,
        part_id: str | None = None,
    ) -> dict[str, Any]:
        """从本地图片文件夹批量执行离线检测。"""
        return _run_offline_folder_inspection(
            self._service,
            input_dir=input_dir,
            seat_model_id=seat_model_id,
            output_dir=output_dir,
            part_id=part_id,
        )


def inspect_once(
    config: ConfigSource,
    part_id: str | None = None,
    *,
    seat_model_id: str | None = None,
) -> InspectionResult:
    """使用配置路径或配置对象执行一次完整在线检测。"""
    return SeatDefectInspector(config).inspect(
        part_id=part_id,
        seat_model_id=seat_model_id,
    )


def inspect_folder_once(
    config: ConfigSource,
    input_dir: str,
    *,
    seat_model_id: str | None = None,
    output_dir: str | None = None,
    part_id: str | None = None,
) -> dict[str, Any]:
    """使用配置路径或配置对象执行一次离线图片文件夹批量检测。"""
    return SeatDefectInspector(config).inspect_folder(
        input_dir=input_dir,
        seat_model_id=seat_model_id,
        output_dir=output_dir,
        part_id=part_id,
    )


def _resolve_config(config: ConfigSource) -> InspectionConfig:
    if isinstance(config, InspectionConfig):
        return config
    return load_config(str(config))


def _create_service(config: InspectionConfig):
    from .service.core import InspectionService

    return InspectionService(config)


def _run_online_inspection(
    service,
    *,
    part_id: str | None,
    seat_model_id: str | None,
) -> InspectionResult:
    from .service.inspection import run_inspection

    return run_inspection(
        service,
        part_id=part_id,
        seat_model_id=seat_model_id,
    )


def _run_offline_folder_inspection(
    service,
    *,
    input_dir: str,
    seat_model_id: str | None,
    output_dir: str | None,
    part_id: str | None,
) -> dict[str, Any]:
    from .service.offline_inspection import inspect_image_folder

    return inspect_image_folder(
        service,
        input_dir=input_dir,
        seat_model_id=seat_model_id,
        output_dir=output_dir,
        part_id=part_id,
    )


__all__ = [
    "ConfigSource",
    "SeatDefectInspector",
    "inspect_folder_once",
    "inspect_once",
]
