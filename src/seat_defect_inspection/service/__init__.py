"""主流程服务入口。"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from seat_defect_core.schemas import InspectionResult

from ..config import InspectionConfig
from ..schemas import CaptureSummary

if TYPE_CHECKING:
    from .core import CameraPipeline, InspectionService, PreparedCameraSample

__all__ = [
    "InspectionService",
    "PreparedCameraSample",
    "CameraPipeline",
    "capture_samples",
    "inspect_image_folder",
    "run_inspection",
    "train_patchcore_models",
]

_LAZY_EXPORTS = {
    "CameraPipeline": (".core", "CameraPipeline"),
    "InspectionService": (".core", "InspectionService"),
    "PreparedCameraSample": (".core", "PreparedCameraSample"),
}


def __getattr__(name: str):
    """延迟导入核心服务对象，避免入口层把 OpenCV 依赖提前拉起。"""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def _create_service(config: InspectionConfig) -> "InspectionService":
    """按需创建服务对象，避免模块导入时就把重依赖拉起来。"""
    from .core import InspectionService

    return InspectionService(config)


def train_patchcore_models(
    config: InspectionConfig,
    seat_model_id: str | None = None,
) -> list[dict[str, Any]]:
    """训练全部机位的 PatchCore 模型。"""
    # 入口层只做轻量路由，真正逻辑仍在具体模块里。
    from .training import train_patchcore_models as _train_patchcore_models

    return _train_patchcore_models(
        _create_service(config),
        seat_model_id=seat_model_id,
    )


def capture_samples(
    config: InspectionConfig,
    part_id: str | None = None,
    *,
    output_dir: str | None = None,
    seat_model_id: str | None = None,
    save_to_train_good_dir: bool = False,
    count: int = 1,
    interval_ms: int = 0,
) -> CaptureSummary:
    """抓取并落盘全部启用机位的图像。"""
    # 入口层只做轻量路由，真正逻辑仍在具体模块里。
    from .capture import capture_samples as _capture_samples

    return _capture_samples(
        _create_service(config),
        part_id=part_id,
        output_dir=output_dir,
        seat_model_id=seat_model_id,
        save_to_train_good_dir=save_to_train_good_dir,
        count=count,
        interval_ms=interval_ms,
    )


def run_inspection(
    config: InspectionConfig,
    part_id: str | None = None,
    *,
    seat_model_id: str | None = None,
) -> InspectionResult:
    """执行一次完整检测。"""
    # 入口层只做轻量路由，真正逻辑仍在具体模块里。
    from .inspection import run_inspection as _run_inspection

    return _run_inspection(
        _create_service(config),
        part_id=part_id,
        seat_model_id=seat_model_id,
    )


def inspect_image_folder(
    config: InspectionConfig,
    input_dir: str,
    *,
    seat_model_id: str | None = None,
    output_dir: str | None = None,
    part_id: str | None = None,
) -> dict[str, Any]:
    """从本地图片文件夹批量执行离线检测。"""
    # 离线批测复用同一套服务骨架，只是把机位输入换成本地图片。
    from .offline_inspection import inspect_image_folder as _inspect_image_folder

    return _inspect_image_folder(
        _create_service(config),
        input_dir=input_dir,
        seat_model_id=seat_model_id,
        output_dir=output_dir,
        part_id=part_id,
    )
