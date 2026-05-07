"""YOLO 训练与标注转换入口。"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .labelme_to_yolo import ConversionSummary, convert_labelme_split
    from .training import train_yolo_model

__all__ = [
    "ConversionSummary",
    "convert_labelme_split",
    "train_yolo_model",
]

_LAZY_EXPORTS = {
    "ConversionSummary": (".labelme_to_yolo", "ConversionSummary"),
    "convert_labelme_split": (".labelme_to_yolo", "convert_labelme_split"),
}


def train_yolo_model(*args, **kwargs):
    """延迟导入训练模块，避免其他命令被训练依赖拖上。"""
    from .training import train_yolo_model as _train_yolo_model

    return _train_yolo_model(*args, **kwargs)


def __getattr__(name: str):
    """按需导入 YOLO 工程子模块，避免训练依赖提前加载。"""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
