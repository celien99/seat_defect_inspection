"""运行时配置解析通用小工具。"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any


def _field_names(cls: type[Any]) -> set[str]:
    """收集 dataclass 字段名，用于提前拦截拼写错误。"""
    return {field.name for field in dataclasses.fields(cls)}


def _reject_unknown_keys(payload: dict[str, Any], allowed_keys: set[str], scope: str) -> None:
    """拒绝未知字段，避免配置拼写错误悄悄落回默认值。"""
    unexpected = sorted(key for key in payload if key not in allowed_keys)
    if not unexpected:
        return
    formatted = ", ".join(f"`{key}`" for key in unexpected)
    raise ValueError(f"{scope} 包含未知字段: {formatted}")


def _expect_dict(value: Any, scope: str) -> dict[str, Any]:
    """确保当前配置块是对象。"""
    if not isinstance(value, dict):
        raise TypeError(f"{scope} 必须是对象")
    return value


def _optional_dict(value: Any, scope: str) -> dict[str, Any] | None:
    """允许为空的对象配置块。"""
    if value is None:
        return None
    return _expect_dict(value, scope)


def _ensure_list(value: Any, scope: str) -> list[Any]:
    """确保当前配置块是数组。"""
    if not isinstance(value, list):
        raise TypeError(f"{scope} 必须是数组")
    return value


def _require_key(payload: dict[str, Any], key: str, scope: str) -> Any:
    """读取必填字段，空字符串也视为缺失。"""
    value = payload.get(key)
    if value is None or value == "":
        raise ValueError(f"{scope} 缺少 `{key}`")
    return value


def _require_string(payload: dict[str, Any], key: str, scope: str) -> str:
    """读取必填字符串字段。"""
    return str(_require_key(payload, key, scope))


def _optional_string(value: Any) -> str | None:
    """读取可选字符串字段。"""
    if value is None or value == "":
        return None
    return str(value)


def _string_or_default(value: Any, default: str) -> str:
    """读取字符串字段，缺省时回退到 dataclass 默认值。"""
    if value is None or value == "":
        return default
    return str(value)


def _bool_or_default(value: Any, default: bool) -> bool:
    """读取布尔字段，缺省时回退默认值。"""
    if value is None:
        return default
    return bool(value)


def _int_or_default(value: Any, default: int) -> int:
    """读取整数字段，缺省时回退默认值。"""
    if value is None:
        return default
    return int(value)


def _optional_int(value: Any) -> int | None:
    """读取可选整数字段。"""
    if value is None or value == "":
        return None
    return int(value)


def _float_or_default(value: Any, default: float) -> float:
    """读取浮点字段，缺省时回退默认值。"""
    if value is None:
        return default
    return float(value)


def _optional_float(value: Any) -> float | None:
    """读取可选浮点字段。"""
    if value is None or value == "":
        return None
    return float(value)


def _string_list(value: Any, *, scope: str, default: list[str]) -> list[str]:
    """读取字符串数组字段。"""
    if value is None:
        return list(default)
    return [str(item) for item in _ensure_list(value, scope)]


def _float_list(value: Any, *, scope: str) -> list[float] | None:
    """读取浮点数组字段。"""
    if value is None:
        return None
    return [float(item) for item in _ensure_list(value, scope)]


def _float_matrix(value: Any, *, scope: str) -> list[list[float]] | None:
    """读取二维浮点矩阵字段。"""
    if value is None:
        return None
    rows = _ensure_list(value, scope)
    return [
        [float(item) for item in _ensure_list(row, f"{scope}[{index}]")]
        for index, row in enumerate(rows)
    ]


def _resolve_source_path(config_dir: Path, value: str) -> str:
    """解析相机数据源路径；URL 协议或纯数字设备号直接透传。"""
    if "://" in value or value.isdigit():
        return value
    return _resolve_local_path(config_dir, value, force=True)


def _resolve_optional_model_path(config_dir: Path, value: str | None) -> str | None:
    """解析可选模型路径；非本地名称则保持原样。"""
    if value is None:
        return None
    return _resolve_local_path(config_dir, value, force=False)


def _resolve_optional_local_path(config_dir: Path, value: str | None) -> str | None:
    """解析可选本地路径。"""
    if not value:
        return None
    return _resolve_local_path(config_dir, value, force=True)


def _resolve_yolo_training_model_path(config_dir: Path, value: str) -> str:
    """解析 YOLO 训练模型来源。"""
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    if value.startswith(".") or os.sep in value or (os.altsep is not None and os.altsep in value):
        return _resolve_local_path(config_dir, value, force=True)

    resolved = (config_dir / candidate).resolve()
    if resolved.exists():
        return str(resolved)
    return value


def _resolve_local_path(config_dir: Path, value: str, *, force: bool) -> str:
    """将相对路径解析为基于 config_dir 的绝对路径。"""
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    if not force and not _looks_like_local_path(value):
        return value
    return str((config_dir / candidate).resolve())


_LOCAL_PATH_SUFFIXES = {
    ".pt", ".pth", ".onnx", ".yaml", ".yml", ".json", ".png", ".jpg", ".jpeg",
}


def _looks_like_local_path(value: str) -> bool:
    """判断字符串是否看起来像本地文件路径。"""
    if value.startswith(".") or os.sep in value:
        return True
    if os.altsep is not None and os.altsep in value:
        return True
    return Path(value).suffix.lower() in _LOCAL_PATH_SUFFIXES
