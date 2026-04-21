"""从 JSON 加载座椅缺陷检测项目配置。"""

from __future__ import annotations

import dataclasses
import json
import os
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from .config import (
    AlignmentConfig,
    CameraConfig,
    DetectionConfig,
    InspectionConfig,
    PatchCoreConfig,
    SeatModelConfig,
    YoloTrainingConfig,
)
from .debug_artifacts import resolve_debug_artifact_names

_T = TypeVar("_T")
_SUPPORTED_PATCHCORE_BACKENDS = {"full", "handcrafted"}


@dataclasses.dataclass(frozen=True)
class _FieldPolicy:
    resolver: Callable[[Path, Any], Any]
    use_default: bool = False


def _required_local_policy(*, use_default: bool = False) -> _FieldPolicy:
    return _FieldPolicy(
        lambda config_dir, value: _resolve_local_path(config_dir, value, force=True),
        use_default=use_default,
    )


def _optional_local_policy() -> _FieldPolicy:
    return _FieldPolicy(
        lambda config_dir, value: _resolve_optional_local_path(config_dir, value),
    )


def _optional_model_policy() -> _FieldPolicy:
    return _FieldPolicy(
        lambda config_dir, value: _resolve_optional_model_path(config_dir, value),
    )


def _source_policy() -> _FieldPolicy:
    return _FieldPolicy(
        lambda config_dir, value: _resolve_source_path(config_dir, value),
    )


def _yolo_model_policy() -> _FieldPolicy:
    return _FieldPolicy(
        lambda config_dir, value: _resolve_yolo_training_model_path(config_dir, value),
    )


_FIELD_POLICIES: dict[type[Any], dict[str, _FieldPolicy]] = {
    InspectionConfig: {
        "output_json_path": _required_local_policy(use_default=True),
        "debug_dir": _required_local_policy(use_default=True),
        "capture_dir": _required_local_policy(use_default=True),
    },
    CameraConfig: {
        "source": _source_policy(),
        "patchcore_model_path": _required_local_policy(),
        "train_good_dir": _optional_local_policy(),
    },
    DetectionConfig: {
        "model_path": _optional_model_policy(),
    },
    AlignmentConfig: {
        "template_image_path": _optional_local_policy(),
    },
    PatchCoreConfig: {
        "backbone_weights_path": _optional_local_policy(),
    },
    YoloTrainingConfig: {
        "model_path": _yolo_model_policy(),
        "data_config_path": _required_local_policy(use_default=True),
        "project": _required_local_policy(use_default=True),
    },
}


def load_config(path: str) -> InspectionConfig:
    """加载缺陷检测主配置。"""
    config_dir, inspection_payload = _load_inspection_payload(path)
    camera_payloads = inspection_payload.get("cameras") or []
    seat_model_payloads = inspection_payload.get("seat_models") or []
    if not camera_payloads and not seat_model_payloads:
        raise ValueError("缺陷检测配置必须包含 `cameras` 或 `seat_models`")
    config = _build_dataclass(InspectionConfig, inspection_payload, config_dir)
    _validate_inspection_config(config)
    return config


def load_yolo_training_config(path: str, seat_model_id: str | None = None) -> YoloTrainingConfig:
    """加载 YOLO 训练配置。"""
    config_dir, inspection_payload = _load_inspection_payload(path)
    training_payload, selected_seat_model_id = _resolve_yolo_training_payload(
        inspection_payload,
        seat_model_id,
    )
    if training_payload is None:
        raise ValueError("配置文件缺少 `yolo_training` 配置块")

    normalized = dict(training_payload)
    normalized["seat_model_id"] = selected_seat_model_id
    return _build_dataclass(YoloTrainingConfig, normalized, config_dir)


def _load_inspection_payload(path: str) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    inspection_payload = payload.get("seat_defect_inspection", payload)
    return config_path.parent, inspection_payload


def _build_dataclass(cls: type[_T], payload: dict[str, Any], config_dir: Path) -> _T:
    normalized = _normalize_payload(cls, payload, config_dir)
    field_map = {field.name: field for field in dataclasses.fields(cls)}
    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}

    for name, field in field_map.items():
        if name not in normalized:
            continue
        kwargs[name] = _coerce_value(
            type_hints.get(name, field.type),
            normalized[name],
            config_dir,
        )
    return cls(**kwargs)


def _coerce_value(type_hint: Any, value: Any, config_dir: Path) -> Any:
    if value is None:
        return None

    origin = get_origin(type_hint)
    if origin is list:
        item_type = get_args(type_hint)[0]
        return [_coerce_value(item_type, item, config_dir) for item in value]

    nested_dataclass = _resolve_dataclass_type(type_hint)
    if nested_dataclass is not None and isinstance(value, dict):
        return _build_dataclass(nested_dataclass, value, config_dir)

    return value


def _resolve_dataclass_type(type_hint: Any) -> type[Any] | None:
    if isinstance(type_hint, type) and dataclasses.is_dataclass(type_hint):
        return type_hint

    origin = get_origin(type_hint)
    if origin in {types.UnionType, Union}:
        for candidate in get_args(type_hint):
            nested = _resolve_dataclass_type(candidate)
            if nested is not None:
                return nested

    return None


def _normalize_payload(cls: type[Any], payload: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    normalized = dict(payload)
    if cls is InspectionConfig:
        _fill_default_seat_model_id(normalized)
    elif cls is SeatModelConfig:
        _attach_seat_model_id_to_training(normalized)

    for field_name, policy in _FIELD_POLICIES.get(cls, {}).items():
        if field_name in normalized:
            value = normalized[field_name]
            if policy.use_default and _is_missing_value(value):
                value = _field_default(cls, field_name)
        elif policy.use_default:
            value = _field_default(cls, field_name)
        else:
            continue
        normalized[field_name] = policy.resolver(config_dir, value)
    return normalized


def _validate_inspection_config(config: InspectionConfig) -> None:
    resolve_debug_artifact_names(config.debug_artifact_mode)
    if config.default_seat_model_id and config.seat_models:
        available_ids = {item.seat_model_id for item in config.seat_models}
        if config.default_seat_model_id not in available_ids:
            available = ", ".join(sorted(available_ids))
            raise ValueError(
                "default_seat_model_id 未出现在 seat_models 中，"
                f"当前值: `{config.default_seat_model_id}`，可选值: {available}"
            )

    _validate_camera_configs(config.cameras, scope="顶层 cameras")
    for seat_model in config.seat_models:
        _validate_camera_configs(
            seat_model.cameras,
            scope=f"seat_model `{seat_model.seat_model_id}`",
        )


def _validate_camera_configs(cameras: list[CameraConfig], *, scope: str) -> None:
    for camera in cameras:
        _validate_patchcore_config(camera, scope=scope)


def _validate_patchcore_config(camera: CameraConfig, *, scope: str) -> None:
    backend = camera.patchcore.backend.strip().lower()
    if backend not in _SUPPORTED_PATCHCORE_BACKENDS:
        supported = ", ".join(sorted(_SUPPORTED_PATCHCORE_BACKENDS))
        raise ValueError(
            f"{scope} 中 camera `{camera.camera_id}` 的 patchcore.backend "
            f"`{camera.patchcore.backend}` 不受支持，可选值: {supported}"
        )
    if backend != "full":
        return
    if camera.patchcore.backbone_pretrained or camera.patchcore.backbone_weights_path:
        return
    raise ValueError(
        f"{scope} 中 camera `{camera.camera_id}` 配置了 patchcore.backend=full，"
        "但没有提供可用 backbone 权重。"
        " 请设置 patchcore.backbone_pretrained=true，"
        "或配置 patchcore.backbone_weights_path，"
        "或把 patchcore.backend 改为 handcrafted。"
    )


def _fill_default_seat_model_id(payload: dict[str, Any]) -> None:
    seat_models = payload.get("seat_models") or []
    if seat_models and not payload.get("default_seat_model_id"):
        first_model_id = seat_models[0].get("seat_model_id")
        if first_model_id:
            payload["default_seat_model_id"] = first_model_id


def _attach_seat_model_id_to_training(payload: dict[str, Any]) -> None:
    yolo_training = payload.get("yolo_training")
    if yolo_training is None:
        return
    training_payload = dict(yolo_training)
    training_payload["seat_model_id"] = payload.get("seat_model_id")
    payload["yolo_training"] = training_payload


def _field_default(cls: type[Any], field_name: str) -> Any:
    for field in dataclasses.fields(cls):
        if field.name != field_name:
            continue
        if field.default is not dataclasses.MISSING:
            return field.default
        if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            return field.default_factory()
        raise TypeError(f"{cls.__name__}.{field_name} 没有默认值")
    raise KeyError(f"{cls.__name__} 不存在字段 `{field_name}`")


def _is_missing_value(value: Any) -> bool:
    return value is None or value == ""


def _select_seat_model_payload(
    seat_models: list[dict[str, Any]],
    seat_model_id: str | None,
) -> dict[str, Any] | None:
    """按 seat_model_id 查找对应 payload；未指定时返回第一个；找不到时抛出。"""
    if not seat_models:
        return None
    if seat_model_id is None:
        return seat_models[0]
    for item in seat_models:
        if item.get("seat_model_id") == seat_model_id:
            return item
    available = ", ".join(str(item.get("seat_model_id")) for item in seat_models)
    raise ValueError(f"未知 seat_model_id `{seat_model_id}`，可选值：{available}")


def _resolve_yolo_training_payload(
    inspection_payload: dict[str, Any],
    seat_model_id: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """从配置中定位 yolo_training payload 及最终使用的 seat_model_id。"""
    top_level_training = inspection_payload.get("yolo_training")
    seat_models: list[dict[str, Any]] = inspection_payload.get("seat_models") or []
    if not seat_models:
        return top_level_training, seat_model_id

    effective_id = seat_model_id or inspection_payload.get("default_seat_model_id")
    selected = _select_seat_model_payload(seat_models, effective_id)
    if selected is None:
        return top_level_training, seat_model_id

    resolved_id: str | None = selected.get("seat_model_id")
    if selected.get("yolo_training") is not None:
        return selected["yolo_training"], resolved_id
    return top_level_training, resolved_id


def _resolve_source_path(config_dir: Path, value: str) -> str:
    """解析相机数据源路径; URL 协议或纯数字设备号直接透传。"""
    if "://" in value or value.isdigit():
        return value
    return _resolve_local_path(config_dir, value, force=True)


def _resolve_optional_model_path(config_dir: Path, value: str | None) -> str | None:
    """解析可选的模型路径；为 None 时直接返回 None。"""
    if value is None:
        return None
    return _resolve_local_path(config_dir, value, force=False)


def _resolve_optional_local_path(config_dir: Path, value: str | None) -> str | None:
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
    """将相对路径解析为基于 config_dir 的绝对路径；绝对路径直接返回。"""
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
    """判断字符串是否看起来像本地文件路径（非 URL、非设备号）。"""
    if value.startswith(".") or os.sep in value:
        return True
    if os.altsep is not None and os.altsep in value:
        return True
    return Path(value).suffix.lower() in _LOCAL_PATH_SUFFIXES
