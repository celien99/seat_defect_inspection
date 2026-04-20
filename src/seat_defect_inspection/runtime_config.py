"""从 JSON 加载座椅缺陷检测项目配置。"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any, TypeVar

from .config import (
    AlignmentConfig,
    CameraConfig,
    ColorBranchConfig,
    DetectionConfig,
    FusionConfig,
    InspectionConfig,
    PatchCoreConfig,
    PreprocessConfig,
    QualityGuardConfig,
    RoiRefineConfig,
    SeatModelConfig,
    YoloTrainingConfig,
)
from .schemas import BoundingBox

_T = TypeVar("_T")
_MISSING = dataclasses.MISSING


def _field_default(cls: type, name: str) -> Any:
    """返回 dataclass 字段的默认值；字段不存在或无默认值时抛出 KeyError / TypeError。"""
    for f in dataclasses.fields(cls):  # type: ignore[arg-type]
        if f.name == name:
            if f.default is not _MISSING:
                return f.default
            if f.default_factory is not _MISSING:  # type: ignore[misc]
                return f.default_factory()
            raise TypeError(f"{cls.__name__}.{name} 没有默认值")
    raise KeyError(f"{cls.__name__} 不存在字段 `{name}`")

def _get_or_default(payload: dict[str, Any], cls: type, key: str) -> Any:
    """从 payload 中取值；键不存在时回退到 dataclass 字段默认值。"""
    return payload.get(key, _field_default(cls, key))

def load_config(path: str) -> InspectionConfig:
    """加载缺陷检测主配置。"""
    config_path = Path(path).resolve()
    config_dir = config_path.parent
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    inspection_payload = payload.get("seat_defect_inspection", payload)

    camera_payloads = inspection_payload.get("cameras") or []
    seat_model_payloads = inspection_payload.get("seat_models") or []
    if not camera_payloads and not seat_model_payloads:
        raise ValueError("缺陷检测配置必须包含 `cameras` 或 `seat_models`")

    seat_models = [
        _build_seat_model_config(model_payload, config_dir)
        for model_payload in seat_model_payloads
    ]
    default_seat_model_id = inspection_payload.get("default_seat_model_id")
    if seat_models and not default_seat_model_id:
        default_seat_model_id = seat_models[0].seat_model_id

    return InspectionConfig(
        cameras=[
            _build_camera_config(camera_payload, config_dir)
            for camera_payload in camera_payloads
        ],
        seat_models=seat_models,
        default_seat_model_id=default_seat_model_id,
        output_json_path=_resolve_local_path(
            config_dir,
            _get_or_default(inspection_payload, InspectionConfig, "output_json_path"),
            force=True,
        ),
        debug_dir=_resolve_local_path(
            config_dir,
            _get_or_default(inspection_payload, InspectionConfig, "debug_dir"),
            force=True,
        ),
        capture_dir=_resolve_local_path(
            config_dir,
            _get_or_default(inspection_payload, InspectionConfig, "capture_dir"),
            force=True,
        ),
        save_debug_artifacts=bool(
            _get_or_default(inspection_payload, InspectionConfig, "save_debug_artifacts")
        ),
        capture_retries=int(
            _get_or_default(inspection_payload, InspectionConfig, "capture_retries")
        ),
        part_id=_get_or_default(inspection_payload, InspectionConfig, "part_id"),
        fusion=FusionConfig(**(inspection_payload.get("fusion") or {})),
    )


def load_yolo_training_config(path: str, seat_model_id: str | None = None) -> YoloTrainingConfig:
    """加载 YOLO 训练配置。"""
    config_path = Path(path).resolve()
    config_dir = config_path.parent
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    inspection_payload = payload.get("seat_defect_inspection", payload)

    training_payload, selected_seat_model_id = _resolve_yolo_training_payload(
        payload, inspection_payload, seat_model_id
    )

    if training_payload is None:
        raise ValueError("配置文件缺少 `yolo_training` 配置块")

    return _build_yolo_training_config(
        training_payload,
        config_dir,
        seat_model_id=selected_seat_model_id,
    )


def _build_seat_model_config(payload: dict[str, Any], config_dir: Path) -> SeatModelConfig:
    return SeatModelConfig(
        seat_model_id=payload["seat_model_id"],
        display_name=payload.get("display_name"),
        cameras=[
            _build_camera_config(camera_payload, config_dir)
            for camera_payload in payload.get("cameras", [])
        ],
        yolo_training=(
            _build_yolo_training_config(
                payload["yolo_training"],
                config_dir,
                seat_model_id=payload["seat_model_id"],
            )
            if payload.get("yolo_training") is not None
            else None
        ),
    )


def _build_yolo_training_config(
    payload: dict[str, Any],
    config_dir: Path,
    *,
    seat_model_id: str | None = None,
) -> YoloTrainingConfig:
    normalized = dict(payload)
    normalized["data_config_path"] = _resolve_local_path(
        config_dir,
        normalized["data_config_path"],
        force=True,
    )
    normalized["project"] = _resolve_local_path(
        config_dir,
        normalized.get("project", _field_default(YoloTrainingConfig, "project")),
        force=True,
    )
    if "model_path" in normalized:
        normalized["model_path"] = _resolve_yolo_training_model_path(
            config_dir,
            normalized["model_path"],
        )
    normalized["seat_model_id"] = seat_model_id
    return YoloTrainingConfig(**normalized)


def _build_camera_config(payload: dict[str, Any], config_dir: Path) -> CameraConfig:
    """构建单个相机配置。"""
    return CameraConfig(
        camera_id=payload["camera_id"],
        source=_resolve_source_path(config_dir, payload["source"]),
        patchcore_model_path=_resolve_local_path(
            config_dir,
            payload["patchcore_model_path"],
            force=True,
        ),
        train_good_dir=(
            _resolve_local_path(config_dir, payload["train_good_dir"], force=True)
            if payload.get("train_good_dir") is not None
            else None
        ),
        enabled=bool(payload.get("enabled", True)),
        color_insensitive_mode=bool(payload.get("color_insensitive_mode", False)),
        quality=QualityGuardConfig(**(payload.get("quality") or {})),
        preprocess=PreprocessConfig(**(payload.get("preprocess") or {})),
        detection=_build_detection_config(payload.get("detection") or {}, config_dir),
        roi=_build_roi_config(payload.get("roi") or {}, config_dir),
        patchcore=_build_patchcore_config(payload.get("patchcore") or {}, config_dir),
        color_branch=ColorBranchConfig(**(payload.get("color_branch") or {})),
    )


def _build_detection_config(payload: dict[str, Any], config_dir: Path) -> DetectionConfig:
    """构建检测子配置，处理 model_path 和 fallback_box 的路径解析。"""
    normalized = dict(payload)
    fallback_box_payload = normalized.pop("fallback_box", None)
    model_path = normalized.pop("model_path", None)
    return DetectionConfig(
        **normalized,
        model_path=_resolve_optional_model_path(config_dir, model_path),
        fallback_box=(
            BoundingBox(
                x1=float(fallback_box_payload["x1"]),
                y1=float(fallback_box_payload["y1"]),
                x2=float(fallback_box_payload["x2"]),
                y2=float(fallback_box_payload["y2"]),
            )
            if fallback_box_payload is not None
            else None
        ),
    )


def _build_roi_config(payload: dict[str, Any], config_dir: Path) -> RoiRefineConfig:
    """构建 ROI 精修子配置，处理对齐模板路径解析。"""
    normalized = dict(payload)
    alignment_payload = dict(normalized.pop("alignment", {}) or {})
    template_image_path = alignment_payload.pop("template_image_path", None)
    return RoiRefineConfig(
        **normalized,
        alignment=AlignmentConfig(
            **alignment_payload,
            template_image_path=(
                _resolve_local_path(config_dir, template_image_path, force=True)
                if template_image_path
                else None
            ),
        ),
    )


def _build_patchcore_config(payload: dict[str, Any], config_dir: Path) -> PatchCoreConfig:
    """构建 PatchCore 子配置，处理 backbone 权重路径解析。"""
    normalized = dict(payload)
    backbone_weights_path = normalized.pop("backbone_weights_path", None)
    return PatchCoreConfig(
        **normalized,
        backbone_weights_path=(
            _resolve_local_path(config_dir, backbone_weights_path, force=True)
            if backbone_weights_path
            else None
        ),
    )


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
    payload: dict[str, Any],
    inspection_payload: dict[str, Any],
    seat_model_id: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """从配置中定位 yolo_training payload 及最终使用的 seat_model_id。

    优先级：seat_models[seat_model_id].yolo_training > 顶层 yolo_training。
    返回 (training_payload, resolved_seat_model_id)。
    """
    top_level_training = inspection_payload.get("yolo_training", payload.get("yolo_training"))
    seat_models: list[dict[str, Any]] = inspection_payload.get("seat_models") or []

    if not seat_models:
        return top_level_training, seat_model_id

    effective_id = seat_model_id or inspection_payload.get("default_seat_model_id")
    selected = _select_seat_model_payload(seat_models, effective_id)

    if selected is None:
        return top_level_training, seat_model_id

    resolved_id: str | None = selected.get("seat_model_id")

    # 优先使用 seat_model 内嵌的 yolo_training
    if selected.get("yolo_training") is not None:
        return selected["yolo_training"], resolved_id

    # seat_model 内无训练配置，回退到顶层，但仍记录匹配到的 seat_model_id
    return top_level_training, resolved_id


def _resolve_source_path(config_dir: Path, value: str) -> str:
    """解析相机数据源路径;URL 协议或纯数字设备号直接透传。"""
    if "://" in value or value.isdigit():
        return value
    return _resolve_local_path(config_dir, value, force=True)


def _resolve_optional_model_path(config_dir: Path, value: str | None) -> str | None:
    """解析可选的模型路径；为 None 时直接返回 None。"""
    if value is None:
        return None
    return _resolve_local_path(config_dir, value, force=False)


def _resolve_yolo_training_model_path(config_dir: Path, value: str) -> str:
    """解析 YOLO 训练模型来源。

    规则：
    - 显式相对路径 / 绝对路径按本地文件处理
    - 纯文件名若在配置目录下真实存在，也按本地文件处理
    - 否则保留原值，允许 `yolo11n.pt` 这类 Ultralytics 模型别名透传
    """
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
