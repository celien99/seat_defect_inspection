"""运行时主配置解析。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import CameraConfig, FusionConfig, InspectionConfig, SeatModelConfig, YoloTrainingConfig
from .runtime_config_camera_parsers import _parse_camera_config, _parse_preprocess_config
from .runtime_config_values import (
    _bool_or_default,
    _ensure_list,
    _expect_dict,
    _field_names,
    _int_or_default,
    _optional_dict,
    _optional_string,
    _reject_unknown_keys,
    _require_string,
    _resolve_local_path,
    _resolve_yolo_training_model_path,
    _string_or_default,
)


def _parse_inspection_config(payload: dict[str, Any], config_dir: Path) -> InspectionConfig:
    """解析缺陷检测主配置。"""
    scope = "InspectionConfig"
    _reject_unknown_keys(payload, _field_names(InspectionConfig), scope)

    cameras_payload = payload.get("cameras") or []
    seat_models_payload = payload.get("seat_models") or []
    if not cameras_payload and not seat_models_payload:
        raise ValueError("缺陷检测配置必须包含 `cameras` 或 `seat_models`")

    cameras = _parse_camera_list(
        cameras_payload,
        config_dir,
        scope=f"{scope}.cameras",
    )
    seat_models = [
        _parse_seat_model_config(item, config_dir, scope=f"{scope}.seat_models[{index}]")
        for index, item in enumerate(_ensure_list(seat_models_payload, f"{scope}.seat_models"))
    ]

    defaults = InspectionConfig()
    default_seat_model_id = _optional_string(payload.get("default_seat_model_id"))
    if default_seat_model_id is None and seat_models:
        default_seat_model_id = seat_models[0].seat_model_id

    yolo_training = _parse_optional_yolo_training(
        payload.get("yolo_training"),
        config_dir,
        scope=f"{scope}.yolo_training",
    )

    return InspectionConfig(
        cameras=cameras,
        seat_models=seat_models,
        default_seat_model_id=default_seat_model_id,
        output_json_path=_resolve_local_path(
            config_dir,
            _string_or_default(payload.get("output_json_path"), defaults.output_json_path),
            force=True,
        ),
        debug_dir=_resolve_local_path(
            config_dir,
            _string_or_default(payload.get("debug_dir"), defaults.debug_dir),
            force=True,
        ),
        capture_dir=_resolve_local_path(
            config_dir,
            _string_or_default(payload.get("capture_dir"), defaults.capture_dir),
            force=True,
        ),
        save_debug_artifacts=_bool_or_default(
            payload.get("save_debug_artifacts"),
            defaults.save_debug_artifacts,
        ),
        debug_artifact_mode=_string_or_default(
            payload.get("debug_artifact_mode"),
            defaults.debug_artifact_mode,
        ),
        capture_retries=_int_or_default(payload.get("capture_retries"), defaults.capture_retries),
        part_id=_string_or_default(payload.get("part_id"), defaults.part_id),
        fusion=_parse_fusion_config(
            payload.get("fusion"),
            scope=f"{scope}.fusion",
        ),
        yolo_training=yolo_training,
    )


def _parse_seat_model_config(payload: dict[str, Any], config_dir: Path, *, scope: str) -> SeatModelConfig:
    """解析座椅型号配置。"""
    payload = _expect_dict(payload, scope)
    _reject_unknown_keys(payload, _field_names(SeatModelConfig), scope)

    cameras = _parse_camera_list(
        payload.get("cameras"),
        config_dir,
        scope=f"{scope}.cameras",
    )
    # 这里沿用旧行为：非空的数值型 seat_model_id 也允许转成字符串。
    seat_model_id = _require_string(payload, "seat_model_id", scope)

    return SeatModelConfig(
        seat_model_id=seat_model_id,
        cameras=cameras,
        display_name=_optional_string(payload.get("display_name")),
        yolo_training=_parse_optional_yolo_training(
            payload.get("yolo_training"),
            config_dir,
            scope=f"{scope}.yolo_training",
            seat_model_id=seat_model_id,
        ),
    )


def _parse_camera_list(
    payload: Any,
    config_dir: Path,
    *,
    scope: str,
) -> list[CameraConfig]:
    """统一解析机位列表，避免顶层和 seat_model 下各写一遍。"""
    items = _ensure_list(payload or [], scope)
    return [
        _parse_camera_config(item, config_dir, scope=f"{scope}[{index}]")
        for index, item in enumerate(items)
    ]


def _parse_fusion_config(payload: Any, *, scope: str) -> FusionConfig:
    """解析多机位融合配置。"""
    defaults = FusionConfig()
    if payload is None:
        return defaults
    payload = _expect_dict(payload, scope)
    _reject_unknown_keys(payload, _field_names(FusionConfig), scope)
    return FusionConfig(
        reject_on_any_reject=_bool_or_default(
            payload.get("reject_on_any_reject"),
            defaults.reject_on_any_reject,
        ),
        ng_strategy=_string_or_default(payload.get("ng_strategy"), defaults.ng_strategy),
        early_stop_on_ng=_bool_or_default(
            payload.get("early_stop_on_ng"),
            defaults.early_stop_on_ng,
        ),
        defect_overrides_reject=_bool_or_default(
            payload.get("defect_overrides_reject"),
            defaults.defect_overrides_reject,
        ),
    )


def _parse_optional_yolo_training(
    payload: Any,
    config_dir: Path,
    *,
    scope: str,
    seat_model_id: str | None = None,
) -> YoloTrainingConfig | None:
    """统一处理可选的 yolo_training 段，少写一层 if/else。"""
    if payload is None:
        return None
    return _parse_yolo_training_config(
        _expect_dict(payload, scope),
        config_dir,
        scope=scope,
        seat_model_id=seat_model_id,
    )


def _parse_yolo_training_config(
    payload: dict[str, Any],
    config_dir: Path,
    *,
    scope: str,
    seat_model_id: str | None = None,
) -> YoloTrainingConfig:
    """解析 YOLO 训练配置。"""
    defaults = YoloTrainingConfig()
    payload = _expect_dict(payload, scope)
    _reject_unknown_keys(payload, _field_names(YoloTrainingConfig), scope)
    resolved_seat_model_id = seat_model_id if seat_model_id is not None else _optional_string(payload.get("seat_model_id"))
    return YoloTrainingConfig(
        model_path=_resolve_yolo_training_model_path(
            config_dir,
            _string_or_default(payload.get("model_path"), defaults.model_path),
        ),
        data_config_path=_resolve_local_path(
            config_dir,
            _string_or_default(payload.get("data_config_path"), defaults.data_config_path),
            force=True,
        ),
        epochs=_int_or_default(payload.get("epochs"), defaults.epochs),
        imgsz=_int_or_default(payload.get("imgsz"), defaults.imgsz),
        batch=_int_or_default(payload.get("batch"), defaults.batch),
        device=_string_or_default(payload.get("device"), defaults.device),
        project=_resolve_local_path(
            config_dir,
            _string_or_default(payload.get("project"), defaults.project),
            force=True,
        ),
        name=_string_or_default(payload.get("name"), defaults.name),
        workers=_int_or_default(payload.get("workers"), defaults.workers),
        patience=_int_or_default(payload.get("patience"), defaults.patience),
        cache=_bool_or_default(payload.get("cache"), defaults.cache),
        pretrained=_bool_or_default(payload.get("pretrained"), defaults.pretrained),
        seat_model_id=resolved_seat_model_id,
        preprocess=(
            _parse_preprocess_config(
                payload.get("preprocess"),
                scope=f"{scope}.preprocess",
            )
            if payload.get("preprocess") is not None
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
    inspection_payload: dict[str, Any],
    seat_model_id: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """从配置中定位 yolo_training payload 及最终使用的 seat_model_id。"""
    top_level_training = inspection_payload.get("yolo_training")
    seat_models = _ensure_list(
        inspection_payload.get("seat_models") or [],
        "InspectionConfig.seat_models",
    )
    if not seat_models:
        return _optional_dict(top_level_training, "InspectionConfig.yolo_training"), seat_model_id

    effective_id = seat_model_id or _optional_string(inspection_payload.get("default_seat_model_id"))
    selected = _select_seat_model_payload(seat_models, effective_id)
    if selected is None:
        return _optional_dict(top_level_training, "InspectionConfig.yolo_training"), seat_model_id

    resolved_id = _optional_string(selected.get("seat_model_id"))
    if selected.get("yolo_training") is not None:
        return _expect_dict(selected["yolo_training"], "SeatModelConfig.yolo_training"), resolved_id
    return _optional_dict(top_level_training, "InspectionConfig.yolo_training"), resolved_id
