"""Core inspect runtime configuration parsing."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

from .config import (
    AlignmentConfig,
    CameraConfig,
    ColorBranchConfig,
    DetectionConfig,
    FusionConfig,
    InspectionConfig,
    PatchCoreConfig,
    QualityGuardConfig,
    RegionConfig,
    RoiRefineConfig,
    SeatModelConfig,
)
_LOCAL_PATH_SUFFIXES = {
    ".pt",
    ".pth",
    ".onnx",
    ".yaml",
    ".yml",
    ".json",
    ".png",
    ".jpg",
    ".jpeg",
}


# 主配置与座椅型号配置。
def _parse_inspection_config(payload: dict[str, Any], config_dir: Path) -> InspectionConfig:
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
        debug_artifacts_enabled=_bool_or_default(
            payload.get("debug_artifacts_enabled"),
            defaults.debug_artifacts_enabled,
        ),
        part_id=_string_or_default(payload.get("part_id"), defaults.part_id),
        fusion=_parse_fusion_config(
            payload.get("fusion"),
            scope=f"{scope}.fusion",
        ),
    )


def _parse_seat_model_config(
    payload: dict[str, Any],
    config_dir: Path,
    *,
    scope: str,
) -> SeatModelConfig:
    payload = _expect_dict(payload, scope)
    _reject_unknown_keys(payload, _field_names(SeatModelConfig), scope)

    seat_model_id = _require_string(payload, "seat_model_id", scope)
    cameras = _parse_camera_list(
        payload.get("cameras"),
        config_dir,
        scope=f"{scope}.cameras",
    )

    return SeatModelConfig(
        seat_model_id=seat_model_id,
        cameras=cameras,
        display_name=_optional_string(payload.get("display_name")),
    )


def _parse_camera_list(
    payload: Any,
    config_dir: Path,
    *,
    scope: str,
) -> list[CameraConfig]:
    items = _ensure_list(payload or [], scope)
    return [
        _parse_camera_config(item, config_dir, scope=f"{scope}[{index}]")
        for index, item in enumerate(items)
    ]


# 单机位及其子配置。
def _parse_camera_config(payload: dict[str, Any], config_dir: Path, *, scope: str) -> CameraConfig:
    payload = _expect_dict(payload, scope)
    config_scope = f"CameraConfig {scope}"
    _reject_unknown_keys(payload, _field_names(CameraConfig), config_scope)

    return CameraConfig(
        camera_id=_require_string(payload, "camera_id", scope),
        patchcore_model_path=_resolve_local_path(
            config_dir,
            _require_string(payload, "patchcore_model_path", scope),
            force=True,
        ),
        source=_resolve_source_path(
            config_dir,
            _string_or_default(payload.get("source"), ""),
        ),
        enabled=_bool_or_default(payload.get("enabled"), True),
        color_insensitive_mode=_bool_or_default(payload.get("color_insensitive_mode"), False),
        quality=_parse_quality_guard_config(
            payload.get("quality"),
            scope=f"{scope}.quality",
        ),
        detection=_parse_detection_config(
            payload.get("detection"),
            config_dir,
            scope=f"{scope}.detection",
        ),
        roi=_parse_roi_refine_config(
            payload.get("roi"),
            scope=f"{scope}.roi",
        ),
        patchcore=_parse_patchcore_config(
            payload.get("patchcore"),
            config_dir,
            scope=f"{scope}.patchcore",
        ),
        color_branch=_parse_color_branch_config(
            payload.get("color_branch"),
            scope=f"{scope}.color_branch",
        ),
        regions=_parse_region_configs(
            payload.get("regions"),
            config_dir,
            scope=f"{scope}.regions",
        ),
    )


def _parse_fusion_config(payload: Any, *, scope: str) -> FusionConfig:
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
        defect_overrides_reject=_bool_or_default(
            payload.get("defect_overrides_reject"),
            defaults.defect_overrides_reject,
        ),
    )


def _parse_quality_guard_config(payload: Any, *, scope: str) -> QualityGuardConfig:
    defaults = QualityGuardConfig()
    if payload is None:
        return defaults
    payload = _expect_dict(payload, scope)
    _reject_unknown_keys(payload, _field_names(QualityGuardConfig), scope)
    return QualityGuardConfig(
        min_laplacian_variance=_float_or_default(
            payload.get("min_laplacian_variance"),
            defaults.min_laplacian_variance,
        ),
        min_brightness_mean=_float_or_default(
            payload.get("min_brightness_mean"),
            defaults.min_brightness_mean,
        ),
        max_brightness_mean=_float_or_default(
            payload.get("max_brightness_mean"),
            defaults.max_brightness_mean,
        ),
        max_overexposed_ratio=_float_or_default(
            payload.get("max_overexposed_ratio"),
            defaults.max_overexposed_ratio,
        ),
        max_underexposed_ratio=_float_or_default(
            payload.get("max_underexposed_ratio"),
            defaults.max_underexposed_ratio,
        ),
    )


def _parse_alignment_config(payload: Any, *, scope: str) -> AlignmentConfig:
    defaults = AlignmentConfig()
    if payload is None:
        return defaults
    payload = _expect_dict(payload, scope)
    _reject_unknown_keys(payload, _field_names(AlignmentConfig), scope)
    return AlignmentConfig(
        output_width=_int_or_default(payload.get("output_width"), defaults.output_width),
        output_height=_int_or_default(payload.get("output_height"), defaults.output_height),
    )


def _parse_roi_refine_config(payload: Any, *, scope: str) -> RoiRefineConfig:
    defaults = RoiRefineConfig()
    if payload is None:
        return defaults
    payload = _expect_dict(payload, scope)
    _reject_unknown_keys(payload, _field_names(RoiRefineConfig), scope)
    return RoiRefineConfig(
        crop_expand_ratio=_float_or_default(
            payload.get("crop_expand_ratio"),
            defaults.crop_expand_ratio,
        ),
        crop_shrink_ratio=_float_or_default(
            payload.get("crop_shrink_ratio"),
            defaults.crop_shrink_ratio,
        ),
        mask_erode_pixels=_int_or_default(
            payload.get("mask_erode_pixels"),
            defaults.mask_erode_pixels,
        ),
        edge_ignore_pixels=_int_or_default(
            payload.get("edge_ignore_pixels"),
            defaults.edge_ignore_pixels,
        ),
        alignment=_parse_alignment_config(
            payload.get("alignment"),
            scope=f"{scope}.alignment",
        ),
    )


def _parse_detection_config(payload: Any, config_dir: Path, *, scope: str) -> DetectionConfig:
    defaults = DetectionConfig()
    if payload is None:
        return defaults
    payload = _expect_dict(payload, scope)
    _reject_unknown_keys(payload, _field_names(DetectionConfig), scope)
    return DetectionConfig(
        model_path=_resolve_optional_model_path(
            config_dir,
            _optional_string(payload.get("model_path")),
        ),
        target_class=_string_or_default(payload.get("target_class"), defaults.target_class),
        confidence=_float_or_default(payload.get("confidence"), defaults.confidence),
        iou=_float_or_default(payload.get("iou"), defaults.iou),
        device=_string_or_default(payload.get("device"), defaults.device),
    )


def _parse_patchcore_config(payload: Any, config_dir: Path, *, scope: str) -> PatchCoreConfig:
    defaults = PatchCoreConfig()
    if payload is None:
        return defaults
    payload = _expect_dict(payload, scope)
    _reject_unknown_keys(payload, _field_names(PatchCoreConfig), scope)
    return PatchCoreConfig(
        backend=_string_or_default(payload.get("backend"), defaults.backend),
        image_size=_int_or_default(payload.get("image_size"), defaults.image_size),
        patch_size=_int_or_default(payload.get("patch_size"), defaults.patch_size),
        stride=_int_or_default(payload.get("stride"), defaults.stride),
        max_memory=_int_or_default(payload.get("max_memory"), defaults.max_memory),
        threshold_quantile=_float_or_default(
            payload.get("threshold_quantile"),
            defaults.threshold_quantile,
        ),
        texture_input=_string_or_default(payload.get("texture_input"), defaults.texture_input),
        min_target_coverage=_float_or_default(
            payload.get("min_target_coverage"),
            defaults.min_target_coverage,
        ),
        max_ignore_overlap=_float_or_default(
            payload.get("max_ignore_overlap"),
            defaults.max_ignore_overlap,
        ),
        min_valid_patch_ratio=_float_or_default(
            payload.get("min_valid_patch_ratio"),
            defaults.min_valid_patch_ratio,
        ),
        decision_score_margin=_float_or_default(
            payload.get("decision_score_margin"),
            defaults.decision_score_margin,
        ),
        strong_patch_score_ratio=_float_or_default(
            payload.get("strong_patch_score_ratio"),
            defaults.strong_patch_score_ratio,
        ),
        min_strong_patch_count=_int_or_default(
            payload.get("min_strong_patch_count"),
            defaults.min_strong_patch_count,
        ),
        min_strong_component_count=_int_or_default(
            payload.get("min_strong_component_count"),
            defaults.min_strong_component_count,
        ),
        min_strong_patch_ratio=_float_or_default(
            payload.get("min_strong_patch_ratio"),
            defaults.min_strong_patch_ratio,
        ),
        min_strong_component_ratio=_float_or_default(
            payload.get("min_strong_component_ratio"),
            defaults.min_strong_component_ratio,
        ),
        critical_score_margin=_float_or_default(
            payload.get("critical_score_margin"),
            defaults.critical_score_margin,
        ),
        critical_peak_score_margin=_float_or_default(
            payload.get("critical_peak_score_margin"),
            defaults.critical_peak_score_margin,
        ),
        critical_min_component_patch_count=_int_or_default(
            payload.get("critical_min_component_patch_count"),
            defaults.critical_min_component_patch_count,
        ),
        backbone_name=_string_or_default(payload.get("backbone_name"), defaults.backbone_name),
        feature_layers=_string_list(
            payload.get("feature_layers"),
            scope=f"{scope}.feature_layers",
            default=defaults.feature_layers,
        ),
        backbone_pretrained=_bool_or_default(
            payload.get("backbone_pretrained"),
            defaults.backbone_pretrained,
        ),
        backbone_weights_path=_resolve_optional_local_path(
            config_dir,
            _optional_string(payload.get("backbone_weights_path")),
        ),
        backbone_device=_string_or_default(
            payload.get("backbone_device"),
            defaults.backbone_device,
        ),
        feature_pool_kernel_size=_int_or_default(
            payload.get("feature_pool_kernel_size"),
            defaults.feature_pool_kernel_size,
        ),
        coreset_sampling_ratio=_float_or_default(
            payload.get("coreset_sampling_ratio"),
            defaults.coreset_sampling_ratio,
        ),
    )


def _parse_color_branch_config(payload: Any, *, scope: str) -> ColorBranchConfig:
    defaults = ColorBranchConfig()
    if payload is None:
        return defaults
    payload = _expect_dict(payload, scope)
    _reject_unknown_keys(payload, _field_names(ColorBranchConfig), scope)
    return ColorBranchConfig(
        enabled=_bool_or_default(payload.get("enabled"), defaults.enabled),
        threshold_quantile=_float_or_default(
            payload.get("threshold_quantile"),
            defaults.threshold_quantile,
        ),
        threshold=_optional_float(payload.get("threshold")),
        min_valid_pixel_ratio=_float_or_default(
            payload.get("min_valid_pixel_ratio"),
            defaults.min_valid_pixel_ratio,
        ),
    )


def _parse_region_configs(
    payload: Any,
    config_dir: Path,
    *,
    scope: str,
) -> list[RegionConfig]:
    if payload is None:
        return []
    return [
        _parse_region_config(item, config_dir, scope=f"{scope}[{index}]")
        for index, item in enumerate(_ensure_list(payload, scope))
    ]


def _parse_region_config(
    payload: Any,
    config_dir: Path,
    *,
    scope: str,
) -> RegionConfig:
    payload = _expect_dict(payload, scope)
    _reject_unknown_keys(payload, _field_names(RegionConfig), scope)
    return RegionConfig(
        region_id=_require_string(payload, "region_id", scope),
        box=_region_box(payload.get("box"), scope=f"{scope}.box"),
        patchcore_model_path=_resolve_local_path(
            config_dir,
            _require_string(payload, "patchcore_model_path", scope),
            force=True,
        ),
        enabled=_bool_or_default(payload.get("enabled"), True),
        patchcore=(
            _parse_patchcore_config(
                payload.get("patchcore"),
                config_dir,
                scope=f"{scope}.patchcore",
            )
            if payload.get("patchcore") is not None
            else None
        ),
    )


def _region_box(value: Any, *, scope: str) -> list[float]:
    items = [float(item) for item in _ensure_list(value, scope)]
    if len(items) != 4:
        raise ValueError(f"{scope} 必须包含 4 个归一化坐标")
    x1, y1, x2, y2 = items
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"{scope} 必须满足 0 <= x1 < x2 <= 1 且 0 <= y1 < y2 <= 1")
    return items


def _select_seat_model_payload(
    seat_models: list[dict[str, Any]],
    seat_model_id: str | None,
) -> dict[str, Any] | None:
    if not seat_models:
        return None
    if seat_model_id is None:
        return seat_models[0]
    for item in seat_models:
        if item.get("seat_model_id") == seat_model_id:
            return item
    available = ", ".join(str(item.get("seat_model_id")) for item in seat_models)
    raise ValueError(f"未知 seat_model_id `{seat_model_id}`，可选值：{available}")


# 通用字段读取与路径解析工具。
def _is_missing(value: Any) -> bool:
    return value is None or value == ""


def _field_names(cls: type[Any]) -> set[str]:
    return {field.name for field in dataclasses.fields(cls)}


def _reject_unknown_keys(payload: dict[str, Any], allowed_keys: set[str], scope: str) -> None:
    unexpected = sorted(key for key in payload if key not in allowed_keys)
    if not unexpected:
        return
    formatted = ", ".join(f"`{key}`" for key in unexpected)
    raise ValueError(f"{scope} 包含未知字段: {formatted}")


def _expect_dict(value: Any, scope: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{scope} 必须是对象")
    return value


def _ensure_list(value: Any, scope: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{scope} 必须是数组")
    return value


def _require_key(payload: dict[str, Any], key: str, scope: str) -> Any:
    value = payload.get(key)
    if _is_missing(value):
        raise ValueError(f"{scope} 缺少 `{key}`")
    return value


def _require_string(payload: dict[str, Any], key: str, scope: str) -> str:
    return str(_require_key(payload, key, scope))


def _optional_string(value: Any) -> str | None:
    if _is_missing(value):
        return None
    return str(value)


def _string_or_default(value: Any, default: str) -> str:
    if _is_missing(value):
        return default
    return str(value)


def _bool_or_default(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise TypeError(f"布尔配置必须是 true/false，当前值: {value!r}")


def _int_or_default(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _float_or_default(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _optional_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    return float(value)


def _has_path_separator(value: str) -> bool:
    return os.sep in value or (os.altsep is not None and os.altsep in value)


def _string_list(value: Any, *, scope: str, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    return [str(item) for item in _ensure_list(value, scope)]


def _resolve_source_path(config_dir: Path, value: str) -> str:
    if _is_missing(value):
        return ""
    if "://" in value or value.isdigit():
        return value
    return _resolve_local_path(config_dir, value, force=True)


def _resolve_optional_model_path(config_dir: Path, value: str | None) -> str | None:
    if value is None:
        return None
    return _resolve_local_path(config_dir, value, force=False)


def _resolve_optional_local_path(config_dir: Path, value: str | None) -> str | None:
    if _is_missing(value):
        return None
    return _resolve_local_path(config_dir, value, force=True)


def _resolve_local_path(config_dir: Path, value: str, *, force: bool) -> str:
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    if not force and not _looks_like_local_path(value):
        return value
    return str((config_dir / candidate).resolve())


def _looks_like_local_path(value: str) -> bool:
    if value.startswith(".") or _has_path_separator(value):
        return True
    return Path(value).suffix.lower() in _LOCAL_PATH_SUFFIXES
