"""从 JSON 加载座椅缺陷检测项目配置。"""

from __future__ import annotations

import dataclasses
import json
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
    PreprocessConfig,
    QualityGuardConfig,
    RoiRefineConfig,
    SeatModelConfig,
    YoloTrainingConfig,
)
from .cvops import resolve_debug_artifact_names
from .schemas import BoundingBox

_SUPPORTED_PATCHCORE_BACKENDS = {"full", "handcrafted"}


def load_config(path: str) -> InspectionConfig:
    """加载缺陷检测主配置。"""
    config_dir, inspection_payload = _load_inspection_payload(path)
    config = _parse_inspection_config(inspection_payload, config_dir)
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
    return _parse_yolo_training_config(
        training_payload,
        config_dir,
        scope="YoloTrainingConfig",
        seat_model_id=selected_seat_model_id,
    )


def _load_inspection_payload(path: str) -> tuple[Path, dict[str, Any]]:
    """加载缺陷检测主配置 payload。"""
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"配置文件顶层必须是对象：{config_path}")
    inspection_payload = payload.get("seat_defect_inspection", payload)
    if not isinstance(inspection_payload, dict):
        raise TypeError(f"`seat_defect_inspection` 必须是对象：{config_path}")
    return config_path.parent, inspection_payload


def _parse_inspection_config(payload: dict[str, Any], config_dir: Path) -> InspectionConfig:
    scope = "InspectionConfig"
    _reject_unknown_keys(payload, _field_names(InspectionConfig), scope)

    cameras_payload = payload.get("cameras") or []
    seat_models_payload = payload.get("seat_models") or []
    if not cameras_payload and not seat_models_payload:
        raise ValueError("缺陷检测配置必须包含 `cameras` 或 `seat_models`")

    cameras = [
        _parse_camera_config(item, config_dir, scope=f"{scope}.cameras[{index}]")
        for index, item in enumerate(_ensure_list(cameras_payload, f"{scope}.cameras"))
    ]
    seat_models = [
        _parse_seat_model_config(item, config_dir, scope=f"{scope}.seat_models[{index}]")
        for index, item in enumerate(_ensure_list(seat_models_payload, f"{scope}.seat_models"))
    ]

    defaults = InspectionConfig()
    default_seat_model_id = _optional_string(payload.get("default_seat_model_id"))
    if default_seat_model_id is None and seat_models:
        default_seat_model_id = seat_models[0].seat_model_id

    yolo_training_payload = payload.get("yolo_training")
    yolo_training = (
        _parse_yolo_training_config(
            _expect_dict(yolo_training_payload, f"{scope}.yolo_training"),
            config_dir,
            scope=f"{scope}.yolo_training",
        )
        if yolo_training_payload is not None
        else None
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
    payload = _expect_dict(payload, scope)
    _reject_unknown_keys(payload, _field_names(SeatModelConfig), scope)

    cameras = [
        _parse_camera_config(item, config_dir, scope=f"{scope}.cameras[{index}]")
        for index, item in enumerate(_ensure_list(payload.get("cameras") or [], f"{scope}.cameras"))
    ]
    seat_model_id = _require_string(payload, "seat_model_id", scope)
    yolo_training_payload = payload.get("yolo_training")
    yolo_training = (
        _parse_yolo_training_config(
            _expect_dict(yolo_training_payload, f"{scope}.yolo_training"),
            config_dir,
            scope=f"{scope}.yolo_training",
            seat_model_id=seat_model_id,
        )
        if yolo_training_payload is not None
        else None
    )
    return SeatModelConfig(
        seat_model_id=seat_model_id,
        cameras=cameras,
        display_name=_optional_string(payload.get("display_name")),
        yolo_training=yolo_training,
    )


def _parse_camera_config(payload: dict[str, Any], config_dir: Path, *, scope: str) -> CameraConfig:
    payload = _expect_dict(payload, scope)
    _reject_unknown_keys(payload, _field_names(CameraConfig), scope)

    return CameraConfig(
        camera_id=_require_string(payload, "camera_id", scope),
        source=_resolve_source_path(config_dir, _require_string(payload, "source", scope)),
        patchcore_model_path=_resolve_local_path(
            config_dir,
            _require_string(payload, "patchcore_model_path", scope),
            force=True,
        ),
        train_good_dir=_resolve_optional_local_path(
            config_dir,
            _optional_string(payload.get("train_good_dir")),
        ),
        enabled=_bool_or_default(payload.get("enabled"), True),
        color_insensitive_mode=_bool_or_default(payload.get("color_insensitive_mode"), False),
        quality=_parse_quality_guard_config(
            payload.get("quality"),
            scope=f"{scope}.quality",
        ),
        preprocess=_parse_preprocess_config(
            payload.get("preprocess"),
            scope=f"{scope}.preprocess",
        ),
        detection=_parse_detection_config(
            payload.get("detection"),
            config_dir,
            scope=f"{scope}.detection",
        ),
        roi=_parse_roi_refine_config(
            payload.get("roi"),
            config_dir,
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


def _parse_preprocess_config(payload: Any, *, scope: str) -> PreprocessConfig:
    defaults = PreprocessConfig()
    if payload is None:
        return defaults
    payload = _expect_dict(payload, scope)
    _reject_unknown_keys(payload, _field_names(PreprocessConfig), scope)
    return PreprocessConfig(
        resize_width=_optional_int(payload.get("resize_width")),
        resize_height=_optional_int(payload.get("resize_height")),
        denoise_method=_string_or_default(payload.get("denoise_method"), defaults.denoise_method),
        gaussian_kernel_size=_int_or_default(
            payload.get("gaussian_kernel_size"),
            defaults.gaussian_kernel_size,
        ),
        bilateral_diameter=_int_or_default(
            payload.get("bilateral_diameter"),
            defaults.bilateral_diameter,
        ),
        bilateral_sigma_color=_float_or_default(
            payload.get("bilateral_sigma_color"),
            defaults.bilateral_sigma_color,
        ),
        bilateral_sigma_space=_float_or_default(
            payload.get("bilateral_sigma_space"),
            defaults.bilateral_sigma_space,
        ),
        white_balance_method=_string_or_default(
            payload.get("white_balance_method"),
            defaults.white_balance_method,
        ),
        max_white_balance_gain=_float_or_default(
            payload.get("max_white_balance_gain"),
            defaults.max_white_balance_gain,
        ),
        apply_illumination_correction=_bool_or_default(
            payload.get("apply_illumination_correction"),
            defaults.apply_illumination_correction,
        ),
        illumination_blur_kernel_size=_int_or_default(
            payload.get("illumination_blur_kernel_size"),
            defaults.illumination_blur_kernel_size,
        ),
        illumination_strength=_float_or_default(
            payload.get("illumination_strength"),
            defaults.illumination_strength,
        ),
        apply_clahe=_bool_or_default(payload.get("apply_clahe"), defaults.apply_clahe),
        clahe_clip_limit=_float_or_default(
            payload.get("clahe_clip_limit"),
            defaults.clahe_clip_limit,
        ),
        clahe_tile_grid_size=_int_or_default(
            payload.get("clahe_tile_grid_size"),
            defaults.clahe_tile_grid_size,
        ),
        gamma=_optional_float(payload.get("gamma")),
        sharpen=_bool_or_default(payload.get("sharpen"), defaults.sharpen),
        sharpen_sigma=_float_or_default(payload.get("sharpen_sigma"), defaults.sharpen_sigma),
        sharpen_amount=_float_or_default(payload.get("sharpen_amount"), defaults.sharpen_amount),
        camera_matrix=_float_matrix(payload.get("camera_matrix"), scope=f"{scope}.camera_matrix"),
        distortion_coeffs=_float_list(
            payload.get("distortion_coeffs"),
            scope=f"{scope}.distortion_coeffs",
        ),
    )


def _parse_alignment_config(payload: Any, config_dir: Path, *, scope: str) -> AlignmentConfig:
    defaults = AlignmentConfig()
    if payload is None:
        return defaults
    payload = _expect_dict(payload, scope)
    _reject_unknown_keys(payload, _field_names(AlignmentConfig), scope)
    return AlignmentConfig(
        enabled=_bool_or_default(payload.get("enabled"), defaults.enabled),
        method=_string_or_default(payload.get("method"), defaults.method),
        template_image_path=_resolve_optional_local_path(
            config_dir,
            _optional_string(payload.get("template_image_path")),
        ),
        output_width=_int_or_default(payload.get("output_width"), defaults.output_width),
        output_height=_int_or_default(payload.get("output_height"), defaults.output_height),
        ecc_iterations=_int_or_default(payload.get("ecc_iterations"), defaults.ecc_iterations),
    )


def _parse_roi_refine_config(payload: Any, config_dir: Path, *, scope: str) -> RoiRefineConfig:
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
        mask_mode=_string_or_default(payload.get("mask_mode"), defaults.mask_mode),
        morphology_kernel_size=_int_or_default(
            payload.get("morphology_kernel_size"),
            defaults.morphology_kernel_size,
        ),
        ignore_dilate_kernel_size=_int_or_default(
            payload.get("ignore_dilate_kernel_size"),
            defaults.ignore_dilate_kernel_size,
        ),
        edge_ignore_pixels=_int_or_default(
            payload.get("edge_ignore_pixels"),
            defaults.edge_ignore_pixels,
        ),
        texture_denoise_method=_string_or_default(
            payload.get("texture_denoise_method"),
            defaults.texture_denoise_method,
        ),
        texture_gaussian_kernel_size=_int_or_default(
            payload.get("texture_gaussian_kernel_size"),
            defaults.texture_gaussian_kernel_size,
        ),
        texture_bilateral_diameter=_int_or_default(
            payload.get("texture_bilateral_diameter"),
            defaults.texture_bilateral_diameter,
        ),
        texture_bilateral_sigma_color=_float_or_default(
            payload.get("texture_bilateral_sigma_color"),
            defaults.texture_bilateral_sigma_color,
        ),
        texture_bilateral_sigma_space=_float_or_default(
            payload.get("texture_bilateral_sigma_space"),
            defaults.texture_bilateral_sigma_space,
        ),
        apply_texture_clahe=_bool_or_default(
            payload.get("apply_texture_clahe"),
            defaults.apply_texture_clahe,
        ),
        texture_clahe_clip_limit=_float_or_default(
            payload.get("texture_clahe_clip_limit"),
            defaults.texture_clahe_clip_limit,
        ),
        texture_clahe_tile_grid_size=_int_or_default(
            payload.get("texture_clahe_tile_grid_size"),
            defaults.texture_clahe_tile_grid_size,
        ),
        texture_illumination_correction=_bool_or_default(
            payload.get("texture_illumination_correction"),
            defaults.texture_illumination_correction,
        ),
        texture_illumination_blur_kernel_size=_int_or_default(
            payload.get("texture_illumination_blur_kernel_size"),
            defaults.texture_illumination_blur_kernel_size,
        ),
        texture_illumination_strength=_float_or_default(
            payload.get("texture_illumination_strength"),
            defaults.texture_illumination_strength,
        ),
        mask_feather_kernel_size=_int_or_default(
            payload.get("mask_feather_kernel_size"),
            defaults.mask_feather_kernel_size,
        ),
        edge_enhance_method=_string_or_default(
            payload.get("edge_enhance_method"),
            defaults.edge_enhance_method,
        ),
        edge_enhance_weight=_float_or_default(
            payload.get("edge_enhance_weight"),
            defaults.edge_enhance_weight,
        ),
        suppress_background=_bool_or_default(
            payload.get("suppress_background"),
            defaults.suppress_background,
        ),
        background_fill_mode=_string_or_default(
            payload.get("background_fill_mode"),
            defaults.background_fill_mode,
        ),
        background_blur_kernel_size=_int_or_default(
            payload.get("background_blur_kernel_size"),
            defaults.background_blur_kernel_size,
        ),
        safe_margin_erode_kernel_size=_int_or_default(
            payload.get("safe_margin_erode_kernel_size"),
            defaults.safe_margin_erode_kernel_size,
        ),
        alignment=_parse_alignment_config(
            payload.get("alignment"),
            config_dir,
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
        ignore_classes=_string_list(
            payload.get("ignore_classes"),
            scope=f"{scope}.ignore_classes",
            default=defaults.ignore_classes,
        ),
        confidence=_float_or_default(payload.get("confidence"), defaults.confidence),
        iou=_float_or_default(payload.get("iou"), defaults.iou),
        device=_string_or_default(payload.get("device"), defaults.device),
        fallback_box=_parse_bounding_box(
            payload.get("fallback_box"),
            scope=f"{scope}.fallback_box",
        ),
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
        early_stop_on_ng=_bool_or_default(
            payload.get("early_stop_on_ng"),
            defaults.early_stop_on_ng,
        ),
        defect_overrides_reject=_bool_or_default(
            payload.get("defect_overrides_reject"),
            defaults.defect_overrides_reject,
        ),
    )


def _parse_yolo_training_config(
    payload: dict[str, Any],
    config_dir: Path,
    *,
    scope: str,
    seat_model_id: str | None = None,
) -> YoloTrainingConfig:
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
    )


def _parse_bounding_box(payload: Any, *, scope: str) -> BoundingBox | None:
    if payload is None:
        return None
    payload = _expect_dict(payload, scope)
    _reject_unknown_keys(payload, _field_names(BoundingBox), scope)
    return BoundingBox(
        x1=float(_require_key(payload, "x1", scope)),
        y1=float(_require_key(payload, "y1", scope)),
        x2=float(_require_key(payload, "x2", scope)),
        y2=float(_require_key(payload, "y2", scope)),
    )


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
    duplicates: set[str] = set()
    seen: set[str] = set()
    for camera in cameras:
        if camera.camera_id in seen:
            duplicates.add(camera.camera_id)
        else:
            seen.add(camera.camera_id)
    if duplicates:
        duplicated_ids = ", ".join(f"`{camera_id}`" for camera_id in sorted(duplicates))
        raise ValueError(f"{scope} 存在重复 camera_id: {duplicated_ids}")

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


def _optional_dict(value: Any, scope: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _expect_dict(value, scope)


def _ensure_list(value: Any, scope: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{scope} 必须是数组")
    return value


def _require_key(payload: dict[str, Any], key: str, scope: str) -> Any:
    value = payload.get(key)
    if value is None or value == "":
        raise ValueError(f"{scope} 缺少 `{key}`")
    return value


def _require_string(payload: dict[str, Any], key: str, scope: str) -> str:
    return str(_require_key(payload, key, scope))


def _optional_string(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _string_or_default(value: Any, default: str) -> str:
    if value is None or value == "":
        return default
    return str(value)


def _bool_or_default(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


def _int_or_default(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _float_or_default(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _string_list(value: Any, *, scope: str, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    return [str(item) for item in _ensure_list(value, scope)]


def _float_list(value: Any, *, scope: str) -> list[float] | None:
    if value is None:
        return None
    return [float(item) for item in _ensure_list(value, scope)]


def _float_matrix(value: Any, *, scope: str) -> list[list[float]] | None:
    if value is None:
        return None
    rows = _ensure_list(value, scope)
    return [
        [float(item) for item in _ensure_list(row, f"{scope}[{index}]")]
        for index, row in enumerate(rows)
    ]


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
