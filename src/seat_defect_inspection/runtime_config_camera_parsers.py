"""运行时相机子配置解析。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import (
    AlignmentConfig,
    CameraConfig,
    ColorBranchConfig,
    DetectionConfig,
    PatchCoreConfig,
    PreprocessConfig,
    QualityGuardConfig,
    RoiRefineConfig,
)
from .runtime_config_values import (
    _bool_or_default,
    _expect_dict,
    _field_names,
    _float_list,
    _float_matrix,
    _float_or_default,
    _int_or_default,
    _optional_float,
    _optional_int,
    _optional_string,
    _reject_unknown_keys,
    _require_key,
    _require_string,
    _resolve_local_path,
    _resolve_optional_local_path,
    _resolve_optional_model_path,
    _resolve_source_path,
    _string_list,
    _string_or_default,
)
from .schemas import BoundingBox


def _parse_camera_config(payload: dict[str, Any], config_dir: Path, *, scope: str) -> CameraConfig:
    """解析单机位总配置。"""
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
    """解析图像质量门控配置。"""
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
    """解析预处理配置。"""
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
    """解析 ROI 对齐配置。"""
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
    """解析 ROI 精修配置。"""
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
    """解析 YOLO 检测配置。"""
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
    """解析 PatchCore 配置。"""
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
    """解析颜色分支配置。"""
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


def _parse_bounding_box(payload: Any, *, scope: str) -> BoundingBox | None:
    """解析 fallback_box。"""
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
