"""Load core inspect runtime configuration from JSON or INI."""

from __future__ import annotations

from typing import List, Set

from .config import CameraConfig, InspectionConfig, PatchCoreConfig
from .config_file import load_inspection_payload
from .runtime_config_parsers import _parse_inspection_config

_SUPPORTED_PATCHCORE_BACKENDS = {"full"}


def load_config(path: str) -> InspectionConfig:
    """加载缺陷检测主配置。"""
    config_dir, inspection_payload = load_inspection_payload(path)
    config = _parse_inspection_config(inspection_payload, config_dir)
    validate_inspection_config(config)
    return config


def validate_inspection_config(config: InspectionConfig) -> None:
    """做整体验证，确保配置在进入主流程前就失败得足够早。"""
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


_validate_inspection_config = validate_inspection_config


def _validate_camera_configs(cameras: List[CameraConfig], *, scope: str) -> None:
    """检查机位 ID 冲突，并校验 PatchCore 后端约束。"""
    duplicates: Set[str] = set()
    seen: Set[str] = set()
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
        _validate_region_configs(camera, scope=scope)


def _validate_patchcore_config(camera: CameraConfig, *, scope: str) -> None:
    """校验 PatchCore 后端选择与权重配置是否匹配。"""
    _validate_patchcore_backend(
        camera.patchcore,
        scope=f"{scope} 中 camera `{camera.camera_id}`",
    )


def _validate_region_configs(camera: CameraConfig, *, scope: str) -> None:
    """校验单机位内局部区域配置。"""
    duplicates: Set[str] = set()
    seen: Set[str] = set()
    for region in camera.regions:
        if region.region_id in seen:
            duplicates.add(region.region_id)
        else:
            seen.add(region.region_id)
        if region.patchcore is not None:
            _validate_patchcore_backend(
                region.patchcore,
                scope=(
                    f"{scope} 中 camera `{camera.camera_id}` "
                    f"region `{region.region_id}`"
                ),
            )
    if duplicates:
        duplicated_ids = ", ".join(f"`{region_id}`" for region_id in sorted(duplicates))
        raise ValueError(f"{scope} 中 camera `{camera.camera_id}` 存在重复 region_id: {duplicated_ids}")


def _validate_patchcore_backend(config: PatchCoreConfig, *, scope: str) -> None:
    backend = config.backend.strip().lower()
    if backend not in _SUPPORTED_PATCHCORE_BACKENDS:
        supported = ", ".join(sorted(_SUPPORTED_PATCHCORE_BACKENDS))
        raise ValueError(
            f"{scope} 的 patchcore.backend `{config.backend}` 不受支持，可选值: {supported}"
        )
    if backend != "full":
        return
    if config.backbone_pretrained or config.backbone_weights_path:
        return
    raise ValueError(
        f"{scope} 配置了 patchcore.backend=full，"
        "但没有提供可用 backbone 权重。"
        " 请设置 patchcore.backbone_pretrained=true，"
        "或配置 patchcore.backbone_weights_path。"
    )
