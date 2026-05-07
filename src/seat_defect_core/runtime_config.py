"""Load SDK runtime configuration from JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import CameraConfig, InspectionConfig
from .runtime_config_parsers import _parse_inspection_config

_SUPPORTED_PATCHCORE_BACKENDS = {"full"}


def load_config(path: str) -> InspectionConfig:
    """加载缺陷检测主配置。"""
    config_dir, inspection_payload = _load_inspection_payload(path)
    config = _parse_inspection_config(inspection_payload, config_dir)
    _validate_inspection_config(config)
    return config


def _load_inspection_payload(path: str) -> tuple[Path, dict[str, Any]]:
    """读取配置文件并定位 seat_defect_inspection 顶层 payload。"""
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"配置文件顶层必须是对象：{config_path}")
    inspection_payload = payload.get("seat_defect_inspection", payload)
    if not isinstance(inspection_payload, dict):
        raise TypeError(f"`seat_defect_inspection` 必须是对象：{config_path}")
    return config_path.parent, inspection_payload


def _validate_inspection_config(config: InspectionConfig) -> None:
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


def _validate_camera_configs(cameras: list[CameraConfig], *, scope: str) -> None:
    """检查机位 ID 冲突，并校验 PatchCore 后端约束。"""
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
    """校验 PatchCore 后端选择与权重配置是否匹配。"""
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
        "或配置 patchcore.backbone_weights_path。"
    )
