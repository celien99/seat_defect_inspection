"""从 JSON 加载座椅缺陷检测项目配置。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from seat_defect_core.runtime_config import validate_inspection_config

from .config import InspectionConfig, YoloTrainingConfig
from .runtime_config_parsers import (
    _parse_inspection_config,
    _parse_yolo_training_config,
    _resolve_yolo_training_payload,
)


def load_config(path: str) -> InspectionConfig:
    """加载缺陷检测主配置。"""
    config_dir, inspection_payload = _load_inspection_payload(path)
    config = _parse_inspection_config(inspection_payload, config_dir)
    validate_inspection_config(config)
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
    """读取配置文件并定位 seat_defect_inspection 顶层 payload。"""
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"配置文件顶层必须是对象：{config_path}")
    inspection_payload = payload.get("seat_defect_inspection", payload)
    if not isinstance(inspection_payload, dict):
        raise TypeError(f"`seat_defect_inspection` 必须是对象：{config_path}")
    return config_path.parent, inspection_payload
