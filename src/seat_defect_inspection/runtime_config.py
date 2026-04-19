"""从 JSON 加载座椅缺陷检测项目配置。"""

from __future__ import annotations

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
    YoloTrainingConfig,
)
from .schemas import BoundingBox


def load_config(path: str) -> InspectionConfig:
    """加载缺陷检测主配置。"""
    config_path = Path(path).resolve()
    config_dir = config_path.parent
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    inspection_payload = payload.get("seat_defect_inspection", payload)
    if "cameras" not in inspection_payload:
        raise ValueError("缺陷检测配置必须包含 `cameras` 列表")

    return InspectionConfig(
        cameras=[
            _build_camera_config(camera_payload, config_dir)
            for camera_payload in inspection_payload.get("cameras", [])
        ],
        output_json_path=_resolve_local_path(
            config_dir,
            inspection_payload.get(
                "output_json_path",
                InspectionConfig.__dataclass_fields__["output_json_path"].default,
            ),
            force=True,
        ),
        debug_dir=_resolve_local_path(
            config_dir,
            inspection_payload.get(
                "debug_dir",
                InspectionConfig.__dataclass_fields__["debug_dir"].default,
            ),
            force=True,
        ),
        capture_dir=_resolve_local_path(
            config_dir,
            inspection_payload.get(
                "capture_dir",
                InspectionConfig.__dataclass_fields__["capture_dir"].default,
            ),
            force=True,
        ),
        save_debug_artifacts=bool(
            inspection_payload.get(
                "save_debug_artifacts",
                InspectionConfig.__dataclass_fields__["save_debug_artifacts"].default,
            ),
        ),
        capture_retries=int(
            inspection_payload.get(
                "capture_retries",
                InspectionConfig.__dataclass_fields__["capture_retries"].default,
            ),
        ),
        part_id=inspection_payload.get(
            "part_id",
            InspectionConfig.__dataclass_fields__["part_id"].default,
        ),
        fusion=FusionConfig(**(inspection_payload.get("fusion") or {})),
    )


def load_yolo_training_config(path: str) -> YoloTrainingConfig:
    """加载 YOLO 训练配置。"""
    config_path = Path(path).resolve()
    config_dir = config_path.parent
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    training_payload = payload.get("yolo_training")
    if training_payload is None:
        raise ValueError("配置文件缺少 `yolo_training` 配置块")
    normalized = dict(training_payload)
    normalized["data_config_path"] = _resolve_local_path(
        config_dir,
        normalized["data_config_path"],
        force=True,
    )
    normalized["project"] = _resolve_local_path(
        config_dir,
        normalized.get("project", YoloTrainingConfig.__dataclass_fields__["project"].default),
        force=True,
    )
    if "model_path" in normalized:
        normalized["model_path"] = _resolve_optional_model_path(
            config_dir,
            normalized["model_path"],
        )
    return YoloTrainingConfig(**normalized)


def _build_camera_config(payload: dict[str, Any], config_dir: Path) -> CameraConfig:
    detection_payload = dict(payload.get("detection") or {})
    fallback_box_payload = detection_payload.pop("fallback_box", None)
    detection_model_path = detection_payload.pop("model_path", None)

    roi_payload = dict(payload.get("roi") or {})
    alignment_payload = dict(roi_payload.pop("alignment", {}) or {})
    template_image_path = alignment_payload.pop("template_image_path", None)

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
        quality=QualityGuardConfig(**(payload.get("quality") or {})),
        preprocess=PreprocessConfig(**(payload.get("preprocess") or {})),
        detection=DetectionConfig(
            **detection_payload,
            model_path=_resolve_optional_model_path(
                config_dir,
                detection_model_path,
            ),
            fallback_box=(
                _build_box(fallback_box_payload)
                if fallback_box_payload is not None
                else None
            ),
        ),
        roi=RoiRefineConfig(
            **roi_payload,
            alignment=AlignmentConfig(
                **alignment_payload,
                template_image_path=(
                    _resolve_local_path(
                        config_dir,
                        template_image_path,
                        force=True,
                    )
                    if template_image_path
                    else None
                ),
            ),
        ),
        patchcore=PatchCoreConfig(**(payload.get("patchcore") or {})),
        color_branch=ColorBranchConfig(**(payload.get("color_branch") or {})),
    )


def _build_box(payload: dict[str, Any]) -> BoundingBox:
    return BoundingBox(
        x1=float(payload["x1"]),
        y1=float(payload["y1"]),
        x2=float(payload["x2"]),
        y2=float(payload["y2"]),
    )


def _resolve_source_path(config_dir: Path, value: str) -> str:
    if "://" in value or value.isdigit():
        return value
    return _resolve_local_path(config_dir, value, force=True)


def _resolve_optional_model_path(config_dir: Path, value: str | None) -> str | None:
    if value is None:
        return None
    return _resolve_local_path(config_dir, value, force=False)


def _resolve_local_path(config_dir: Path, value: str, *, force: bool) -> str:
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    if not force and not _looks_like_local_path(value):
        return value
    return str((config_dir / candidate).resolve())


def _looks_like_local_path(value: str) -> bool:
    return value.startswith(".") or os.sep in value or (os.altsep is not None and os.altsep in value)
