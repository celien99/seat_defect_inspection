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
    SeatModelConfig,
    YoloTrainingConfig,
)
from .schemas import BoundingBox


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


def load_yolo_training_config(path: str, seat_model_id: str | None = None) -> YoloTrainingConfig:
    """加载 YOLO 训练配置。"""
    config_path = Path(path).resolve()
    config_dir = config_path.parent
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    inspection_payload = payload.get("seat_defect_inspection", payload)

    selected_seat_model_id = seat_model_id
    training_payload = payload.get("yolo_training")

    if inspection_payload.get("seat_models"):
        selected_model_payload = _select_seat_model_payload(
            inspection_payload.get("seat_models") or [],
            selected_seat_model_id or inspection_payload.get("default_seat_model_id"),
        )
        if selected_model_payload is not None and selected_model_payload.get("yolo_training") is not None:
            training_payload = selected_model_payload["yolo_training"]
            selected_seat_model_id = selected_model_payload["seat_model_id"]
        elif training_payload is None and selected_model_payload is not None:
            selected_seat_model_id = selected_model_payload["seat_model_id"]

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
        normalized.get("project", YoloTrainingConfig.__dataclass_fields__["project"].default),
        force=True,
    )
    if "model_path" in normalized:
        normalized["model_path"] = _resolve_optional_model_path(
            config_dir,
            normalized["model_path"],
        )
    normalized["seat_model_id"] = seat_model_id
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
        color_insensitive_mode=bool(payload.get("color_insensitive_mode", False)),
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
