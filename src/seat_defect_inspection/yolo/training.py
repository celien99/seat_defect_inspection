"""YOLO 训练入口。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import YoloTrainingConfig
from .dataset_validation import _prepare_training_dataset

YOLO_SEGMENT_TASK = "segment"
YOLO_SEGMENT_MODEL = "yolo11m-seg.pt"
YOLO_SEGMENT_MODEL_YAML = "yolo11m-seg.yaml"


def train_yolo_model(config: YoloTrainingConfig) -> dict[str, Any]:
    """使用给定数据集配置训练 YOLO 并输出摘要。"""
    from ultralytics import YOLO

    data_config_path = Path(config.data_config_path)
    if not data_config_path.exists():
        raise FileNotFoundError(f"YOLO 数据集配置不存在：{data_config_path}")

    dataset_root, resolved_data_config_path = _prepare_training_dataset(
        data_config_path,
        Path(config.project),
    )

    model, resolved_model_path, effective_pretrained = _load_yolo_model(config, YOLO)
    results = model.train(
        data=str(resolved_data_config_path),
        task=YOLO_SEGMENT_TASK,
        epochs=int(config.epochs),
        imgsz=int(config.imgsz),
        batch=int(config.batch),
        device=config.device,
        project=config.project,
        name=config.name,
        workers=int(config.workers),
        patience=int(config.patience),
        cache=bool(config.cache),
        pretrained=bool(effective_pretrained),
    )

    save_dir = Path(getattr(results, "save_dir", Path(config.project) / config.name))
    summary = {
        "seat_model_id": config.seat_model_id,
        "requested_model_path": config.model_path,
        "resolved_model_path": resolved_model_path,
        "data_config_path": str(data_config_path),
        "resolved_data_config_path": str(resolved_data_config_path),
        "dataset_root": str(dataset_root),
        "task": YOLO_SEGMENT_TASK,
        "epochs": int(config.epochs),
        "imgsz": int(config.imgsz),
        "batch": int(config.batch),
        "device": config.device,
        "effective_pretrained": bool(effective_pretrained),
        "save_dir": str(save_dir),
        "best_weights_path": str(save_dir / "weights" / "best.pt"),
        "last_weights_path": str(save_dir / "weights" / "last.pt"),
    }
    summary_path = save_dir / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _load_yolo_model(config: YoloTrainingConfig, yolo_cls) -> tuple[Any, str, bool]:
    """按项目约束加载分割模型，离线时回退到 yaml 初始化。"""
    requested_model_path = str(config.model_path)
    try:
        model = yolo_cls(requested_model_path, task=YOLO_SEGMENT_TASK)
        resolved_model_path = requested_model_path
        effective_pretrained = bool(config.pretrained)
    except ConnectionError as exc:
        candidate = Path(requested_model_path)
        if (
            candidate.suffix.lower() != ".pt"
            or candidate.is_absolute()
            or candidate.parent != Path(".")
            or candidate.name != YOLO_SEGMENT_MODEL
        ):
            raise RuntimeError(
                "YOLO 模型初始化失败，当前项目固定使用 yolo11m-seg.pt。"
                " 离线环境请提供本地 yolo11m-seg.pt，或直接传入可用的 segmentation checkpoint。"
            ) from exc
        model = yolo_cls(YOLO_SEGMENT_MODEL_YAML, task=YOLO_SEGMENT_TASK)
        resolved_model_path = YOLO_SEGMENT_MODEL_YAML
        effective_pretrained = False

    _validate_yolo_model_task(model, resolved_model_path)
    return model, resolved_model_path, effective_pretrained


def _validate_yolo_model_task(model: Any, model_path: str) -> None:
    """训练前确认载入的确实是分割任务模型。"""
    task = str(getattr(model, "task", "")).strip().lower()
    if task == YOLO_SEGMENT_TASK:
        return
    display_task = task or "unknown"
    raise ValueError(
        "当前项目只支持 YOLO segmentation 模型，"
        f"但 `{model_path}` 的任务类型是 `{display_task}`。"
        " 请改用 yolo11m-seg.pt 或分割训练产物。"
    )
