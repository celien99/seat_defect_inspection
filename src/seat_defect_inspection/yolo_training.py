"""YOLO 训练入口。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import YoloTrainingConfig


def train_yolo_model(config: YoloTrainingConfig) -> dict[str, Any]:
    """使用给定数据集配置训练 YOLO 并输出摘要。"""
    from ultralytics import YOLO

    data_config_path = Path(config.data_config_path)
    if not data_config_path.exists():
        raise FileNotFoundError(f"YOLO 数据集配置不存在：{data_config_path}")

    model = YOLO(config.model_path)
    results = model.train(
        data=str(data_config_path),
        epochs=int(config.epochs),
        imgsz=int(config.imgsz),
        batch=int(config.batch),
        device=config.device,
        project=config.project,
        name=config.name,
        workers=int(config.workers),
        patience=int(config.patience),
        cache=bool(config.cache),
        pretrained=bool(config.pretrained),
    )

    save_dir = Path(getattr(results, "save_dir", Path(config.project) / config.name))
    summary = {
        "model_path": config.model_path,
        "data_config_path": str(data_config_path),
        "epochs": int(config.epochs),
        "imgsz": int(config.imgsz),
        "batch": int(config.batch),
        "device": config.device,
        "save_dir": str(save_dir),
        "best_weights_path": str(save_dir / "weights" / "best.pt"),
        "last_weights_path": str(save_dir / "weights" / "last.pt"),
    }
    summary_path = save_dir / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
