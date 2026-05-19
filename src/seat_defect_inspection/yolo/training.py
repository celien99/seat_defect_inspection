"""YOLO training entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..config import YoloTrainingConfig
from .dataset_validation import (
    _prepare_training_dataset,
)

YOLO_SEGMENT_TASK = "segment"
YOLO_SEGMENT_MODEL = "yolo11m-seg.pt"
YOLO_SEGMENT_MODEL_YAML = "yolo11m-seg.yaml"
MODEL_INIT_NETWORK_ERROR_TOKENS = (
    "connection aborted",
    "connection error",
    "connection refused",
    "connection reset",
    "failed to establish a new connection",
    "max retries exceeded",
    "name or service not known",
    "network is unreachable",
    "nodename nor servname provided",
    "read timed out",
    "temporary failure in name resolution",
    "timed out",
    "urlopen error",
)


def train_yolo_model(config: YoloTrainingConfig) -> Dict[str, Any]:
    """Train a YOLO segmentation model and return a summary."""
    YOLO = _import_ultralytics_yolo()

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
        cache=config.cache,
        pretrained=bool(effective_pretrained),
        amp=bool(config.amp),
        optimizer=str(config.optimizer),
        lr0=float(config.lr0),
        lrf=float(config.lrf),
        momentum=float(config.momentum),
        weight_decay=float(config.weight_decay),
        warmup_epochs=float(config.warmup_epochs),
        warmup_momentum=float(config.warmup_momentum),
        warmup_bias_lr=float(config.warmup_bias_lr),
        mixup=float(config.mixup),
        copy_paste=float(config.copy_paste),
        degrees=float(config.degrees),
        translate=float(config.translate),
        scale=float(config.scale),
        shear=float(config.shear),
        perspective=float(config.perspective),
        flipud=float(config.flipud),
        fliplr=float(config.fliplr),
        rect=config.rect,
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


def _load_yolo_model(config: YoloTrainingConfig, yolo_cls) -> Tuple[Any, str, bool]:
    """Load the constrained segmentation model, with an offline-safe fallback."""
    requested_model_path = str(config.model_path)
    candidate = Path(requested_model_path)
    try:
        model = yolo_cls(requested_model_path, task=YOLO_SEGMENT_TASK)
        resolved_model_path = requested_model_path
        effective_pretrained = bool(config.pretrained)
    except Exception as exc:
        if _should_fallback_to_yaml(candidate, exc):
            model = yolo_cls(YOLO_SEGMENT_MODEL_YAML, task=YOLO_SEGMENT_TASK)
            resolved_model_path = YOLO_SEGMENT_MODEL_YAML
            effective_pretrained = False
        elif _is_model_download_error(exc):
            raise RuntimeError(
                "YOLO 模型初始化失败，当前项目固定使用 yolo11m-seg.pt。"
                " 离线环境请提供本地 yolo11m-seg.pt，或直接传入可用的 segmentation checkpoint。"
            ) from exc
        else:
            raise

    _validate_yolo_model_task(model, resolved_model_path)
    return model, resolved_model_path, effective_pretrained


def _import_ultralytics_yolo():
    """延迟导入 Ultralytics，避免非训练流程加载训练依赖。"""
    from ultralytics import YOLO

    return YOLO


def _should_fallback_to_yaml(candidate: Path, exc: Exception) -> bool:
    """Fallback only for the default checkpoint when download-like failures occur."""
    return (
        _is_default_model_request(candidate)
        and not candidate.exists()
        and _is_model_download_error(exc)
    )


def _is_default_model_request(candidate: Path) -> bool:
    """Whether the request targets the project's default downloadable checkpoint."""
    return (
        candidate.suffix.lower() == ".pt"
        and not candidate.is_absolute()
        and candidate.parent == Path(".")
        and candidate.name == YOLO_SEGMENT_MODEL
    )


def _is_model_download_error(exc: Exception) -> bool:
    """Best-effort detection for network failures raised during model download/init."""
    for current in _iter_exception_chain(exc):
        if isinstance(current, (ConnectionError, TimeoutError)):
            return True
        module_name = type(current).__module__.lower()
        class_name = type(current).__name__.lower()
        if module_name.startswith(("requests", "urllib", "socket", "httpx")):
            return True
        if class_name in {"httperror", "urlerror", "connecttimeout", "readtimeout"}:
            return True

        message = str(current).strip().lower()
        if any(token in message for token in MODEL_INIT_NETWORK_ERROR_TOKENS):
            return True
    return False


def _iter_exception_chain(exc: BaseException) -> List[BaseException]:
    """Flatten an exception and its causes/contexts for error classification."""
    chain: List[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _validate_yolo_model_task(model: Any, model_path: str) -> None:
    """Ensure the loaded model is a segmentation model before training."""
    task = str(getattr(model, "task", "")).strip().lower()
    if task == YOLO_SEGMENT_TASK:
        return
    display_task = task or "unknown"
    raise ValueError(
        "当前项目只支持 YOLO segmentation 模型，"
        f"但 `{model_path}` 的任务类型是 `{display_task}`。"
        " 请改用 yolo11m-seg.pt 或分割训练产物。"
    )
