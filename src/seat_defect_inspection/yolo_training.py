"""YOLO 训练入口。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .config import YoloTrainingConfig

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
REQUIRED_DATASET_SPLITS = ("train", "val")
OPTIONAL_DATASET_SPLITS = ("train", "val", "test")


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


def _prepare_training_dataset(
    data_config_path: Path,
    project_root: Path,
) -> tuple[Path, Path]:
    loaded = yaml.safe_load(data_config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"YOLO 数据集配置格式错误：{data_config_path}")
    if not isinstance(loaded.get("names"), dict) or not loaded["names"]:
        raise ValueError(f"YOLO 数据集配置缺少有效的 `names`：{data_config_path}")

    dataset_root = _resolve_path(data_config_path.parent, str(loaded.get("path", ".")))
    if not dataset_root.exists():
        raise FileNotFoundError(f"YOLO 数据集根目录不存在：{dataset_root}")
    resolved = dict(loaded)
    resolved["path"] = str(dataset_root)

    for split_name in OPTIONAL_DATASET_SPLITS:
        split_value = loaded.get(split_name)
        if split_name in REQUIRED_DATASET_SPLITS and not split_value:
            raise ValueError(f"YOLO 数据集配置缺少 `{split_name}`：{data_config_path}")
        if not split_value:
            continue

        split_path = _resolve_path(dataset_root, str(split_value))
        resolved[split_name] = str(split_path)
        if split_name in REQUIRED_DATASET_SPLITS:
            _validate_dataset_split(split_name, split_path, dataset_root)

    target_dir = project_root / "_resolved_dataset_configs"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{data_config_path.stem}.resolved.yaml"
    target_path.write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return dataset_root, target_path


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _validate_dataset_split(
    split_name: str,
    image_dir: Path,
    dataset_root: Path,
) -> None:
    if not image_dir.exists():
        raise FileNotFoundError(f"YOLO 数据集 `{split_name}` 路径不存在：{image_dir}")
    if not image_dir.is_dir():
        return
    if not _has_supported_file(image_dir, IMAGE_SUFFIXES):
        raise ValueError(f"YOLO 数据集 `{split_name}` 目录中没有图像：{image_dir}")

    label_dir = _infer_label_dir(image_dir, dataset_root)
    if not label_dir.exists():
        raise FileNotFoundError(f"YOLO 数据集 `{split_name}` 缺少标签目录：{label_dir}")
    if not _has_supported_file(label_dir, {".txt"}):
        raise ValueError(f"YOLO 数据集 `{split_name}` 标签目录为空：{label_dir}")


def _infer_label_dir(image_dir: Path, dataset_root: Path) -> Path:
    try:
        relative = image_dir.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        return image_dir.parent / "labels" / image_dir.name

    parts = list(relative.parts)
    if parts and parts[0] == "images":
        parts[0] = "labels"
        return dataset_root.joinpath(*parts)
    return image_dir.parent / "labels" / image_dir.name


def _has_supported_file(folder: Path, suffixes: set[str]) -> bool:
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            return True
    return False


def _load_yolo_model(config: YoloTrainingConfig, yolo_cls) -> tuple[Any, str, bool]:
    requested_model_path = str(config.model_path)
    try:
        return yolo_cls(requested_model_path), requested_model_path, bool(config.pretrained)
    except ConnectionError as exc:
        candidate = Path(requested_model_path)
        if candidate.suffix.lower() != ".pt" or candidate.is_absolute() or candidate.parent != Path("."):
            raise RuntimeError(
                "YOLO 模型初始化失败，当前环境无法下载权重，且没有可用的本地架构 YAML 回退。"
            ) from exc
        yaml_candidate = f"{candidate.stem}.yaml"
        return yolo_cls(yaml_candidate), yaml_candidate, False
