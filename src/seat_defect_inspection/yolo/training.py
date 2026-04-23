"""YOLO 训练入口。"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import yaml

from ..config import PreprocessConfig, YoloTrainingConfig
from ..preprocess import PreprocessEngine
from .dataset_validation import IMAGE_SUFFIXES, OPTIONAL_DATASET_SPLITS, _prepare_training_dataset

YOLO_SEGMENT_TASK = "segment"
YOLO_SEGMENT_MODEL = "yolo11m-seg.pt"
YOLO_SEGMENT_MODEL_YAML = "yolo11m-seg.yaml"


def train_yolo_model(config: YoloTrainingConfig) -> dict[str, Any]:
    """使用给定数据集配置训练 YOLO，并输出摘要。"""
    from ultralytics import YOLO

    data_config_path = Path(config.data_config_path)
    if not data_config_path.exists():
        raise FileNotFoundError(f"YOLO 数据集配置不存在：{data_config_path}")

    dataset_root, resolved_data_config_path = _prepare_training_dataset(
        data_config_path,
        Path(config.project),
    )
    effective_dataset_root = dataset_root
    effective_data_config_path = resolved_data_config_path
    if config.preprocess is not None:
        effective_dataset_root, effective_data_config_path = _prepare_preprocessed_dataset(
            dataset_root=dataset_root,
            resolved_data_config_path=resolved_data_config_path,
            config=config,
        )

    model, resolved_model_path, effective_pretrained = _load_yolo_model(config, YOLO)
    results = model.train(
        data=str(effective_data_config_path),
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
        "effective_data_config_path": str(effective_data_config_path),
        "dataset_root": str(dataset_root),
        "effective_dataset_root": str(effective_dataset_root),
        "preprocess_applied": config.preprocess is not None,
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


def _prepare_preprocessed_dataset(
    *,
    dataset_root: Path,
    resolved_data_config_path: Path,
    config: YoloTrainingConfig,
) -> tuple[Path, Path]:
    """复制一份训练数据集，并对图像执行与线上一致的 preprocess。"""
    preprocess = config.preprocess
    if preprocess is None:
        return dataset_root, resolved_data_config_path

    _validate_yolo_preprocess(preprocess)
    resolved = yaml.safe_load(resolved_data_config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(resolved, dict):
        raise TypeError(f"YOLO 训练配置格式错误：{resolved_data_config_path}")

    target_root = _build_preprocessed_dataset_root(Path(config.project), config)
    shutil.copytree(dataset_root, target_root)
    engine = PreprocessEngine(preprocess)

    dataset_root_resolved = dataset_root.resolve()
    for split_name in OPTIONAL_DATASET_SPLITS:
        split_value = resolved.get(split_name)
        if not split_value:
            continue

        split_path = Path(str(split_value))
        if not split_path.exists() or not split_path.is_dir():
            raise ValueError(
                "当前 train-yolo 的 preprocess 仅支持目录型数据集切分，"
                f"请检查 `{split_name}`：{split_path}"
            )
        try:
            relative_split = split_path.resolve().relative_to(dataset_root_resolved)
        except ValueError as exc:
            raise ValueError(
                f"YOLO 数据集 `{split_name}` 必须位于 dataset_root 内，当前路径：{split_path}"
            ) from exc

        target_split_dir = target_root / relative_split
        _preprocess_image_dir(target_split_dir, engine)
        resolved[split_name] = relative_split.as_posix()

    resolved["path"] = str(target_root)
    target_config_dir = Path(config.project) / "_resolved_dataset_configs"
    target_config_dir.mkdir(parents=True, exist_ok=True)
    target_config_path = target_config_dir / f"{resolved_data_config_path.stem}.{target_root.name}.yaml"
    target_config_path.write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return target_root, target_config_path


def _build_preprocessed_dataset_root(project_root: Path, config: YoloTrainingConfig) -> Path:
    """为 preprocess 后的数据集生成独立目录，避免覆盖原始训练集。"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    seat_model = (config.seat_model_id or "default").replace(" ", "_")
    training_name = str(config.name).replace(" ", "_")
    target_root = project_root / "_preprocessed_yolo_dataset" / f"{seat_model}_{training_name}_{timestamp}"
    target_root.parent.mkdir(parents=True, exist_ok=True)
    return target_root


def _preprocess_image_dir(image_dir: Path, engine: PreprocessEngine) -> None:
    """原地重写复制后的训练图像，使训练分布与线上 preprocess 保持一致。"""
    for image_path in sorted(image_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"YOLO 训练图像读取失败：{image_path}")
        processed = engine.process(image)
        if not cv2.imwrite(str(image_path), processed):
            raise OSError(f"YOLO 训练图像写入失败：{image_path}")


def _validate_yolo_preprocess(preprocess: PreprocessConfig) -> None:
    """拦住会破坏标签几何关系的 preprocess 项。"""
    if preprocess.camera_matrix or preprocess.distortion_coeffs:
        raise ValueError(
            "train-yolo 当前不支持带畸变矫正的 preprocess。"
            " 这会改变分割标注几何位置，必须同步重算标签后才能训练。"
        )


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
