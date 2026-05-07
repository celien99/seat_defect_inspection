"""YOLO training entrypoints."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import yaml

from seat_defect_core.preprocess import PreprocessEngine

from ..config import PreprocessConfig, YoloTrainingConfig
from .dataset_validation import (
    IMAGE_SUFFIXES,
    OPTIONAL_DATASET_SPLITS,
    REQUIRED_DATASET_SPLITS,
    _infer_label_dir,
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


def train_yolo_model(config: YoloTrainingConfig) -> dict[str, Any]:
    """Train a YOLO segmentation model and return a summary."""
    YOLO = _import_ultralytics_yolo()

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
    """Copy only the referenced dataset splits, then preprocess their images in place."""
    preprocess = config.preprocess
    if preprocess is None:
        return dataset_root, resolved_data_config_path

    _validate_yolo_preprocess(preprocess)
    resolved = yaml.safe_load(resolved_data_config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(resolved, dict):
        raise TypeError(f"YOLO 训练配置格式错误：{resolved_data_config_path}")

    target_root = _build_preprocessed_dataset_root(Path(config.project), config)
    target_root.mkdir(parents=True, exist_ok=False)
    engine = PreprocessEngine(preprocess)
    dataset_root_resolved = dataset_root.resolve()
    copied_relative_dirs: set[Path] = set()
    preprocessed_relative_dirs: set[Path] = set()

    try:
        for split_name in OPTIONAL_DATASET_SPLITS:
            split_value = resolved.get(split_name)
            if not split_value:
                continue

            split_path = Path(str(split_value))
            if not split_path.exists():
                if split_name in REQUIRED_DATASET_SPLITS:
                    raise ValueError(
                        f"当前 train-yolo 的 preprocess 仅支持目录型数据集切分，请检查 `{split_name}`：{split_path}"
                    )
                resolved.pop(split_name, None)
                continue
            if not split_path.is_dir():
                raise ValueError(
                    f"当前 train-yolo 的 preprocess 仅支持目录型数据集切分，请检查 `{split_name}`：{split_path}"
                )

            relative_split = _resolve_dataset_relative_dir(
                split_path,
                dataset_root_resolved=dataset_root_resolved,
                split_name=split_name,
            )
            target_split_dir = target_root / relative_split
            _copy_dataset_tree(
                split_path,
                target_split_dir,
                relative_dir=relative_split,
                copied_relative_dirs=copied_relative_dirs,
            )

            label_dir = _infer_label_dir(split_path, dataset_root)
            if label_dir.exists():
                if not label_dir.is_dir():
                    raise ValueError(f"YOLO 数据集 `{split_name}` 标签路径不是目录：{label_dir}")
                relative_label_dir = _resolve_dataset_relative_dir(
                    label_dir,
                    dataset_root_resolved=dataset_root_resolved,
                    split_name=split_name,
                )
                _copy_dataset_tree(
                    label_dir,
                    target_root / relative_label_dir,
                    relative_dir=relative_label_dir,
                    copied_relative_dirs=copied_relative_dirs,
                )
            elif split_name in REQUIRED_DATASET_SPLITS:
                raise FileNotFoundError(f"YOLO 数据集 `{split_name}` 缺少标签目录：{label_dir}")

            if relative_split not in preprocessed_relative_dirs:
                _preprocess_image_dir(target_split_dir, engine)
                preprocessed_relative_dirs.add(relative_split)
            resolved[split_name] = relative_split.as_posix()
    except Exception:
        shutil.rmtree(target_root, ignore_errors=True)
        raise

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
    """Build an isolated dataset directory for training-time preprocessing."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    seat_model = (config.seat_model_id or "default").replace(" ", "_")
    training_name = str(config.name).replace(" ", "_")
    target_root = project_root / "_preprocessed_yolo_dataset" / f"{seat_model}_{training_name}_{timestamp}"
    target_root.parent.mkdir(parents=True, exist_ok=True)
    return target_root


def _resolve_dataset_relative_dir(
    path: Path,
    *,
    dataset_root_resolved: Path,
    split_name: str,
) -> Path:
    """Resolve one dataset path and ensure it stays under dataset_root."""
    try:
        return path.resolve().relative_to(dataset_root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"YOLO 数据集 `{split_name}` 必须位于 dataset_root 内，当前路径：{path}"
        ) from exc


def _copy_dataset_tree(
    source_dir: Path,
    target_dir: Path,
    *,
    relative_dir: Path,
    copied_relative_dirs: set[Path],
) -> None:
    """Copy one dataset subtree into the preprocessing workspace."""
    if relative_dir in copied_relative_dirs:
        return
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir)
    copied_relative_dirs.add(relative_dir)


def _preprocess_image_dir(image_dir: Path, engine: PreprocessEngine) -> None:
    """Rewrite copied training images in place to mirror online preprocessing."""
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
    """Reject preprocessing that changes geometry without relabeling polygons."""
    if preprocess.camera_matrix or preprocess.distortion_coeffs:
        raise ValueError(
            "train-yolo 当前不支持带畸变矫正的 preprocess。"
            " 这会改变分割标注的几何位置，必须同步重算标签后才能训练。"
        )


def _load_yolo_model(config: YoloTrainingConfig, yolo_cls) -> tuple[Any, str, bool]:
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
    """Import Ultralytics only after redirecting its config dir into the project."""
    from seat_defect_core.yolo.detection import _ensure_local_yolo_config_dir

    _ensure_local_yolo_config_dir()

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


def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
    """Flatten an exception and its causes/contexts for error classification."""
    chain: list[BaseException] = []
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
