"""YOLO 数据集预检与标签校验。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
REQUIRED_DATASET_SPLITS = ("train", "val")
OPTIONAL_DATASET_SPLITS = ("train", "val", "test")
YOLO_SEGMENT_TASK = "segment"


def _prepare_training_dataset(
    data_config_path: Path,
    project_root: Path,
) -> Tuple[Path, Path]:
    """解析数据集配置，补齐绝对路径，并在训练前做一次轻量预检。"""
    loaded = yaml.safe_load(data_config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"YOLO 数据集配置格式错误：{data_config_path}")
    valid_class_ids = _extract_valid_class_ids(loaded.get("names"), data_config_path)
    if not valid_class_ids:
        raise ValueError(f"YOLO 数据集配置缺少有效的 `names`：{data_config_path}")
    explicit_task = str(loaded.get("task", "")).strip().lower()
    if explicit_task and explicit_task not in {"segment", "seg", "segmentation"}:
        raise ValueError(
            "当前项目固定使用 YOLO segmentation 训练，"
            f"数据集配置中的 task 只能是 `segment`，收到 `{loaded.get('task')}`。"
        )

    dataset_root = _resolve_path(data_config_path.parent, str(loaded.get("path", ".")))
    if not dataset_root.exists():
        raise FileNotFoundError(f"YOLO 数据集根目录不存在：{dataset_root}")
    resolved = dict(loaded)
    resolved["path"] = str(dataset_root)
    resolved["task"] = YOLO_SEGMENT_TASK

    for split_name in OPTIONAL_DATASET_SPLITS:
        split_value = loaded.get(split_name)
        if split_name in REQUIRED_DATASET_SPLITS and not split_value:
            raise ValueError(f"YOLO 数据集配置缺少 `{split_name}`：{data_config_path}")
        if not split_value:
            continue

        split_path = _resolve_path(dataset_root, str(split_value))
        resolved[split_name] = str(split_path)
        if split_name in REQUIRED_DATASET_SPLITS:
            _validate_dataset_split(
                split_name,
                split_path,
                dataset_root,
                valid_class_ids=valid_class_ids,
            )

    target_dir = project_root / "_resolved_dataset_configs"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{data_config_path.stem}.resolved.yaml"
    target_path.write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return dataset_root, target_path


def _extract_valid_class_ids(raw_names: Any, data_config_path: Path) -> Set[int]:
    """从 names 配置中提取合法类别 ID。"""
    if isinstance(raw_names, list):
        return set(range(len(raw_names)))
    if not isinstance(raw_names, dict):
        raise ValueError(f"YOLO 数据集配置缺少有效的 `names`：{data_config_path}")

    valid_class_ids: Set[int] = set()
    for raw_key in raw_names:
        try:
            class_id = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"YOLO 数据集配置 `names` 包含非法类别 id：{data_config_path}"
            ) from exc
        if class_id < 0:
            raise ValueError(f"YOLO 数据集配置 `names` 包含负数类别 id：{data_config_path}")
        valid_class_ids.add(class_id)
    return valid_class_ids


def _resolve_path(base_dir: Path, value: str) -> Path:
    """把数据集里的相对路径解析成绝对路径。"""
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _validate_dataset_split(
    split_name: str,
    image_dir: Path,
    dataset_root: Path,
    *,
    valid_class_ids: Set[int],
) -> None:
    """检查数据集切分目录、图片和标签是否齐全。"""
    if not image_dir.exists():
        raise FileNotFoundError(f"YOLO 数据集 `{split_name}` 路径不存在：{image_dir}")
    if not image_dir.is_dir():
        return
    if not _has_supported_file(image_dir, IMAGE_SUFFIXES):
        raise ValueError(f"YOLO 数据集 `{split_name}` 目录中没有图像：{image_dir}")

    label_dir = _infer_label_dir(image_dir, dataset_root)
    if not label_dir.exists():
        raise FileNotFoundError(f"YOLO 数据集 `{split_name}` 缺少标签目录：{label_dir}")
    label_files = sorted(path for path in label_dir.rglob("*.txt") if path.is_file())
    if not label_files:
        raise ValueError(f"YOLO 数据集 `{split_name}` 标签目录为空：{label_dir}")
    _validate_label_files(
        label_files,
        split_name=split_name,
        valid_class_ids=valid_class_ids,
    )


def _infer_label_dir(image_dir: Path, dataset_root: Path) -> Path:
    """优先按 images/ -> labels/ 规则推断标签目录。"""
    try:
        relative = image_dir.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        return image_dir.parent / "labels" / image_dir.name

    parts = list(relative.parts)
    if parts and parts[0] == "images":
        parts[0] = "labels"
        return dataset_root.joinpath(*parts)
    return image_dir.parent / "labels" / image_dir.name


def _has_supported_file(folder: Path, suffixes: Set[str]) -> bool:
    """判断目录中是否至少有一张受支持的图片。"""
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            return True
    return False


def _validate_label_files(
    label_files: List[Path],
    *,
    split_name: str,
    valid_class_ids: Set[int],
) -> None:
    """抽样检查一批标签文件格式，尽量在训练前就暴露坏数据。"""
    inspected = 0
    for label_file in label_files:
        content = label_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        inspected += 1
        for line in content.splitlines():
            _validate_label_line(
                line,
                label_file=label_file,
                split_name=split_name,
                valid_class_ids=valid_class_ids,
            )
        if inspected >= 20:
            return
    if inspected > 0:
        return
    raise ValueError(
        f"YOLO 数据集 `{split_name}` 没有找到可用于分割训练的非空标签。"
        " 当前项目固定使用 yolo11m-seg.pt，请确认 labels 为分割多边形格式。"
    )


def _validate_label_line(
    line: str,
    *,
    label_file: Path,
    split_name: str,
    valid_class_ids: Set[int],
) -> None:
    """校验单行标签是否满足当前项目固定使用的分割格式。"""
    tokens = line.split()
    if not tokens:
        return
    try:
        values = [float(item) for item in tokens]
    except ValueError as exc:
        raise ValueError(f"YOLO 标签文件存在非数字内容：{label_file}") from exc
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"YOLO 标签文件存在非有限数值：{label_file}")
    if values[0] < 0 or int(values[0]) != values[0]:
        raise ValueError(f"YOLO 标签文件类别 id 非法：{label_file}")
    class_id = int(values[0])
    if class_id not in valid_class_ids:
        raise ValueError(f"YOLO 标签文件类别 id 超出 `names` 范围：{label_file}")

    coord_count = len(values) - 1
    if coord_count < 6 or coord_count % 2 != 0:
        raise ValueError(
            f"YOLO 数据集 `{split_name}` 标签不是分割多边形格式：{label_file}"
        )
    for coordinate in values[1:]:
        if coordinate < 0.0 or coordinate > 1.0:
            raise ValueError(f"YOLO 标签文件坐标超出 [0, 1] 范围：{label_file}")
