"""项目内公共小工具。"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Optional

import cv2


def build_model_scoped_root(base_dir: Path, seat_model_id: Optional[str]) -> Path:
    """按型号构造输出根目录；单型号场景直接复用原目录。"""
    if seat_model_id is None:
        return base_dir
    return base_dir / seat_model_id


def select_patchcore_input(roi) -> Any:
    """统一返回 PatchCore 真正消费的图像，避免训练、推理、调试图脱节。"""
    return (
        roi.texture_ready_image
        if roi.texture_ready_image is not None
        else roi.aligned_roi_image
    )


def write_image(path: Path, image: Any) -> None:
    """把图像写到磁盘，失败时直接抛错。"""
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write image: {path}")


def write_json(path: Path, payload: Any) -> None:
    """把 JSON 文本写到磁盘，自动补齐父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def format_reason_counter(counter: Counter[str]) -> str:
    """把原因计数格式化成便于报错阅读的一行文本。"""
    if not counter:
        return "none"
    return ", ".join(f"{reason}={count}" for reason, count in sorted(counter.items()))
