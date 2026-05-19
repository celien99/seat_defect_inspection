"""工程层采图数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(slots=True)
class CaptureRecord:
    """一次采图命令中某个机位的落盘结果。"""

    camera_id: str
    frame_id: str
    part_id: str
    source: str
    source_kind: str
    timestamp: str
    status: str
    seat_model_id: Optional[str] = None
    reason: Optional[str] = None
    output_path: Optional[str] = None
    train_good_path: Optional[str] = None


@dataclass(slots=True)
class CaptureSummary:
    """一次采图任务的汇总结果。"""

    part_id: str
    run_id: str
    output_dir: str
    manifest_path: str
    seat_model_id: Optional[str] = None
    records: List[CaptureRecord] = field(default_factory=list)


__all__ = [
    "CaptureRecord",
    "CaptureSummary",
]
