"""工程层采图数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field


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
    seat_model_id: str | None = None
    reason: str | None = None
    output_path: str | None = None
    train_good_path: str | None = None


@dataclass(slots=True)
class CaptureSummary:
    """一次采图任务的汇总结果。"""

    part_id: str
    run_id: str
    output_dir: str
    manifest_path: str
    seat_model_id: str | None = None
    records: list[CaptureRecord] = field(default_factory=list)


__all__ = [
    "CaptureRecord",
    "CaptureSummary",
]
