"""工程层 JSON 输出工具。

检测报告导出统一复用 ``seat_defect_core.reporting``；本模块只保留采图 manifest。
"""

from __future__ import annotations

from pathlib import Path

from seat_defect_core.reporting import (
    export_inspection_report,
    resolve_inspection_archive_path,
)

from .schemas import CaptureRecord, CaptureSummary
from .util import write_json


def export_capture_manifest(summary: CaptureSummary) -> Path:
    """写出一次采图任务的 manifest。"""
    path = Path(summary.manifest_path)
    payload = {
        "part_id": summary.part_id,
        "run_id": summary.run_id,
        "output_dir": summary.output_dir,
        "seat_model_id": summary.seat_model_id,
        "capture_count": len(summary.records),
        "success_count": sum(1 for item in summary.records if item.status == "OK"),
        "failure_count": sum(1 for item in summary.records if item.status != "OK"),
        "records": [_capture_record_to_dict(item) for item in summary.records],
    }
    write_json(path, payload)
    return path


def _capture_record_to_dict(record: CaptureRecord) -> dict[str, str | None]:
    return {
        "camera_id": record.camera_id,
        "frame_id": record.frame_id,
        "part_id": record.part_id,
        "source": record.source,
        "source_kind": record.source_kind,
        "timestamp": record.timestamp,
        "status": record.status,
        "seat_model_id": record.seat_model_id,
        "reason": record.reason,
        "output_path": record.output_path,
        "train_good_path": record.train_good_path,
    }


__all__ = [
    "export_capture_manifest",
    "export_inspection_report",
    "resolve_inspection_archive_path",
]
