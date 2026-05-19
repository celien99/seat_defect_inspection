"""工程层采图 manifest 输出工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from seat_defect_core.util import write_json

from .schemas import CaptureRecord, CaptureSummary


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


def _capture_record_to_dict(record: CaptureRecord) -> Dict[str, Optional[str]]:
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
]
