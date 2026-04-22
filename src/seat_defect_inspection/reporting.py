"""JSON 输出工具。"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .schemas import (
    BoundingBox,
    CameraInspectionResult,
    CaptureRecord,
    CaptureSummary,
    InspectionResult,
)
from .util import write_json


def export_inspection_report(result: InspectionResult, output_path: str) -> Path:
    """写出一次检测任务的结果 JSON。"""
    path = Path(output_path)
    payload = {
        "part_id": result.part_id,
        "frame_id": result.frame_id,
        "timestamp": result.timestamp,
        "status": result.status,
        "decision_reason": result.decision_reason,
        "seat_model_id": result.seat_model_id,
        "camera_results": [_camera_result_to_dict(item) for item in result.camera_results],
    }
    write_json(path, payload)
    # 固定路径继续保留为“最新结果”，同时按型号/工件/帧号归档，避免历史结果被覆盖。
    archive_path = _build_inspection_archive_path(path, result)
    write_json(archive_path, payload)
    return path


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


def _camera_result_to_dict(result: CameraInspectionResult) -> dict:
    return {
        "camera_id": result.camera_id,
        "frame_id": result.frame_id,
        "source": result.source,
        "source_kind": result.source_kind,
        "status": result.status,
        "reason": result.reason,
        "seat_model_id": result.seat_model_id,
        "quality": (
            {
                "accepted": result.quality.accepted,
                "reason": result.quality.reason,
                "metrics": {
                    "laplacian_variance": result.quality.metrics.laplacian_variance,
                    "brightness_mean": result.quality.metrics.brightness_mean,
                    "overexposed_ratio": result.quality.metrics.overexposed_ratio,
                    "underexposed_ratio": result.quality.metrics.underexposed_ratio,
                    "is_black_frame": result.quality.metrics.is_black_frame,
                    "is_white_frame": result.quality.metrics.is_white_frame,
                },
            }
            if result.quality is not None
            else None
        ),
        "target_box": _resolve_target_box(result),
        "crop_box": _box_to_dict(result.crop_box),
        "texture_result": (
            {
                "score": result.texture_result.score,
                "threshold": result.texture_result.threshold,
                "decision_threshold": result.texture_result.decision_threshold,
                "is_anomaly": result.texture_result.is_anomaly,
                "valid_patch_ratio": result.texture_result.valid_patch_ratio,
                "valid_patch_count": result.texture_result.valid_patch_count,
                "total_patch_count": result.texture_result.total_patch_count,
                "peak_patch_score": result.texture_result.peak_patch_score,
                "strong_patch_count": result.texture_result.strong_patch_count,
                "largest_component_patch_count": result.texture_result.largest_component_patch_count,
                "strong_patch_ratio": result.texture_result.strong_patch_ratio,
                "largest_component_patch_ratio": result.texture_result.largest_component_patch_ratio,
                "decision_mode": result.texture_result.decision_mode,
            }
            if result.texture_result is not None
            else None
        ),
        "color_result": (
            {
                "score": result.color_result.score,
                "threshold": result.color_result.threshold,
                "is_anomaly": result.color_result.is_anomaly,
                "diagnostics": result.color_result.diagnostics,
            }
            if result.color_result is not None
            else None
        ),
        "artifact_paths": result.artifact_paths,
    }


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


def _box_to_dict(box: BoundingBox | None) -> dict[str, float] | None:
    if box is None:
        return None
    return {
        "x1": box.x1,
        "y1": box.y1,
        "x2": box.x2,
        "y2": box.y2,
    }


def _resolve_target_box(result: CameraInspectionResult) -> dict[str, float] | None:
    detection = result.detection
    if detection is None or detection.target is None:
        return None
    return _box_to_dict(detection.target.bounding_box)


def _build_inspection_archive_path(base_path: Path, result: InspectionResult) -> Path:
    history_root = base_path.parent / f"{base_path.stem}_history"
    seat_model_dir = _sanitize_path_component(result.seat_model_id or "default")
    part_dir = _sanitize_path_component(result.part_id or "unknown_part")
    report_id = result.frame_id or result.timestamp or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{_sanitize_path_component(report_id)}.json"
    return history_root / seat_model_dir / part_dir / filename


def _sanitize_path_component(value: str) -> str:
    normalized = re.sub(r"[\\\\/:*?\"<>|]+", "_", value).strip()
    return normalized or "unknown"
