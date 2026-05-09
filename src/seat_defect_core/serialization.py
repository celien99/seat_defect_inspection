"""Stable JSON-serializable result mapping helpers."""

from __future__ import annotations

from typing import Any

from .types import BoundingBox, CameraInspectionResult, InspectionError, InspectionResult


def inspection_result_to_dict(result: InspectionResult) -> dict[str, Any]:
    """Convert an inspection result to the public/report JSON payload."""
    return {
        "part_id": result.part_id,
        "frame_id": result.frame_id,
        "timestamp": result.timestamp,
        "status": result.status,
        "decision_reason": result.decision_reason,
        "seat_model_id": result.seat_model_id,
        "timings_ms": dict(result.timings_ms),
        "camera_results": [
            camera_result_to_dict(item)
            for item in result.camera_results
        ],
    }


def camera_result_to_dict(result: CameraInspectionResult) -> dict[str, Any]:
    """Convert one camera result to a JSON-safe payload."""
    return {
        "camera_id": result.camera_id,
        "frame_id": result.frame_id,
        "source": result.source,
        "source_kind": result.source_kind,
        "status": result.status,
        "reason": result.reason,
        "seat_model_id": result.seat_model_id,
        "timings_ms": dict(result.timings_ms),
        "error": error_to_dict(result.error),
        "quality": quality_to_dict(result.quality),
        "target_box": resolve_target_box(result),
        "crop_box": box_to_dict(result.crop_box),
        "texture_result": texture_result_to_dict(result.texture_result),
        "region_results": [
            {
                "region_id": item.region_id,
                "status": item.status,
                "reason": item.reason,
                "box": box_to_dict(item.box),
                "patchcore_model_path": item.patchcore_model_path,
                "texture_result": texture_result_to_dict(item.texture_result),
                "artifact_paths": dict(item.artifact_paths),
                "timings_ms": dict(item.timings_ms),
                "error": error_to_dict(item.error),
            }
            for item in result.region_results
        ],
        "color_result": color_result_to_dict(result.color_result),
        "artifact_paths": dict(result.artifact_paths),
    }


def quality_to_dict(quality) -> dict[str, Any] | None:
    if quality is None:
        return None
    return {
        "accepted": quality.accepted,
        "reason": quality.reason,
        "metrics": {
            "laplacian_variance": quality.metrics.laplacian_variance,
            "brightness_mean": quality.metrics.brightness_mean,
            "overexposed_ratio": quality.metrics.overexposed_ratio,
            "underexposed_ratio": quality.metrics.underexposed_ratio,
            "is_black_frame": quality.metrics.is_black_frame,
            "is_white_frame": quality.metrics.is_white_frame,
        },
    }


def texture_result_to_dict(texture_result) -> dict[str, Any] | None:
    if texture_result is None:
        return None
    return {
        "score": texture_result.score,
        "threshold": texture_result.threshold,
        "decision_threshold": texture_result.decision_threshold,
        "is_anomaly": texture_result.is_anomaly,
        "valid_patch_ratio": texture_result.valid_patch_ratio,
        "valid_patch_count": texture_result.valid_patch_count,
        "total_patch_count": texture_result.total_patch_count,
        "peak_patch_score": texture_result.peak_patch_score,
        "strong_patch_count": texture_result.strong_patch_count,
        "largest_component_patch_count": texture_result.largest_component_patch_count,
        "strong_patch_ratio": texture_result.strong_patch_ratio,
        "largest_component_patch_ratio": texture_result.largest_component_patch_ratio,
        "decision_patch_count": texture_result.decision_patch_count,
        "largest_decision_component_patch_count": (
            texture_result.largest_decision_component_patch_count
        ),
        "decision_patch_ratio": texture_result.decision_patch_ratio,
        "largest_decision_component_patch_ratio": (
            texture_result.largest_decision_component_patch_ratio
        ),
        "decision_mode": texture_result.decision_mode,
    }


def color_result_to_dict(color_result) -> dict[str, Any] | None:
    if color_result is None:
        return None
    return {
        "score": color_result.score,
        "threshold": color_result.threshold,
        "is_anomaly": color_result.is_anomaly,
        "diagnostics": dict(color_result.diagnostics),
    }


def error_to_dict(error: InspectionError | None) -> dict[str, str] | None:
    if error is None:
        return None
    return {
        "code": error.code,
        "message": error.message,
        "stage": error.stage,
    }


def box_to_dict(box: BoundingBox | None) -> dict[str, float] | None:
    if box is None:
        return None
    return {
        "x1": box.x1,
        "y1": box.y1,
        "x2": box.x2,
        "y2": box.y2,
    }


def resolve_target_box(result: CameraInspectionResult) -> dict[str, float] | None:
    detection = result.detection
    if detection is None or detection.target is None:
        return None
    return box_to_dict(detection.target.bounding_box)


__all__ = [
    "box_to_dict",
    "camera_result_to_dict",
    "color_result_to_dict",
    "error_to_dict",
    "inspection_result_to_dict",
    "quality_to_dict",
    "resolve_target_box",
    "texture_result_to_dict",
]
