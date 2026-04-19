"""多机位结果融合。"""

from __future__ import annotations

from .config import FusionConfig
from .schemas import CameraInspectionResult, InspectionResult


def fuse_camera_results(
    *,
    part_id: str,
    frame_id: str,
    timestamp: str,
    camera_results: list[CameraInspectionResult],
    fusion_config: FusionConfig,
) -> InspectionResult:
    """将多个机位结果融合成最终 OK/NG/REJECT 结论。"""
    if not camera_results:
        return InspectionResult(
            part_id=part_id,
            frame_id=frame_id,
            timestamp=timestamp,
            status="REJECT",
            decision_reason="no_camera_results",
            camera_results=[],
        )

    rejects = [result for result in camera_results if result.status == "REJECT"]
    ng_results = [result for result in camera_results if result.status == "NG"]
    ok_results = [result for result in camera_results if result.status == "OK"]

    if rejects and fusion_config.reject_on_any_reject:
        return InspectionResult(
            part_id=part_id,
            frame_id=frame_id,
            timestamp=timestamp,
            status="REJECT",
            decision_reason=f"reject_from_{rejects[0].camera_id}",
            camera_results=camera_results,
        )

    if _apply_ng_strategy(len(ng_results), len(camera_results), fusion_config.ng_strategy):
        return InspectionResult(
            part_id=part_id,
            frame_id=frame_id,
            timestamp=timestamp,
            status="NG",
            decision_reason=f"ng_from_{','.join(result.camera_id for result in ng_results)}",
            camera_results=camera_results,
        )

    if len(ok_results) == len(camera_results):
        return InspectionResult(
            part_id=part_id,
            frame_id=frame_id,
            timestamp=timestamp,
            status="OK",
            decision_reason="all_cameras_ok",
            camera_results=camera_results,
        )

    return InspectionResult(
        part_id=part_id,
        frame_id=frame_id,
        timestamp=timestamp,
        status="REJECT",
        decision_reason="incomplete_camera_results",
        camera_results=camera_results,
    )


def _apply_ng_strategy(ng_count: int, total_count: int, strategy: str) -> bool:
    normalized = strategy.strip().lower()
    if normalized == "all":
        return total_count > 0 and ng_count == total_count
    if normalized == "majority":
        return ng_count >= (total_count // 2 + 1)
    if normalized != "any":
        raise ValueError(f"Unsupported NG fusion strategy: {strategy}")
    return ng_count > 0
