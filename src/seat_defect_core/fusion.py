"""多机位结果融合。"""

from __future__ import annotations

from .config import FusionConfig
from .types import CameraInspectionResult, InspectionResult


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
    ng_triggered = _apply_ng_strategy(
        len(ng_results),
        len(camera_results),
        fusion_config.ng_strategy,
    )

    if ng_triggered and fusion_config.defect_overrides_reject:
        return InspectionResult(
            part_id=part_id,
            frame_id=frame_id,
            timestamp=timestamp,
            status="NG",
            decision_reason=_build_ng_decision_reason(ng_results, rejects),
            camera_results=camera_results,
        )

    if rejects and fusion_config.reject_on_any_reject:
        return InspectionResult(
            part_id=part_id,
            frame_id=frame_id,
            timestamp=timestamp,
            status="REJECT",
            decision_reason=f"reject_from_{rejects[0].camera_id}",
            camera_results=camera_results,
        )

    if ng_triggered:
        return InspectionResult(
            part_id=part_id,
            frame_id=frame_id,
            timestamp=timestamp,
            status="NG",
            decision_reason=_build_ng_decision_reason(ng_results, []),
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


def _build_ng_decision_reason(
    ng_results: list[CameraInspectionResult],
    rejects: list[CameraInspectionResult],
) -> str:
    ng_cameras = ",".join(result.camera_id for result in ng_results)
    defect_info = _extract_defect_type_info(ng_results)
    if rejects:
        reject_cameras = ",".join(result.camera_id for result in rejects)
        return f"ng_from_{ng_cameras}_types:{defect_info}_override_reject_from_{reject_cameras}"
    return f"ng_from_{ng_cameras}_types:{defect_info}"


def _extract_defect_type_info(
    ng_results: list[CameraInspectionResult],
) -> str:
    """从 NG 相机结果中提取缺陷类型摘要。"""
    defect_types: list[str] = []
    for result in ng_results:
        if result.texture_result and result.texture_result.classification_results:
            primary = result.texture_result.classification_results[0]
            if primary.defect_type.value != "none":
                defect_types.append(
                    f"{result.camera_id}:{primary.defect_type.value}"
                )
        if result.region_results:
            for region in result.region_results:
                if (
                    region.texture_result
                    and region.texture_result.classification_results
                ):
                    primary = region.texture_result.classification_results[0]
                    if primary.defect_type.value != "none":
                        defect_types.append(
                            f"{result.camera_id}/{region.region_id}:{primary.defect_type.value}"
                        )
    return ",".join(defect_types) if defect_types else "unknown_anomaly"
