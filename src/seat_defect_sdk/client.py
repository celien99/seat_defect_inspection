"""不包含采图动作的座椅缺陷检测 SDK。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

from seat_defect_core.config import InspectionConfig
from seat_defect_core.runtime_config import load_config
from seat_defect_core.schemas import (
    CameraInspectionResult,
    FramePacket,
    InspectionResult,
)

if TYPE_CHECKING:
    from seat_defect_core.config import CameraConfig
    from seat_defect_core.service.core import InspectionService

ConfigSource = str | PathLike[str] | InspectionConfig


@dataclass(slots=True)
class CameraFrame:
    """外部系统传给 SDK 的单机位图片。"""

    camera_id: str
    image: Any
    source: str | None = None
    frame_id: str | None = None
    timestamp: str | None = None
    source_kind: str = "external_image"


@dataclass(slots=True)
class InspectionSdkResponse:
    """SDK 执行一次检测后的完整响应。"""

    result: InspectionResult
    report_path: str
    archive_report_path: str
    artifact_paths: dict[str, dict[str, str]]

    @property
    def status(self) -> str:
        return self.result.status

    @property
    def decision_reason(self) -> str:
        return self.result.decision_reason

    @property
    def part_id(self) -> str:
        return self.result.part_id

    @property
    def seat_model_id(self) -> str | None:
        return self.result.seat_model_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_id": self.result.part_id,
            "frame_id": self.result.frame_id,
            "timestamp": self.result.timestamp,
            "status": self.result.status,
            "decision_reason": self.result.decision_reason,
            "seat_model_id": self.result.seat_model_id,
            "report_path": self.report_path,
            "archive_report_path": self.archive_report_path,
            "artifact_paths": self.artifact_paths,
            "camera_results": [
                {
                    "camera_id": camera_result.camera_id,
                    "frame_id": camera_result.frame_id,
                    "source": camera_result.source,
                    "source_kind": camera_result.source_kind,
                    "status": camera_result.status,
                    "reason": camera_result.reason,
                    "seat_model_id": camera_result.seat_model_id,
                    "artifact_paths": dict(camera_result.artifact_paths),
                }
                for camera_result in self.result.camera_results
            ],
        }


class SeatDefectInspector:
    """可复用 SDK 检测器。

    SDK 不负责采集图片。调用方必须把每个机位的图片作为 frames 传入。
    """

    def __init__(self, config: ConfigSource) -> None:
        self.config = _resolve_config(config)
        self._service = _create_service(self.config)

    def inspect(
        self,
        frames: list[CameraFrame | dict[str, Any]],
        *,
        part_id: str | None = None,
        seat_model_id: str | None = None,
    ) -> InspectionSdkResponse:
        """基于外部传入图片执行一次完整检测。"""
        normalized_frames = [_normalize_frame(frame) for frame in frames]
        _build_frame_map(normalized_frames)
        result = _run_inspection_from_frames(
            self._service,
            normalized_frames,
            part_id=part_id,
            seat_model_id=seat_model_id,
        )
        return _build_response(self.config, result)


def inspect_once(
    config: ConfigSource,
    frames: list[CameraFrame | dict[str, Any]],
    *,
    part_id: str | None = None,
    seat_model_id: str | None = None,
) -> InspectionSdkResponse:
    """使用配置路径或配置对象，对外部传入图片执行一次检测。"""
    return SeatDefectInspector(config).inspect(
        frames,
        part_id=part_id,
        seat_model_id=seat_model_id,
    )


def _run_inspection_from_frames(
    service: "InspectionService",
    frames: list[CameraFrame],
    *,
    part_id: str | None,
    seat_model_id: str | None,
) -> InspectionResult:
    context = service.resolve_context(seat_model_id)
    resolved_part_id = part_id or service.config.part_id
    if not context.cameras:
        return _export_result(
            service,
            InspectionResult(
                part_id=resolved_part_id,
                frame_id="",
                timestamp="",
                status="REJECT",
                decision_reason="no_enabled_cameras",
                seat_model_id=context.seat_model_id,
                camera_results=[],
            ),
        )

    frame_map = _build_frame_map(frames)
    _validate_frame_camera_ids(frame_map, [camera.camera_id for camera in context.cameras])
    run_frame_id = _resolve_run_frame_id(frames)
    run_timestamp = _resolve_run_timestamp(frames)
    camera_results: list[CameraInspectionResult] = []
    total_camera_count = len(context.cameras)

    for camera in context.cameras:
        external_frame = frame_map.get(camera.camera_id)
        if external_frame is None:
            camera_results.append(
                _build_missing_frame_result(
                    camera,
                    frame_id=run_frame_id,
                    seat_model_id=context.seat_model_id,
                )
            )
        else:
            frame_packet = _build_frame_packet(
                external_frame,
                camera,
                part_id=resolved_part_id,
                fallback_frame_id=run_frame_id,
                fallback_timestamp=run_timestamp,
            )
            try:
                camera_results.append(
                    _inspect_external_camera(
                        service,
                        frame_packet,
                        camera,
                        context.pipelines[camera.camera_id],
                        context.seat_model_id,
                    )
                )
            except Exception as exc:
                camera_results.append(
                    _build_reject_result(
                        camera_id=frame_packet.camera_id,
                        frame_id=frame_packet.frame_id,
                        source=frame_packet.source,
                        source_kind=frame_packet.source_kind,
                        reason=f"pipeline_failed:{exc}",
                        seat_model_id=context.seat_model_id,
                    )
                )

        if _should_early_stop_on_ng(
            camera_results=camera_results,
            total_camera_count=total_camera_count,
            fusion_config=service.config.fusion,
        ):
            return _export_result(
                service,
                InspectionResult(
                    part_id=resolved_part_id,
                    frame_id=run_frame_id,
                    timestamp=run_timestamp,
                    status="NG",
                    decision_reason=_build_early_stop_reason(camera_results),
                    seat_model_id=context.seat_model_id,
                    camera_results=camera_results,
                ),
            )

    fused = _fuse_camera_results(
        part_id=resolved_part_id,
        frame_id=run_frame_id,
        timestamp=run_timestamp,
        camera_results=camera_results,
        fusion_config=service.config.fusion,
    )
    fused.seat_model_id = context.seat_model_id
    return _export_result(service, fused)


def _normalize_frame(frame: CameraFrame | dict[str, Any]) -> CameraFrame:
    if isinstance(frame, CameraFrame):
        return frame
    try:
        camera_id = frame["camera_id"]
        image = frame["image"]
    except KeyError as exc:
        raise ValueError("每个 frame 必须包含 camera_id 和 image") from exc
    return CameraFrame(
        camera_id=str(camera_id),
        image=image,
        source=frame.get("source"),
        frame_id=frame.get("frame_id"),
        timestamp=frame.get("timestamp"),
        source_kind=str(frame.get("source_kind", "external_image")),
    )


def _build_frame_map(frames: list[CameraFrame]) -> dict[str, CameraFrame]:
    frame_map: dict[str, CameraFrame] = {}
    duplicates: set[str] = set()
    for frame in frames:
        if frame.camera_id in frame_map:
            duplicates.add(frame.camera_id)
        frame_map[frame.camera_id] = frame
    if duplicates:
        duplicated_ids = ", ".join(f"`{camera_id}`" for camera_id in sorted(duplicates))
        raise ValueError(f"frames 中存在重复 camera_id: {duplicated_ids}")
    return frame_map


def _validate_frame_camera_ids(
    frame_map: dict[str, CameraFrame],
    active_camera_ids: list[str],
) -> None:
    active_id_set = set(active_camera_ids)
    unknown_ids = sorted(set(frame_map) - active_id_set)
    if not unknown_ids:
        return
    unknown = ", ".join(f"`{camera_id}`" for camera_id in unknown_ids)
    available = ", ".join(f"`{camera_id}`" for camera_id in active_camera_ids) or "none"
    raise ValueError(f"frames 包含未配置或未启用的 camera_id: {unknown}；可用 camera_id: {available}")


def _build_frame_packet(
    frame: CameraFrame,
    camera: "CameraConfig",
    *,
    part_id: str,
    fallback_frame_id: str,
    fallback_timestamp: str,
) -> FramePacket:
    frame_id = frame.frame_id or fallback_frame_id
    timestamp = frame.timestamp or fallback_timestamp
    return FramePacket(
        camera_id=camera.camera_id,
        frame_id=frame_id,
        part_id=part_id,
        source=frame.source or f"external://{camera.camera_id}",
        source_kind=frame.source_kind,
        timestamp=timestamp,
        image=frame.image,
        image_path=frame.source,
    )


def _build_missing_frame_result(
    camera: "CameraConfig",
    *,
    frame_id: str,
    seat_model_id: str | None,
) -> CameraInspectionResult:
    return _build_reject_result(
        camera_id=camera.camera_id,
        frame_id=frame_id,
        source=f"external://{camera.camera_id}",
        source_kind="external_image",
        reason="missing_external_frame",
        seat_model_id=seat_model_id,
    )


def _build_early_stop_reason(camera_results: list[CameraInspectionResult]) -> str:
    ng_cameras = [result.camera_id for result in camera_results if result.status == "NG"]
    if not ng_cameras:
        return "early_stop_without_ng"
    return f"early_stop_ng_from_{','.join(ng_cameras)}"


def _resolve_run_frame_id(frames: list[CameraFrame]) -> str:
    for frame in frames:
        if frame.frame_id:
            return frame.frame_id
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")


def _resolve_run_timestamp(frames: list[CameraFrame]) -> str:
    for frame in frames:
        if frame.timestamp:
            return frame.timestamp
    return datetime.now().astimezone().isoformat()


def _export_result(service: "InspectionService", result: InspectionResult) -> InspectionResult:
    from seat_defect_core.reporting import export_inspection_report

    export_inspection_report(result, service.config.output_json_path)
    return result


def _build_response(
    config: InspectionConfig,
    result: InspectionResult,
) -> InspectionSdkResponse:
    from seat_defect_core.reporting import resolve_inspection_archive_path

    report_path = Path(config.output_json_path)
    archive_report_path = resolve_inspection_archive_path(report_path, result)
    return InspectionSdkResponse(
        result=result,
        report_path=str(report_path),
        archive_report_path=str(archive_report_path),
        artifact_paths=_collect_artifact_paths(result),
    )


def _collect_artifact_paths(result: InspectionResult) -> dict[str, dict[str, str]]:
    return {
        camera_result.camera_id: dict(camera_result.artifact_paths)
        for camera_result in result.camera_results
        if camera_result.artifact_paths
    }


def _resolve_config(config: ConfigSource) -> InspectionConfig:
    if isinstance(config, InspectionConfig):
        return config
    return load_config(str(config))


def _create_service(config: InspectionConfig):
    from seat_defect_core.service.core import InspectionService

    return InspectionService(config)


def _inspect_external_camera(*args, **kwargs) -> CameraInspectionResult:
    from seat_defect_core.service.inspection_camera import inspect_one_camera

    return inspect_one_camera(*args, **kwargs)


def _build_reject_result(
    *,
    camera_id: str,
    frame_id: str,
    source: str,
    source_kind: str,
    reason: str,
    seat_model_id: str | None,
) -> CameraInspectionResult:
    return CameraInspectionResult(
        camera_id=camera_id,
        frame_id=frame_id,
        source=source,
        source_kind=source_kind,
        status="REJECT",
        reason=reason,
        seat_model_id=seat_model_id,
    )


def _should_early_stop_on_ng(**kwargs) -> bool:
    from seat_defect_core.fusion import should_early_stop_on_ng

    return should_early_stop_on_ng(**kwargs)


def _fuse_camera_results(**kwargs) -> InspectionResult:
    from seat_defect_core.fusion import fuse_camera_results

    return fuse_camera_results(**kwargs)


__all__ = [
    "CameraFrame",
    "ConfigSource",
    "InspectionSdkResponse",
    "SeatDefectInspector",
    "inspect_once",
]
