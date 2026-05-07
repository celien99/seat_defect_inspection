"""可嵌入外部项目的缺陷检测调用入口。"""

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from .config import InspectionConfig
from .runtime_config import load_config
from .schemas import InspectionResult

ConfigSource = str | PathLike[str] | InspectionConfig


@dataclass(slots=True)
class InspectionApiResponse:
    """外部系统调用一次在线检测后的完整响应。"""

    result: InspectionResult
    report_path: str
    archive_report_path: str
    artifact_paths: dict[str, dict[str, str]]

    @property
    def status(self) -> str:
        """融合后的最终状态，等价于 result.status。"""
        return self.result.status

    @property
    def decision_reason(self) -> str:
        """融合后的最终原因，等价于 result.decision_reason。"""
        return self.result.decision_reason

    @property
    def part_id(self) -> str:
        """本次检测的工件编号，等价于 result.part_id。"""
        return self.result.part_id

    @property
    def seat_model_id(self) -> str | None:
        """本次检测使用的座椅型号路由，等价于 result.seat_model_id。"""
        return self.result.seat_model_id

    def to_dict(self) -> dict[str, Any]:
        """返回适合外部 HTTP/RPC 层直接序列化的摘要。"""
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
    """可复用的座椅缺陷检测运行器。

    外部项目如果需要连续检测，应复用同一个实例，这样可以复用已构造的
    YOLO、PatchCore 和相机管线缓存，避免每次调用都重新初始化。
    """

    def __init__(self, config: ConfigSource) -> None:
        self.config = _resolve_config(config)
        self._service = _create_service(self.config)

    def inspect(
        self,
        part_id: str | None = None,
        *,
        seat_model_id: str | None = None,
    ) -> InspectionApiResponse:
        """执行一次完整在线检测。"""
        result = _run_online_inspection(
            self._service,
            part_id=part_id,
            seat_model_id=seat_model_id,
        )
        return _build_inspection_response(self.config, result)

    def inspect_folder(
        self,
        input_dir: str,
        *,
        seat_model_id: str | None = None,
        output_dir: str | None = None,
        part_id: str | None = None,
    ) -> dict[str, Any]:
        """从本地图片文件夹批量执行离线检测。"""
        return _run_offline_folder_inspection(
            self._service,
            input_dir=input_dir,
            seat_model_id=seat_model_id,
            output_dir=output_dir,
            part_id=part_id,
        )


def inspect_once(
    config: ConfigSource,
    part_id: str | None = None,
    *,
    seat_model_id: str | None = None,
) -> InspectionApiResponse:
    """使用配置路径或配置对象执行一次完整在线检测。"""
    return SeatDefectInspector(config).inspect(
        part_id=part_id,
        seat_model_id=seat_model_id,
    )


def inspect_folder_once(
    config: ConfigSource,
    input_dir: str,
    *,
    seat_model_id: str | None = None,
    output_dir: str | None = None,
    part_id: str | None = None,
) -> dict[str, Any]:
    """使用配置路径或配置对象执行一次离线图片文件夹批量检测。"""
    return SeatDefectInspector(config).inspect_folder(
        input_dir=input_dir,
        seat_model_id=seat_model_id,
        output_dir=output_dir,
        part_id=part_id,
    )


def _resolve_config(config: ConfigSource) -> InspectionConfig:
    if isinstance(config, InspectionConfig):
        return config
    return load_config(str(config))


def _create_service(config: InspectionConfig):
    from .service.core import InspectionService

    return InspectionService(config)


def _run_online_inspection(
    service,
    *,
    part_id: str | None,
    seat_model_id: str | None,
) -> InspectionResult:
    from .service.inspection import run_inspection

    return run_inspection(
        service,
        part_id=part_id,
        seat_model_id=seat_model_id,
    )


def _run_offline_folder_inspection(
    service,
    *,
    input_dir: str,
    seat_model_id: str | None,
    output_dir: str | None,
    part_id: str | None,
) -> dict[str, Any]:
    from .service.offline_inspection import inspect_image_folder

    return inspect_image_folder(
        service,
        input_dir=input_dir,
        seat_model_id=seat_model_id,
        output_dir=output_dir,
        part_id=part_id,
    )


def _build_inspection_response(
    config: InspectionConfig,
    result: InspectionResult,
) -> InspectionApiResponse:
    report_path = Path(config.output_json_path)
    archive_report_path = _resolve_archive_report_path(report_path, result)
    return InspectionApiResponse(
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


def _resolve_archive_report_path(report_path: Path, result: InspectionResult) -> Path:
    from .reporting import resolve_inspection_archive_path

    return resolve_inspection_archive_path(report_path, result)


__all__ = [
    "ConfigSource",
    "InspectionApiResponse",
    "SeatDefectInspector",
    "inspect_folder_once",
    "inspect_once",
]
