"""主流程服务入口。"""

from __future__ import annotations

from typing import Any

from ..config import InspectionConfig
from ..schemas import CaptureSummary, InspectionResult
from .core import InspectionService, PreparedCameraSample, _CameraPipeline

__all__ = [
    "InspectionService",
    "PreparedCameraSample",
    "_CameraPipeline",
    "capture_samples",
    "run_inspection",
    "train_patchcore_models",
]


def train_patchcore_models(
    config: InspectionConfig,
    seat_model_id: str | None = None,
) -> list[dict[str, Any]]:
    """训练全部机位的 PatchCore 模型。"""
    return InspectionService(config).train_patchcore_models(seat_model_id=seat_model_id)


def capture_samples(
    config: InspectionConfig,
    part_id: str | None = None,
    *,
    output_dir: str | None = None,
    seat_model_id: str | None = None,
    save_to_train_good_dir: bool = False,
    count: int = 1,
    interval_ms: int = 0,
) -> CaptureSummary:
    """抓取并落盘全部启用机位的图像。"""
    return InspectionService(config).capture(
        part_id=part_id,
        output_dir=output_dir,
        seat_model_id=seat_model_id,
        save_to_train_good_dir=save_to_train_good_dir,
        count=count,
        interval_ms=interval_ms,
    )


def run_inspection(
    config: InspectionConfig,
    part_id: str | None = None,
    *,
    seat_model_id: str | None = None,
) -> InspectionResult:
    """执行一次完整检测。"""
    return InspectionService(config).run_inspection(
        part_id=part_id,
        seat_model_id=seat_model_id,
    )
