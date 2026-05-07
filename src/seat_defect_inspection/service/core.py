"""工程层服务骨架。

检测 runtime 继承自 ``seat_defect_core``，本层只补充采图能力。
"""

from __future__ import annotations

from seat_defect_core.service.core import (
    CameraPipeline,
    InspectionService as CoreInspectionService,
    PreparedCameraSample,
    ResolvedInspectionContext,
)

from ..acquisition import AcquisitionService
from ..config import InspectionConfig


class InspectionService(CoreInspectionService):
    """带采图能力的工程层服务。"""

    config: InspectionConfig

    def __init__(self, config: InspectionConfig) -> None:
        super().__init__(config)
        self.acquisition = AcquisitionService(config.capture_retries)


__all__ = [
    "CameraPipeline",
    "InspectionService",
    "PreparedCameraSample",
    "ResolvedInspectionContext",
]
