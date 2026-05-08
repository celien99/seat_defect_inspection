"""主检测流程输出结果类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .geometry import BoundingBox
from .pipeline import DetectionResult, ImageQualityDecision


@dataclass(slots=True)
class TextureAnomalyResult:
    """纹理异常分支输出。"""

    score: float
    """图像级异常分数。"""

    threshold: float
    """训练阶段得到的基础异常阈值。"""

    is_anomaly: bool
    """是否判定为异常。"""

    heatmap: Any
    """ROI 坐标系下的异常热力图。"""

    valid_patch_ratio: float
    """有效 patch 占全部 patch 的比例。"""

    valid_patch_count: int
    """有效 patch 数量。"""

    total_patch_count: int
    """全部 patch 数量。"""

    decision_threshold: float = 0.0
    """最终工业判定使用的阈值。"""

    peak_patch_score: float = 0.0
    """当前图像最高 patch 异常分数。"""

    strong_patch_count: int = 0
    """达到强异常阈值的 patch 数量。"""

    largest_component_patch_count: int = 0
    """最大强异常连通域包含的 patch 数量。"""

    strong_patch_ratio: float = 0.0
    """强异常 patch 占有效 patch 的比例。"""

    largest_component_patch_ratio: float = 0.0
    """最大强异常连通域占有效 patch 的比例。"""

    decision_patch_count: int = 0
    """达到最终判定阈值的 patch 数量。"""

    largest_decision_component_patch_count: int = 0
    """最大最终判定连通域包含的 patch 数量。"""

    decision_patch_ratio: float = 0.0
    """达到最终判定阈值的 patch 占比。"""

    largest_decision_component_patch_ratio: float = 0.0
    """最大最终判定连通域占比。"""

    decision_mode: str = "none"
    """最终命中的判定模式。"""


@dataclass(slots=True)
class ColorAnomalyResult:
    """颜色一致性分支输出。"""

    score: float
    """颜色异常分数。"""

    threshold: float
    """颜色异常阈值。"""

    is_anomaly: bool
    """是否判定为颜色异常。"""

    diagnostics: dict[str, float]
    """颜色分支诊断指标。"""


@dataclass(slots=True)
class RegionPatchCoreResult:
    """单个局部区域的 PatchCore 输出。"""

    region_id: str
    """区域 ID。"""

    status: str
    """区域状态：OK / NG / REJECT。"""

    reason: str
    """区域状态原因。"""

    box: BoundingBox
    """标准 ROI 坐标系下的区域矩形框。"""

    texture_result: TextureAnomalyResult | None = None
    """该区域的纹理异常结果。"""

    patchcore_model_path: str | None = None
    """该区域使用的 PatchCore 模型路径。"""

    artifact_paths: dict[str, str] = field(default_factory=dict)
    """该区域关联的调试产物路径。"""


@dataclass(slots=True)
class CameraInspectionResult:
    """单机位最终检测结果。"""

    camera_id: str
    """机位 ID。"""

    frame_id: str
    """帧编号。"""

    source: str
    """输入来源标识。"""

    source_kind: str
    """输入来源类型。"""

    status: str
    """单机位状态：OK / NG / REJECT。"""

    reason: str
    """单机位状态原因。"""

    seat_model_id: str | None = None
    """本次检测使用的座椅型号 ID。"""

    quality: ImageQualityDecision | None = None
    """图像质量判定结果。"""

    detection: DetectionResult | None = None
    """YOLO 检测结果。"""

    texture_result: TextureAnomalyResult | None = None
    """完整 ROI 模式下的纹理异常结果。"""

    region_results: list[RegionPatchCoreResult] = field(default_factory=list)
    """regions 模式下的区域检测结果。"""

    color_result: ColorAnomalyResult | None = None
    """颜色一致性分支结果。"""

    crop_box: BoundingBox | None = None
    """原图坐标系下最终使用的 ROI 裁剪框。"""

    artifact_paths: dict[str, str] = field(default_factory=dict)
    """该机位关联的调试产物路径。"""


@dataclass(slots=True)
class InspectionResult:
    """多机位融合后的整件检测结果。"""

    part_id: str
    """工件编号。"""

    frame_id: str
    """本次检测批次帧编号。"""

    timestamp: str
    """本次检测时间戳。"""

    status: str
    """整件状态：OK / NG / REJECT。"""

    decision_reason: str
    """整件融合判定原因。"""

    seat_model_id: str | None = None
    """本次检测使用的座椅型号 ID。"""

    camera_results: list[CameraInspectionResult] = field(default_factory=list)
    """所有机位检测结果。"""


@dataclass(slots=True)
class InspectionResponse:
    """core 对外返回的检测响应。"""

    result: InspectionResult
    """完整检测结果对象。"""

    report_path: str
    """最新检测报告 JSON 路径。"""

    archive_report_path: str
    """历史归档报告 JSON 路径。"""

    artifact_paths: dict[str, dict[str, str]]
    """按机位聚合的调试产物路径。"""

    @property
    def status(self) -> str:
        """整件状态快捷访问。"""
        return self.result.status

    @property
    def decision_reason(self) -> str:
        """整件判定原因快捷访问。"""
        return self.result.decision_reason

    @property
    def part_id(self) -> str:
        """工件编号快捷访问。"""
        return self.result.part_id

    @property
    def seat_model_id(self) -> str | None:
        """座椅型号 ID 快捷访问。"""
        return self.result.seat_model_id

    def to_dict(self) -> dict[str, Any]:
        """转换为适合外部系统序列化的字典。"""
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
                    "region_results": [
                        {
                            "region_id": region_result.region_id,
                            "status": region_result.status,
                            "reason": region_result.reason,
                            "patchcore_model_path": region_result.patchcore_model_path,
                            "artifact_paths": dict(region_result.artifact_paths),
                        }
                        for region_result in camera_result.region_results
                    ],
                }
                for camera_result in self.result.camera_results
            ],
        }


__all__ = [
    "CameraInspectionResult",
    "ColorAnomalyResult",
    "InspectionResponse",
    "InspectionResult",
    "RegionPatchCoreResult",
    "TextureAnomalyResult",
]
