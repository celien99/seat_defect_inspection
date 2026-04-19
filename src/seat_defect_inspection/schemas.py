"""座椅缺陷检测流程共用数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BoundingBox:
    """矩形框。"""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


@dataclass(slots=True)
class FramePacket:
    """标准化后的单帧数据。"""

    camera_id: str
    frame_id: str
    part_id: str
    source: str
    source_kind: str
    timestamp: str
    image: Any
    image_path: str | None = None


@dataclass(slots=True)
class ImageQualityMetrics:
    """图像质量指标。"""

    laplacian_variance: float
    brightness_mean: float
    overexposed_ratio: float
    underexposed_ratio: float
    is_black_frame: bool
    is_white_frame: bool


@dataclass(slots=True)
class ImageQualityDecision:
    """图像质量判定结果。"""

    accepted: bool
    reason: str | None
    metrics: ImageQualityMetrics


@dataclass(slots=True)
class DetectionObject:
    """YOLO 输出的单个目标。"""

    label: str
    confidence: float
    bounding_box: BoundingBox
    segmentation_mask: Any | None = None


@dataclass(slots=True)
class DetectionResult:
    """ROI 精修阶段使用的检测结果。"""

    target: DetectionObject | None
    ignores: list[DetectionObject] = field(default_factory=list)
    all_objects: list[DetectionObject] = field(default_factory=list)


@dataclass(slots=True)
class RoiRefineResult:
    """ROI 图像和掩膜结果。"""

    crop_box: BoundingBox
    roi_image: Any
    aligned_roi_image: Any
    target_mask: Any
    ignore_mask: Any
    valid_mask: Any
    alignment_applied: bool = False


@dataclass(slots=True)
class TextureAnomalyResult:
    """纹理异常分支输出。"""

    score: float
    threshold: float
    is_anomaly: bool
    heatmap: Any
    valid_patch_ratio: float
    valid_patch_count: int
    total_patch_count: int


@dataclass(slots=True)
class ColorAnomalyResult:
    """颜色一致性分支输出。"""

    score: float
    threshold: float
    is_anomaly: bool
    diagnostics: dict[str, float]


@dataclass(slots=True)
class CameraInspectionResult:
    """单机位检测结果。"""

    camera_id: str
    frame_id: str
    source: str
    source_kind: str
    status: str
    reason: str
    quality: ImageQualityDecision | None = None
    detection: DetectionResult | None = None
    texture_result: TextureAnomalyResult | None = None
    color_result: ColorAnomalyResult | None = None
    crop_box: BoundingBox | None = None
    artifact_paths: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class InspectionResult:
    """多机位融合后的最终结果。"""

    part_id: str
    frame_id: str
    timestamp: str
    status: str
    decision_reason: str
    camera_results: list[CameraInspectionResult] = field(default_factory=list)


@dataclass(slots=True)
class CaptureRecord:
    """一次采图命令中某个机位的落盘结果。"""

    camera_id: str
    frame_id: str
    part_id: str
    source: str
    source_kind: str
    timestamp: str
    status: str
    reason: str | None = None
    output_path: str | None = None
    train_good_path: str | None = None


@dataclass(slots=True)
class CaptureSummary:
    """一次采图任务的汇总结果。"""

    part_id: str
    run_id: str
    output_dir: str
    manifest_path: str
    records: list[CaptureRecord] = field(default_factory=list)
