"""座椅缺陷检测流程共用数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BoundingBox:
    """矩形框。

    字段：
    - x1 / y1: 左上角坐标
    - x2 / y2: 右下角坐标

    属性：
    - width / height: 由坐标推导出的宽高
    """

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
    """标准化后的单帧数据。

    字段：
    - camera_id: 当前帧所属机位
    - frame_id: 当前帧唯一编号
    - part_id: 工件编号
    - source / source_kind: 输入源及其类型
    - timestamp: 采图时间
    - image: BGR 图像数据
    - image_path: 当输入源是图片文件时保留其路径
    """

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
    """图像质量指标。

    字段分别对应清晰度、平均亮度、过曝占比、欠曝占比，以及极端黑白帧标记。
    """

    laplacian_variance: float
    brightness_mean: float
    overexposed_ratio: float
    underexposed_ratio: float
    is_black_frame: bool
    is_white_frame: bool


@dataclass(slots=True)
class ImageQualityDecision:
    """图像质量判定结果。

    字段：
    - accepted: 是否通过质量门控
    - reason: 拒绝原因；通过时为 None
    - metrics: 原始质量指标
    """

    accepted: bool
    reason: str | None
    metrics: ImageQualityMetrics


@dataclass(slots=True)
class DetectionObject:
    """YOLO 输出的单个目标。

    字段：
    - label: 类别名
    - confidence: 检测置信度
    - bounding_box: 检测框
    - segmentation_mask: 分割掩膜，可为空
    """

    label: str
    confidence: float
    bounding_box: BoundingBox
    segmentation_mask: Any | None = None


@dataclass(slots=True)
class DetectionResult:
    """ROI 精修阶段使用的检测结果。

    字段：
    - target: 主目标，一般是 seat_main
    - all_objects: YOLO 输出的全部目标，便于调试
    """

    target: DetectionObject | None
    all_objects: list[DetectionObject] = field(default_factory=list)
    used_fallback: bool = False


@dataclass(slots=True)
class RoiRefineResult:
    """ROI 图像和掩膜结果。

    字段：
    - crop_box: 原图中的裁剪框
    - roi_image: 原始裁剪结果
    - aligned_roi_image: 对齐/缩放后的标准 ROI
    - texture_ready_image: 为 PatchCore 准备的纹理增强 ROI
    - target_mask: 目标前景掩膜
    - valid_mask: 最终可用区域掩膜
    - foreground_weight: 前景羽化权重图
    - alignment_applied: 是否实际执行了 ECC 对齐
    """

    crop_box: BoundingBox
    roi_image: Any
    aligned_roi_image: Any
    texture_ready_image: Any | None
    target_mask: Any
    valid_mask: Any
    ignore_mask: Any
    foreground_weight: Any | None
    alignment_applied: bool = False


@dataclass(slots=True)
class TextureAnomalyResult:
    """纹理异常分支输出。

    字段：
    - score / threshold / is_anomaly: PatchCore 判定结果
    - heatmap: ROI 级异常热力图
    - valid_patch_ratio / valid_patch_count / total_patch_count:
      patch 有效性统计
    - decision_threshold: 应用于最终工业判定的保守阈值
    - peak_patch_score: 当前图像最强 patch 分数
    - strong_patch_count: 达到强异常阈值的 patch 数量
    - largest_component_patch_count: 最大连通强异常区域包含的 patch 数量
    - strong_patch_ratio: 强异常 patch 占全部有效 patch 的比例
    - largest_component_patch_ratio: 最大连通强异常区域占全部有效 patch 的比例
    - decision_mode: 最终命中路径，便于区分常规命中还是强缺陷快速命中
    """

    score: float
    threshold: float
    is_anomaly: bool
    heatmap: Any
    valid_patch_ratio: float
    valid_patch_count: int
    total_patch_count: int
    decision_threshold: float = 0.0
    peak_patch_score: float = 0.0
    strong_patch_count: int = 0
    largest_component_patch_count: int = 0
    strong_patch_ratio: float = 0.0
    largest_component_patch_ratio: float = 0.0
    decision_mode: str = "none"


@dataclass(slots=True)
class ColorAnomalyResult:
    """颜色一致性分支输出。

    字段：
    - score / threshold / is_anomaly: 颜色分支判定结果
    - diagnostics: 颜色分支调试指标
    """

    score: float
    threshold: float
    is_anomaly: bool
    diagnostics: dict[str, float]


@dataclass(slots=True)
class CameraInspectionResult:
    """单机位检测结果。

    字段：
    - camera_id / frame_id / source / source_kind: 当前机位与输入源信息
    - status / reason: 单机位最终状态与原因
    - seat_model_id: 当前路由到的型号
    - quality / detection: 前处理阶段中间结果
    - texture_result / color_result: 纹理与颜色分支结果
    - crop_box: 最终使用的 ROI 框
    - artifact_paths: 调试图路径集合
    """

    camera_id: str
    frame_id: str
    source: str
    source_kind: str
    status: str
    reason: str
    seat_model_id: str | None = None
    quality: ImageQualityDecision | None = None
    detection: DetectionResult | None = None
    texture_result: TextureAnomalyResult | None = None
    color_result: ColorAnomalyResult | None = None
    crop_box: BoundingBox | None = None
    artifact_paths: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class InspectionResult:
    """多机位融合后的最终结果。

    字段：
    - part_id / frame_id / timestamp: 本次任务标识
    - status / decision_reason: 融合后的最终判定
    - seat_model_id: 本次使用的型号路由
    - camera_results: 所有机位的检测结果
    """

    part_id: str
    frame_id: str
    timestamp: str
    status: str
    decision_reason: str
    seat_model_id: str | None = None
    camera_results: list[CameraInspectionResult] = field(default_factory=list)


@dataclass(slots=True)
class CaptureRecord:
    """一次采图命令中某个机位的落盘结果。

    字段包含当前机位的采图状态、失败原因以及输出路径。
    """

    camera_id: str
    frame_id: str
    part_id: str
    source: str
    source_kind: str
    timestamp: str
    status: str
    seat_model_id: str | None = None
    reason: str | None = None
    output_path: str | None = None
    train_good_path: str | None = None


@dataclass(slots=True)
class CaptureSummary:
    """一次采图任务的汇总结果。

    字段：
    - part_id / run_id: 本次采图任务标识
    - output_dir / manifest_path: 输出目录和 manifest
    - seat_model_id: 当前使用的型号路由
    - records: 所有机位采图记录
    """

    part_id: str
    run_id: str
    output_dir: str
    manifest_path: str
    seat_model_id: str | None = None
    records: list[CaptureRecord] = field(default_factory=list)
