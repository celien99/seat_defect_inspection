"""主检测 pipeline 中间结果类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .geometry import BoundingBox


@dataclass
class ImageQualityMetrics:
    """图像质量指标。"""

    laplacian_variance: float
    """拉普拉斯方差，用于衡量清晰度。"""

    brightness_mean: float
    """平均亮度。"""

    overexposed_ratio: float
    """过曝像素占比。"""

    underexposed_ratio: float
    """欠曝像素占比。"""

    is_black_frame: bool
    """是否判定为黑帧。"""

    is_white_frame: bool
    """是否判定为白帧。"""


@dataclass
class ImageQualityDecision:
    """图像质量门控判定。"""

    accepted: bool
    """是否通过质量门控。"""

    reason: Optional[str]
    """未通过时的原因；通过时为 None。"""

    metrics: ImageQualityMetrics
    """原始质量指标。"""


@dataclass
class DetectionObject:
    """YOLO 输出的单个目标。"""

    label: str
    """目标类别名。"""

    confidence: float
    """检测置信度。"""

    bounding_box: BoundingBox
    """目标检测框。"""

    segmentation_mask: Optional[Any] = None
    """目标分割掩膜；检测模型无 mask 时为空。"""


@dataclass
class DetectionResult:
    """YOLO 检测结果。"""

    target: Optional[DetectionObject]
    """主目标对象，一般为座椅；未找到时为 None。"""

    all_objects: List[DetectionObject] = field(default_factory=list)
    """YOLO 输出的全部目标，主要用于调试。"""


@dataclass
class RoiRefineResult:
    """ROI 精修后的图像与掩膜集合。"""

    crop_box: BoundingBox
    """原图坐标系中的 ROI 裁剪框。"""

    roi_image: Any
    """原始 ROI 裁剪图。"""

    aligned_roi_image: Any
    """对齐和缩放后的标准 ROI 图。"""

    texture_ready_image: Optional[Any]
    """为 PatchCore 准备的纹理输入图；未启用时为空。"""

    target_mask: Any
    """标准 ROI 内的目标前景 mask。"""

    valid_mask: Any
    """最终可参与检测的有效区域 mask。"""

    ignore_mask: Any
    """需要忽略的区域 mask。"""

    foreground_weight: Optional[Any]
    """前景羽化权重图；未生成时为空。"""

    alignment_applied: bool = False
    """是否实际执行过对齐处理。"""


__all__ = [
    "DetectionObject",
    "DetectionResult",
    "ImageQualityDecision",
    "ImageQualityMetrics",
    "RoiRefineResult",
]
