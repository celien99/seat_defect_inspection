"""图像质量检查。"""

from __future__ import annotations

import cv2

from .config import QualityGuardConfig
from .schemas import ImageQualityDecision, ImageQualityMetrics


class ImageQualityGuard:
    """过滤过糊、过暗或过曝的图像。"""

    def __init__(self, config: QualityGuardConfig) -> None:
        self.config = config

    def evaluate(self, image) -> ImageQualityDecision:
        """计算质量指标并返回通过或拒绝结论。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var()) # 拉普拉斯算子方差，用于检测图像清晰度
        brightness_mean = float(gray.mean()) # 图像平均亮度
        overexposed_ratio = float((gray >= 245).mean()) # 过曝像素比例
        underexposed_ratio = float((gray <= 10).mean()) # 欠曝像素比例

        metrics = ImageQualityMetrics(
            laplacian_variance=laplacian_variance,
            brightness_mean=brightness_mean,
            overexposed_ratio=overexposed_ratio,
            underexposed_ratio=underexposed_ratio,
            is_black_frame=brightness_mean <= 3.0,
            is_white_frame=brightness_mean >= 252.0,
        )

        if metrics.is_black_frame:
            return ImageQualityDecision(False, "black_frame", metrics)
        if metrics.is_white_frame:
            return ImageQualityDecision(False, "white_frame", metrics)
        if laplacian_variance < self.config.min_laplacian_variance:
            return ImageQualityDecision(False, "blur", metrics)
        if brightness_mean < self.config.min_brightness_mean:
            return ImageQualityDecision(False, "underexposed", metrics)
        if brightness_mean > self.config.max_brightness_mean:
            return ImageQualityDecision(False, "overexposed", metrics)
        if overexposed_ratio > self.config.max_overexposed_ratio:
            return ImageQualityDecision(False, "overexposed_ratio", metrics)
        if underexposed_ratio > self.config.max_underexposed_ratio:
            return ImageQualityDecision(False, "underexposed_ratio", metrics)
        return ImageQualityDecision(True, None, metrics)
