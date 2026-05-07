"""基于 LAB 统计量的颜色一致性分支。"""

from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import ColorBranchConfig
from ..schemas import ColorAnomalyResult


@dataclass(slots=True)
class ColorReferenceProfile:
    """单机位正常颜色分布。"""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    threshold: float

    def to_json(self) -> str:
        return json.dumps(
            {
                "feature_mean": self.feature_mean.tolist(),
                "feature_std": self.feature_std.tolist(),
                "threshold": float(self.threshold),
            }
        )

    @classmethod
    def from_json(cls, payload: str) -> "ColorReferenceProfile":
        loaded = json.loads(payload)
        return cls(
            feature_mean=np.asarray(loaded["feature_mean"], dtype=np.float32),
            feature_std=np.asarray(loaded["feature_std"], dtype=np.float32),
            threshold=float(loaded["threshold"]),
        )


class ColorConsistencyService:
    """Run a loaded lightweight color-consistency model."""

    def __init__(
        self,
        config: ColorBranchConfig,
        profile: ColorReferenceProfile | None = None,
    ) -> None:
        self.config = config
        self.profile = profile

    def predict(self, image: np.ndarray, valid_mask: np.ndarray) -> ColorAnomalyResult:
        """用学习到的正常颜色分布对单张 ROI 打分。"""
        if self.profile is None:
            raise RuntimeError("颜色分支尚未加载参考分布")

        feature = _extract_color_feature(image, valid_mask)
        normalized_distance = np.linalg.norm(
            (feature - self.profile.feature_mean) / self.profile.feature_std,
        )
        return ColorAnomalyResult(
            score=float(normalized_distance),
            threshold=float(self.profile.threshold),
            is_anomaly=bool(normalized_distance > self.profile.threshold),
            diagnostics={
                "valid_pixel_ratio": float(_valid_pixel_ratio(valid_mask)),
                "mean_l": float(feature[0]),
                "mean_a": float(feature[1]),
                "mean_b": float(feature[2]),
            },
        )


def _extract_color_feature(image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    mask = valid_mask.astype(bool)
    if not np.any(mask):
        raise ValueError("颜色特征提取至少需要一个有效像素")
    pixels = lab[mask]
    return np.concatenate(
        [
            pixels.mean(axis=0),
            pixels.std(axis=0),
        ]
    ).astype(np.float32)


def _valid_pixel_ratio(valid_mask: np.ndarray) -> float:
    if valid_mask.size == 0:
        return 0.0
    return float((valid_mask > 0).mean())
