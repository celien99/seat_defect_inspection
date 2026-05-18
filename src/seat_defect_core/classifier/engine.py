"""缺陷分类器推理服务。

以 PatchCore heatmap + ROI 图像为输入，输出多分类缺陷类型预测。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..types.results import DefectClassificationResult, DefectType

if TYPE_CHECKING:
    from ..config import ClassificationConfig


class DefectClassifierService:
    """缺陷分类器推理服务。

    加载预训练的分类模型，接受 PatchCore 热力图和 ROI 图像，
    输出多分类缺陷类型及置信度。
    """

    _DEFECT_CLASS_NAMES: list[str] = [
        DefectType.NONE.value,
        DefectType.SCRATCH.value,
        DefectType.STAIN.value,
        DefectType.WRINKLE.value,
        DefectType.THREAD_JUMP.value,
        DefectType.FOREIGN_MATTER.value,
        DefectType.DENT.value,
        DefectType.COLOR_SHIFT.value,
        DefectType.OTHER.value,
    ]

    def __init__(self, config: "ClassificationConfig") -> None:
        self._config = config
        self._model: object | None = None
        self._device: str = "cpu"
        self._class_names: list[str] = list(self._DEFECT_CLASS_NAMES)
        self._image_size: int = 224
        self._version: str = "unknown"
        self._loaded: bool = False
        self._loaded_mtime_ns: int = 0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def version(self) -> str:
        return self._version

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    @property
    def model_path(self) -> str | None:
        return self._config.model_path

    def is_stale(self) -> bool:
        """检查模型文件是否已被更新（mtime 变化）。"""
        if self._config.model_path is None:
            return False
        try:
            current_mtime = Path(self._config.model_path).stat().st_mtime_ns
        except OSError:
            return False
        return self._loaded and current_mtime != self._loaded_mtime_ns

    def reload(self) -> None:
        """如果模型文件已更新，重新加载。"""
        if self.is_stale():
            self._loaded = False
            self._model = None
        if not self._loaded:
            self.load()

    def load(self) -> None:
        """加载分类模型权重。"""
        import json
        import struct
        from pathlib import Path

        import torch

        if self._config.model_path is None:
            raise ValueError("ClassificationConfig.model_path is not set")

        model_path = Path(self._config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Classifier model not found: {model_path}")

        self._loaded_mtime_ns = model_path.stat().st_mtime_ns

        checkpoint = torch.load(
            str(model_path), map_location="cpu", weights_only=False
        )
        metadata = checkpoint.get("metadata", {})
        self._version = metadata.get("version", "unknown")
        self._class_names = metadata.get("class_names", list(self._DEFECT_CLASS_NAMES))
        self._image_size = metadata.get("image_size", 224)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        num_classes = len(self._class_names)
        backbone_name = metadata.get("backbone", "efficientnet_b0")
        self._model = _build_classifier_model(
            backbone_name=backbone_name,
            num_classes=num_classes,
            in_channels=2,
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self._device)
        self._model.eval()
        self._loaded = True

    def predict(
        self,
        heatmap: np.ndarray,
        roi_image: np.ndarray,
    ) -> list[DefectClassificationResult]:
        """对单张 heatmap + ROI 图像进行缺陷分类。

        Args:
            heatmap: (H, W) float32 异常热力图。
            roi_image: (H, W, C) uint8 BGR ROI 对齐图像。

        Returns:
            按置信度降序排列的分类结果列表。
        """
        if not self._loaded:
            self.load()

        import torch

        input_tensor = self._prepare_input(heatmap, roi_image)
        with torch.no_grad():
            if self._device != "cpu":
                input_tensor = input_tensor.to(self._device)
            logits = self._model(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        results = []
        for idx in np.argsort(probs)[::-1]:
            conf = float(probs[idx])
            if conf < 0.01:
                break
            defect_type = self._class_name_to_enum(self._class_names[idx])
            results.append(
                DefectClassificationResult(
                    defect_type=defect_type,
                    confidence=conf,
                    classifier_version=self._version,
                )
            )

        if not results:
            results.append(
                DefectClassificationResult(
                    defect_type=DefectType.NONE,
                    confidence=0.0,
                    classifier_version=self._version,
                )
            )
        return results

    def predict_primary(
        self,
        heatmap: np.ndarray,
        roi_image: np.ndarray,
    ) -> DefectClassificationResult:
        """返回置信度最高的单个分类结果。"""
        results = self.predict(heatmap, roi_image)
        return results[0]

    def _prepare_input(
        self,
        heatmap: np.ndarray,
        roi_image: np.ndarray,
    ) -> object:
        """将 heatmap + ROI 图像组装为模型输入张量。"""
        import cv2
        import torch

        # 热力图归一化并缩放到输入尺寸
        heatmap_resized = cv2.resize(
            heatmap.astype(np.float32),
            (self._image_size, self._image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        heatmap_normalized = np.clip(heatmap_resized / max(heatmap_resized.max(), 1e-8), 0, 2)

        # ROI 图像提取 L 通道（亮度）以解耦颜色
        if roi_image.ndim == 3 and roi_image.shape[2] >= 3:
            roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        elif roi_image.ndim == 3:
            roi_gray = roi_image[:, :, 0]
        else:
            roi_gray = roi_image
        roi_resized = cv2.resize(
            roi_gray.astype(np.float32),
            (self._image_size, self._image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        roi_normalized = roi_resized / 255.0

        # 组装为 (1, 2, H, W) 张量
        input_tensor = torch.from_numpy(
            np.stack([heatmap_normalized, roi_normalized], axis=0)
        ).float().unsqueeze(0)
        return input_tensor

    @staticmethod
    def _class_name_to_enum(name: str) -> DefectType:
        try:
            return DefectType(name)
        except ValueError:
            return DefectType.OTHER


def _build_classifier_model(
    *,
    backbone_name: str,
    num_classes: int,
    in_channels: int = 2,
) -> object:
    """构建分类器模型。

    支持的 backbone：
    - ``efficientnet_b0``：EfficientNet-B0（默认）
    - ``efficientnet_b1``：EfficientNet-B1
    - ``mobilenet_v3_small``：MobileNetV3-Small
    """
    import torch
    import torch.nn as nn

    if backbone_name.startswith("efficientnet"):
        try:
            from torchvision.models import (
                EfficientNet_B0_Weights,
                EfficientNet_B1_Weights,
                efficientnet_b0,
                efficientnet_b1,
            )
        except ImportError:
            raise ImportError(
                "torchvision>=0.16 required for EfficientNet backbone"
            )

        if backbone_name == "efficientnet_b0":
            backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            feature_dim = 1280
        elif backbone_name == "efficientnet_b1":
            backbone = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
            feature_dim = 1280
        else:
            raise ValueError(f"Unknown EfficientNet backbone: {backbone_name}")

        # 替换第一层以接受 2 通道输入
        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :1] = old_conv.weight.mean(dim=1, keepdim=True) * 0.5
            new_conv.weight[:, 1:2] = old_conv.weight.mean(dim=1, keepdim=True) * 0.5
        backbone.features[0][0] = new_conv

    elif backbone_name == "mobilenet_v3_small":
        try:
            from torchvision.models import (
                MobileNet_V3_Small_Weights,
                mobilenet_v3_small,
            )
        except ImportError:
            raise ImportError(
                "torchvision>=0.12 required for MobileNetV3 backbone"
            )

        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        feature_dim = 576

        # 替换第一层
        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :1] = old_conv.weight.mean(dim=1, keepdim=True) * 0.5
            new_conv.weight[:, 1:2] = old_conv.weight.mean(dim=1, keepdim=True) * 0.5
        backbone.features[0][0] = new_conv
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    # 构建分类头
    backbone.classifier = nn.Identity()
    classifier_head = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(feature_dim, num_classes),
    )
    model = nn.Sequential(backbone, classifier_head)
    return model
