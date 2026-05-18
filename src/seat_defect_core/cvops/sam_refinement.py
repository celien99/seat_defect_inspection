"""SAM (Segment Anything) 缺陷边界精修。

当分类器检测到缺陷后，用热力图峰值区域作为提示点调用 SAM，
生成精确的缺陷分割 mask。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..types.results import DefectClassificationResult

_logger = logging.getLogger(__name__)


class SamRefinementService:
    """SAM 缺陷边界精修服务（懒加载，按进程缓存）。"""

    _instance: SamRefinementService | None = None

    def __init__(self) -> None:
        self._model: Any = None
        self._device: str = "cpu"
        self._loaded: bool = False

    @classmethod
    def get_instance(cls) -> SamRefinementService:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """加载 SAM 模型（首次调用时自动触发）。"""
        import torch

        try:
            from segment_anything import sam_model_build, SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "SAM 需要 segment-anything 包。请执行: pip install segment-anything"
            )

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # 使用轻量的 vit_b 模型（~375MB，推理 ~50ms GPU / ~500ms CPU）
        try:
            sam = sam_model_build("vit_b", checkpoint=None)
        except Exception:
            # 后备：通过 torch.hub 加载
            try:
                sam = torch.hub.load(
                    "facebookresearch/segment-anything",
                    "sam_vit_b_01ec14.pth",
                )
            except Exception:
                raise RuntimeError(
                    "无法加载 SAM 模型。请下载 vit_b checkpoint 或通过 pip install segment-anything 安装。"
                )

        sam.to(self._device)
        sam.eval()
        self._model = sam
        self._loaded = True

    def refine(
        self,
        roi_image: np.ndarray,
        heatmap: np.ndarray,
        *,
        classification_result: "DefectClassificationResult | None" = None,
    ) -> np.ndarray | None:
        """使用 SAM 精修缺陷边界。

        Args:
            roi_image: (H, W, 3) BGR uint8 ROI 对齐图像。
            heatmap: (H, W) float32 异常热力图。
            classification_result: 分类器输出（可选，用于定位提示点）。

        Returns:
            二值缺陷 mask (H, W) uint8，或 None（精修失败时降级）。
        """
        import cv2

        if not self._loaded:
            try:
                self.load()
            except Exception:
                _logger.warning("SAM 模型加载失败，跳过缺陷精修", exc_info=True)
                return None

        h, w = roi_image.shape[:2]

        # 从热力图中确定提示点：取热力图峰值位置
        if heatmap.size == 0 or heatmap.max() <= 0:
            return None

        peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

        # 将提示点映射到 SAM 输入尺寸 (1024x1024)
        sam_size = 1024
        scale_x = sam_size / w
        scale_y = sam_size / h
        prompt_point = np.array([[peak_x * scale_x, peak_y * scale_y]])

        # 准备 SAM 输入
        import torch

        roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        sam_input = cv2.resize(roi_rgb, (sam_size, sam_size))
        input_tensor = (
            torch.from_numpy(sam_input)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(self._device)
        )

        with torch.inference_mode():
            image_embedding = self._model.image_encoder(input_tensor)

            point_coords = (
                torch.from_numpy(prompt_point)
                .float()
                .unsqueeze(0)
                .to(self._device)
            )
            point_labels = torch.ones(
                (1, prompt_point.shape[0]),
                dtype=torch.long,
                device=self._device,
            )

            masks, scores, _ = self._model.prompt_encoder.predict(
                image_embedding=image_embedding,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )

        # 取最高分 mask，缩放到原始 ROI 尺寸
        best_idx = int(scores.argmax())
        mask = masks[best_idx, 0].cpu().numpy().astype(np.float32)
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

        return binary_mask


def refine_defect_boundary(
    roi_image: np.ndarray,
    heatmap: np.ndarray,
    *,
    classification_result: "DefectClassificationResult | None" = None,
) -> np.ndarray | None:
    """便捷函数：调用 SAM 精修缺陷边界。"""
    return SamRefinementService.get_instance().refine(
        roi_image,
        heatmap,
        classification_result=classification_result,
    )
