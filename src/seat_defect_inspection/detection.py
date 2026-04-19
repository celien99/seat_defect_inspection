"""基于 YOLO 的目标与忽略区域检测。"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .config import DetectionConfig
from .schemas import BoundingBox, DetectionObject, DetectionResult


class DetectionService:
    """检测主座椅区域以及需要忽略的干扰区域。"""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self._model = None

    def detect(self, image: Any) -> DetectionResult:
        """有 YOLO 权重时执行检测，否则退回到静态框。"""
        if self.config.model_path is None:
            return DetectionResult(
                target=self._build_fallback_target(),
                ignores=[],
                all_objects=[],
            )

        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.config.model_path)

        result = self._model.predict(
            image,
            conf=float(self.config.confidence),
            iou=float(self.config.iou),
            device=self.config.device,
            verbose=False,
        )[0]

        detections = self._extract_detections(result, image.shape[:2])
        target_candidates = [
            detection
            for detection in detections
            if detection.label == self.config.target_class
        ]
        ignore_candidates = [
            detection
            for detection in detections
            if detection.label in self.config.ignore_classes
        ]

        target = max(target_candidates, key=lambda item: item.confidence, default=None)
        if target is None:
            target = self._build_fallback_target()

        return DetectionResult(
            target=target,
            ignores=ignore_candidates,
            all_objects=detections,
        )

    def _build_fallback_target(self) -> DetectionObject | None:
        if self.config.fallback_box is None:
            return None
        return DetectionObject(
            label=self.config.target_class,
            confidence=1.0,
            bounding_box=self.config.fallback_box,
            segmentation_mask=None,
        )

    def _extract_detections(
        self,
        result: Any,
        image_shape: tuple[int, int],
    ) -> list[DetectionObject]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = (
            boxes.conf.cpu().numpy()
            if getattr(boxes, "conf", None) is not None
            else np.ones((len(xyxy),), dtype=np.float32)
        )
        classes = (
            boxes.cls.cpu().numpy().astype(int)
            if getattr(boxes, "cls", None) is not None
            else np.zeros((len(xyxy),), dtype=np.int32)
        )
        names = getattr(result, "names", {}) or {}
        masks = self._extract_masks(result, image_shape)

        detections: list[DetectionObject] = []
        for index, box in enumerate(xyxy):
            class_id = int(classes[index])
            if isinstance(names, dict):
                label = names.get(class_id, str(class_id))
            elif isinstance(names, list) and 0 <= class_id < len(names):
                label = names[class_id]
            else:
                label = str(class_id)
            detections.append(
                DetectionObject(
                    label=str(label),
                    confidence=float(confidences[index]),
                    bounding_box=BoundingBox(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                    ),
                    segmentation_mask=(
                        masks[index]
                        if self.config.prefer_segmentation_mask and index < len(masks)
                        else None
                    ),
                ),
            )
        return detections

    def _extract_masks(
        self,
        result: Any,
        image_shape: tuple[int, int],
    ) -> list[np.ndarray]:
        mask_data = getattr(getattr(result, "masks", None), "data", None)
        if mask_data is None:
            return []

        height, width = image_shape
        masks: list[np.ndarray] = []
        for item in mask_data.cpu().numpy():
            resized = cv2.resize(
                item.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            masks.append((resized > 0.5).astype(np.uint8))
        return masks
