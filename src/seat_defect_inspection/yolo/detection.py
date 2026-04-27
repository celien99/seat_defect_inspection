"""基于 YOLO 的目标与忽略区域检测。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import DetectionConfig
from ..schemas import BoundingBox, DetectionObject, DetectionResult

YOLO_SEGMENT_TASK = "segment"


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
                all_objects=[],
            )

        if self._model is None:
            _ensure_local_yolo_config_dir()
            from ultralytics import YOLO

            self._model = YOLO(self.config.model_path)
            _validate_model_task(self._model, str(self.config.model_path))

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
        target = max(target_candidates, key=lambda item: item.confidence, default=None)
        return DetectionResult(
            target=target,
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
                    segmentation_mask=masks[index] if index < len(masks) else None,
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


def _validate_model_task(model: Any, model_path: str) -> None:
    task = str(getattr(model, "task", "")).strip().lower()
    if task == YOLO_SEGMENT_TASK:
        return
    display_task = task or "unknown"
    raise ValueError(
        "当前项目只支持 YOLO segmentation 权重，"
        f"但 `{model_path}` 的任务类型是 `{display_task}`。"
        " 请改用 yolo11m-seg.pt 或分割训练产物。"
    )


def _ensure_local_yolo_config_dir() -> None:
    """把 Ultralytics 配置目录收敛到项目内，避免现场环境目录异常影响检测。"""
    if os.getenv("YOLO_CONFIG_DIR"):
        return

    project_runtime_dir = (
        Path(__file__).resolve().parents[3]
        / "outputs"
        / "seat_defect_inspection"
        / "_runtime"
    )
    project_runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(project_runtime_dir)
