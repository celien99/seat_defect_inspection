"""基于 YOLO 的目标与忽略区域检测。"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from ultralytics.engine.results import Results

from ..config import DetectionConfig
from ..types import BoundingBox, DetectionObject, DetectionResult


class DetectionService:
    """检测主座椅区域以及需要忽略的干扰区域。"""

    _model_cache: dict[tuple[str, str], Any] = {}

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self._model = None

    def detect(self, image: Any) -> DetectionResult:
        """Run YOLO detection when a model is configured."""
        if self.config.model_path is None:
            return DetectionResult(
                target=None,
                all_objects=[],
            )

        result = self.predict_raw([image])[0]
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

    def detect_many(self, images: list[Any]) -> list[DetectionResult]:
        """Run one shared YOLO model on a batch of camera images."""
        if self.config.model_path is None:
            return [
                DetectionResult(target=None, all_objects=[])
                for _image in images
            ]
        raw_results = self.predict_raw(images)
        results: list[DetectionResult] = []
        for image, raw_result in zip(images, raw_results):
            detections = self._extract_detections(raw_result, image.shape[:2])
            target_candidates = [
                detection
                for detection in detections
                if detection.label == self.config.target_class
            ]
            target = max(target_candidates, key=lambda item: item.confidence, default=None)
            results.append(
                DetectionResult(
                    target=target,
                    all_objects=detections,
                )
            )
        return results

    def predict_raw(self, images: list[Any]) -> list[Results]:
        """Run YOLO and return raw Ultralytics result objects."""
        if not images:
            return []
        model = self._load_model()
        return list(
            model.predict(
                images,
                conf=float(self.config.confidence),
                iou=float(self.config.iou),
                device=self.config.device,
                verbose=False,
            )
        )

    def warmup(self) -> None:
        """Load the shared YOLO model and run a lightweight dummy forward."""
        if self.config.model_path is not None:
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            self.predict_raw([dummy_image])

    def _load_model(self) -> Any:
        if self.config.model_path is None:
            raise RuntimeError("YOLO model_path is not configured")
        if self._model is not None:
            return self._model
        cache_key = (str(self.config.model_path), str(self.config.device))
        model = self._model_cache.get(cache_key)
        if model is None:
            from ultralytics import YOLO

            model = YOLO(self.config.model_path)
            self._model_cache[cache_key] = model
        self._model = model
        return model

    def _detect_from_raw(self, image: Any, result: Results) -> DetectionResult:
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

    def _legacy_predict_single(self, image: Any) -> Results:
        model = self._load_model()
        return model.predict(
            image,
            conf=float(self.config.confidence),
            iou=float(self.config.iou),
            device=self.config.device,
            verbose=False,
        )[0]

    def _extract_detections(
        self,
        result: Results,
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
        result: Results,
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
