"""缺陷分类器工具层训练编排。

回放检测 pipeline 生成 heatmap + ROI 训练样本，再调用核心训练器。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from seat_defect_core.service.core import InspectionService

    from ..config import InspectionConfig


def train_classifier_models(
    service: "InspectionService",
    seat_model_id: str | None = None,
    *,
    dataset_dir: str,
    output_path: str | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    backbone: str = "efficientnet_b0",
) -> dict[str, Any]:
    """从标注数据集训练缺陷分类器。

    数据集目录结构：
        {dataset_dir}/
            scratch/       ← 划痕缺陷样本
            stain/         ← 污渍缺陷样本
            wrinkle/       ← 褶皱缺陷样本
            thread_jump/   ← 跳针缺陷样本
            foreign_matter/ ← 异物缺陷样本
            dent/          ← 凹陷缺陷样本
            color_shift/   ← 颜色异常样本
            other/         ← 其他缺陷样本
            none/          ← 正常样本（可选）
            good/          ← 正常样本（可选，与 none/ 等价）
    """
    from collections import Counter

    import cv2

    from seat_defect_core.classifier.training import DefectClassifierTrainer

    dataset_path = Path(dataset_dir)
    if not dataset_path.is_dir():
        raise ValueError(f"数据集目录不存在: {dataset_dir}")

    config = service.config
    context = service.resolve_context(seat_model_id)

    # 发现标注样本
    class_names = _DEFAULT_CLASS_NAMES
    samples: list[tuple[np.ndarray, np.ndarray, str]] = []

    for class_name in class_names:
        # "none" 类型同时检查 none/ 和 good/ 目录
        if class_name == "none":
            class_dirs = [dataset_path / "none", dataset_path / "good"]
        else:
            class_dirs = [dataset_path / class_name]

        for class_dir in class_dirs:
            if not class_dir.is_dir():
                continue
            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
                    continue
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                # 对每张图像，用第一个启用相机回放 pipeline 获取 heatmap
                roi_image, heatmap = _replay_pipeline_for_sample(
                    service, context, image
                )
                if roi_image is not None and heatmap is not None:
                    samples.append((heatmap, roi_image, class_name))
                else:
                    # 样本即使没有生成 heatmap，也保留 ROI 用于分类
                    pass

    if not samples:
        raise RuntimeError(
            f"未在 {dataset_dir} 中发现可用训练样本。"
            " 请确保数据集包含至少一个缺陷类别的子目录。"
        )

    label_counts = Counter(label for _, _, label in samples)
    print(f"训练样本分布: {dict(label_counts)}")

    # 确定输出路径
    if output_path is None:
        if context.cameras:
            first_cam = context.cameras[0]
            output_path = first_cam.classification.model_path
        if output_path is None:
            output_path = str(dataset_path / "defect_classifier.pt")

    # 训练
    from seat_defect_core.config import ClassificationConfig

    classifier_config = ClassificationConfig(
        enabled=True,
        model_path=output_path,
    )
    trainer = DefectClassifierTrainer(
        config=classifier_config,
        backbone_name=backbone,
        image_size=224,
        class_names=class_names,
        device="cpu",
    )

    metrics = trainer.fit(
        samples,
        val_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=0.001,
        weight_decay=1e-4,
        focal_loss_gamma=2.0,
        patience=10,
        augment=True,
    )

    trainer.save(
        output_path,
        metadata={
            "dataset_dir": dataset_dir,
            "num_samples": len(samples),
            "label_counts": dict(label_counts),
        },
    )

    summary = {
        "output_path": output_path,
        "num_samples": len(samples),
        "label_counts": dict(label_counts),
        **{k: v for k, v in metrics.items()},
    }
    return summary


def _replay_pipeline_for_sample(
    service: "InspectionService",
    context: object,
    image: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """对单张标注图像回放检测 pipeline，返回 (roi_image, heatmap)。"""
    import cv2

    if not context.cameras:
        return None, None

    camera = context.cameras[0]
    pipeline = context.pipelines[camera.camera_id]

    try:
        detection = pipeline.detection_service.detect(image, frame_id="classifier_sample")
    except Exception:
        return None, None

    if detection.target is None:
        return None, None

    prepared = pipeline.prepare_from_detection(image, detection)
    if prepared.roi is None:
        return None, None

    roi_image = prepared.roi.aligned_roi_image

    # 生成 heatmap：需要运行 PatchCore predict
    try:
        model_bundle = service.load_model_bundle(camera, context.seat_model_id)
        service.prepare_patchcore_for_predict(model_bundle.patchcore)
        from seat_defect_core.util import select_patchcore_input

        texture_input = select_patchcore_input(prepared.roi)
        texture_result = model_bundle.patchcore.predict(
            texture_input,
            prepared.roi.target_mask,
            prepared.roi.ignore_mask,
        )
        heatmap = np.asarray(texture_result.heatmap, dtype=np.float32)
    except Exception:
        # 如果 PatchCore 失败，生成一个空 heatmap
        if roi_image.ndim == 3:
            heatmap = np.zeros(roi_image.shape[:2], dtype=np.float32)
        else:
            heatmap = np.zeros_like(roi_image, dtype=np.float32)

    return roi_image, heatmap


_DEFAULT_CLASS_NAMES = [
    "none",
    "scratch",
    "stain",
    "wrinkle",
    "thread_jump",
    "foreign_matter",
    "dent",
    "color_shift",
    "other",
]
