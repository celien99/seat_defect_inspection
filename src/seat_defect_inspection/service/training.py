"""PatchCore 训练流程。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import CameraConfig
from ..patchcore import ColorConsistencyService, list_images
from ..util import format_reason_counter, select_patchcore_input, write_json
from .core import InspectionService, _CameraPipeline


def train_patchcore_models(
    service: InspectionService,
    seat_model_id: str | None = None,
) -> list[dict[str, Any]]:
    """按机位训练 PatchCore 模型。"""
    summaries: list[dict[str, Any]] = []
    for candidate_model_id in service._resolve_training_scope(seat_model_id):
        context = service._resolve_context(candidate_model_id)
        for camera in context.cameras:
            summaries.append(
                _train_one_camera(
                    service,
                    seat_model_id=context.seat_model_id,
                    camera=camera,
                    pipeline=context.pipelines[camera.camera_id],
                )
            )
    return summaries


def _train_one_camera(
    service: InspectionService,
    *,
    seat_model_id: str | None,
    camera: CameraConfig,
    pipeline: _CameraPipeline,
) -> dict[str, Any]:
    """训练单个机位的 PatchCore 模型，并按需补充颜色分支。"""
    if not camera.train_good_dir:
        raise ValueError(f"机位 `{camera.camera_id}` 缺少 `train_good_dir` 配置")

    train_dir = Path(camera.train_good_dir)
    image_paths = list_images(train_dir)
    if not image_paths:
        raise FileNotFoundError(f"训练目录中没有图像：{train_dir}")

    patchcore_samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    color_samples: list[tuple[np.ndarray, np.ndarray]] = []
    skipped_images: list[str] = []
    skipped_reason_counter: Counter[str] = Counter()

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            skipped_images.append(str(image_path))
            skipped_reason_counter["image_read_failed"] += 1
            continue

        prepared = pipeline.prepare_image(image)
        if prepared.rejection_reason is not None or prepared.roi is None:
            skipped_images.append(str(image_path))
            skipped_reason_counter[prepared.rejection_reason or "roi_missing"] += 1
            continue

        patchcore_samples.append(
            (
                select_patchcore_input(prepared.roi),
                prepared.roi.valid_mask,
                np.zeros_like(prepared.roi.valid_mask, dtype=np.uint8),
            )
        )
        color_samples.append(
            (
                prepared.roi.aligned_roi_image,
                prepared.roi.valid_mask,
            )
        )

    if not patchcore_samples:
        raise ValueError(
            "PatchCore 训练前没有可用的 ROI 样本。"
            f" 机位：{camera.camera_id}，训练目录：{train_dir}，"
            f"原始图像数：{len(image_paths)}，"
            f"跳过图像数：{len(skipped_images)}，"
            f"跳过原因：{format_reason_counter(skipped_reason_counter)}。"
            " 请优先检查：1) YOLO 是否检出 target_class；"
            "2) 采图亮度/清晰度是否触发质量门控；"
            "3) ROI 精修后的有效区域是否正常。"
        )

    patchcore = service._build_patchcore_service(camera)
    try:
        patchcore_summary = patchcore.fit(patchcore_samples)
    except ValueError as exc:
        if str(exc) != "PatchCore 没有可用的有效训练样本":
            raise
        raise ValueError(
            "PatchCore 训练样本已通过 ROI 阶段，但有效 patch 数仍为 0。"
            f" 机位：{camera.camera_id}，训练目录：{train_dir}，"
            f"ROI 样本数：{len(patchcore_samples)}，"
            f"跳过图像数：{len(skipped_images)}，"
            f"跳过原因：{format_reason_counter(skipped_reason_counter)}，"
            f"patch 参数：min_target_coverage={camera.patchcore.min_target_coverage}, "
            f"max_ignore_overlap={camera.patchcore.max_ignore_overlap}, "
            f"min_valid_patch_ratio={camera.patchcore.min_valid_patch_ratio}。"
            " 这通常说明 ROI 掩膜太碎、有效前景过小，或 patch 阈值过严。"
        ) from exc

    color_profile = None
    color_summary: dict[str, Any] | None = None
    if camera.color_branch.enabled and not camera.color_insensitive_mode:
        color_service = ColorConsistencyService(camera.color_branch)
        color_summary = color_service.fit(color_samples)
        color_profile = color_service.profile
    elif camera.color_insensitive_mode:
        color_summary = {
            "skipped": True,
            "reason": "color_insensitive_mode",
        }

    patchcore.save(camera.patchcore_model_path, color_profile=color_profile)
    summary = {
        "seat_model_id": seat_model_id,
        "camera_id": camera.camera_id,
        "model_path": camera.patchcore_model_path,
        "patchcore": patchcore_summary,
        "color_branch": color_summary,
        "skipped_image_count": len(skipped_images),
    }
    _write_training_summary(camera.patchcore_model_path, summary)
    return summary


def _write_training_summary(model_path: str, summary: dict[str, Any]) -> None:
    """把训练摘要写到模型文件旁边，便于现场排查。"""
    write_json(Path(model_path).with_suffix(".summary.json"), summary)
