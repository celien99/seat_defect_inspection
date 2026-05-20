"""PatchCore 训练流程。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from seat_defect_core.config import (PatchCoreConfig, RegionConfig)
from seat_defect_core.cvops import split_roi_regions
from seat_defect_core.patchcore import ColorConsistencyService
from seat_defect_core.patchcore.features import (
    _TorchPatchFeatureExtractor,
    extract_patch_embeddings,
)
from seat_defect_core.util import (
    format_reason_counter,
    select_patchcore_input,
    write_json,
)

from ..config import CameraConfig
from ..patchcore import PatchCoreTrainer, list_images

if TYPE_CHECKING:
    from .core import CameraPipeline, InspectionService


def train_patchcore_models(
    service: "InspectionService",
    seat_model_id: Optional[str] = None,
    camera_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """按机位训练 PatchCore 模型。"""
    summaries: List[Dict[str, Any]] = []
    for candidate_model_id in _resolve_training_scope(service, seat_model_id):
        context = service.resolve_context(candidate_model_id)
        for camera in context.cameras:
            if camera_id is not None and camera.camera_id != camera_id:
                continue
            summaries.append(
                _train_one_camera(
                    service,
                    seat_model_id=context.seat_model_id,
                    camera=camera,
                    pipeline=context.pipelines[camera.camera_id],
                )
            )
    if camera_id is not None and not summaries:
        raise ValueError(f"未找到机位 `{camera_id}`，请检查配置中的 camera_id 是否正确")
    return summaries


def _train_one_camera(
    service: InspectionService,
    *,
    seat_model_id: Optional[str],
    camera: CameraConfig,
    pipeline: "CameraPipeline",
) -> Dict[str, Any]:
    """训练单个机位的 PatchCore 模型，并按需补充颜色分支。"""
    active_regions = [region for region in camera.regions if region.enabled]
    if active_regions:
        return _train_one_camera_regions(
            service,
            seat_model_id=seat_model_id,
            camera=camera,
            pipeline=pipeline,
        )

    if not camera.train_good_dir:
        raise ValueError(f"机位 `{camera.camera_id}` 缺少 `train_good_dir` 配置")

    train_dir = Path(camera.train_good_dir)
    image_paths = list_images(train_dir)
    if not image_paths:
        raise FileNotFoundError(f"训练目录中没有图像：{train_dir}")

    patchcore_samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    color_samples: List[Tuple[np.ndarray, np.ndarray]] = []
    skipped_images: List[str] = []
    skipped_reason_counter: Counter[str] = Counter()

    yolo_batch_size = 64
    for start in range(0, len(image_paths), yolo_batch_size):
        chunk_paths = image_paths[start : start + yolo_batch_size]
        chunk_images: List[np.ndarray] = []
        path_by_index: List[Path] = []
        for path in chunk_paths:
            image = cv2.imread(str(path))
            if image is None:
                skipped_images.append(str(path))
                skipped_reason_counter["image_read_failed"] += 1
                continue
            chunk_images.append(image)
            path_by_index.append(path)

        if not chunk_images:
            continue

        detections = pipeline.detection_service.detect_many(chunk_images)
        for path, image, detection in zip(path_by_index, chunk_images, detections):
            prepared = pipeline.prepare_from_detection(image, detection)
            if prepared.rejection_reason is not None or prepared.roi is None:
                reason = prepared.rejection_reason or "roi_missing"
                skipped_images.append(str(path))
                skipped_reason_counter[reason] += 1
                continue

            patchcore_samples.append(
                (
                    select_patchcore_input(prepared.roi),
                    prepared.roi.target_mask,
                    prepared.roi.ignore_mask,
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

    patchcore = PatchCoreTrainer(service.resolve_patchcore_config(camera))
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
    color_summary: Optional[Dict[str, Any]] = None
    if camera.color_branch.enabled and not camera.color_insensitive_mode:
        color_service = ColorConsistencyService(camera.color_branch)
        color_summary = color_service.fit(color_samples)
        color_profile = color_service.profile
    elif camera.color_insensitive_mode:
        color_summary = {
            "skipped": True,
            "reason": "color_insensitive_mode",
        }

    patchcore.save(
        camera.patchcore_model_path,
        color_profile=color_profile,
    )
    summary = {
        "seat_model_id": seat_model_id,
        "camera_id": camera.camera_id,
        "model_path": camera.patchcore_model_path,
        "patchcore": patchcore_summary,
        "color_branch": color_summary,
        "train_image_count": len(image_paths),
        "accepted_image_count": len(patchcore_samples),
        "skipped_image_count": len(skipped_images),
        "skipped_reasons": dict(sorted(skipped_reason_counter.items())),
    }
    _write_training_summary(camera.patchcore_model_path, summary)
    return summary


def _train_one_camera_regions(
    service: InspectionService,
    *,
    seat_model_id: Optional[str],
    camera: CameraConfig,
    pipeline: "CameraPipeline",
) -> Dict[str, Any]:
    """按配置区域分别训练一个机位下的多个 PatchCore 模型。"""
    if not camera.train_good_dir:
        raise ValueError(f"机位 `{camera.camera_id}` 缺少 `train_good_dir` 配置")

    train_dir = Path(camera.train_good_dir)
    image_paths = list_images(train_dir)
    if not image_paths:
        raise FileNotFoundError(f"训练目录中没有图像：{train_dir}")

    active_regions = [region for region in camera.regions if region.enabled]
    patchcore_samples_by_region: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        region.region_id: []
        for region in active_regions
    }
    skipped_images: List[str] = []
    skipped_reason_counter: Counter[str] = Counter()
    accepted_image_count = 0

    yolo_batch_size = 64
    for start in range(0, len(image_paths), yolo_batch_size):
        chunk_paths = image_paths[start : start + yolo_batch_size]
        chunk_images: List[np.ndarray] = []
        path_by_index: List[Path] = []
        for path in chunk_paths:
            image = cv2.imread(str(path))
            if image is None:
                skipped_images.append(str(path))
                skipped_reason_counter["image_read_failed"] += 1
                continue
            chunk_images.append(image)
            path_by_index.append(path)

        if not chunk_images:
            continue

        detections = pipeline.detection_service.detect_many(chunk_images)
        for path, image, detection in zip(path_by_index, chunk_images, detections):
            prepared = pipeline.prepare_from_detection(image, detection)
            if prepared.rejection_reason is not None or prepared.roi is None:
                reason = prepared.rejection_reason or "roi_missing"
                skipped_images.append(str(path))
                skipped_reason_counter[reason] += 1
                continue

            sample_by_region = {
                sample.region_id: sample
                for sample in split_roi_regions(prepared.roi, active_regions)
            }
            accepted_region_count = 0
            for region in active_regions:
                sample = sample_by_region.get(region.region_id)
                if sample is None:
                    skipped_reason_counter[f"region_empty:{region.region_id}"] += 1
                    continue
                patchcore_samples_by_region[region.region_id].append(
                    (
                        sample.image,
                        sample.target_mask,
                        sample.ignore_mask,
                    )
                )
                accepted_region_count += 1

            if accepted_region_count <= 0:
                skipped_images.append(str(path))
                skipped_reason_counter["no_region_samples"] += 1
            else:
                accepted_image_count += 1

    # 检查所有区域的特征提取配置是否一致，以决定是否跨区域共享 backbone。
    _FEATURE_EXTRACTION_FIELDS = (
        "backend",
        "image_size",
        "backbone_name",
        "backbone_pretrained",
        "backbone_weights_path",
        "backbone_device",
        "feature_layers",
        "feature_pool_kernel_size",
        "texture_input",
    )
    sample_config = service.resolve_patchcore_config(camera, active_regions[0])
    all_same_extraction_config = all(
        all(
            getattr(service.resolve_patchcore_config(camera, region), field)
            == getattr(sample_config, field)
            for field in _FEATURE_EXTRACTION_FIELDS
        )
        for region in active_regions
    )

    if all_same_extraction_config:
        region_summaries = _train_regions_with_shared_extractor(
            service,
            seat_model_id=seat_model_id,
            camera=camera,
            active_regions=active_regions,
            patchcore_samples_by_region=patchcore_samples_by_region,
            sample_config=sample_config,
            train_dir=train_dir,
            image_paths=image_paths,
            skipped_reason_counter=skipped_reason_counter,
        )
    else:
        region_summaries = _train_regions_independent(
            service,
            seat_model_id=seat_model_id,
            camera=camera,
            active_regions=active_regions,
            patchcore_samples_by_region=patchcore_samples_by_region,
            train_dir=train_dir,
            image_paths=image_paths,
            skipped_reason_counter=skipped_reason_counter,
        )

    summary = {
        "seat_model_id": seat_model_id,
        "camera_id": camera.camera_id,
        "mode": "regions",
        "train_image_count": len(image_paths),
        "accepted_image_count": accepted_image_count,
        "skipped_image_count": len(skipped_images),
        "skipped_reasons": dict(sorted(skipped_reason_counter.items())),
        "regions": region_summaries,
    }
    _write_training_summary(camera.patchcore_model_path, summary)
    return summary


def _train_regions_with_shared_extractor(
    service: InspectionService,
    *,
    seat_model_id: Optional[str],
    camera: CameraConfig,
    active_regions: List[RegionConfig],
    patchcore_samples_by_region: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    sample_config: PatchCoreConfig,
    train_dir: Path,
    image_paths: List[Path],
    skipped_reason_counter: Counter[str],
) -> List[Dict[str, Any]]:
    """跨区域共享特征提取器：同一机位的所有区域共享 backbone 前向。"""
    extractor: "_TorchPatchFeatureExtractor | None" = None
    if sample_config.backend.strip().lower() == "full":
        extractor = _TorchPatchFeatureExtractor(sample_config)

    tagged_samples: List[Tuple[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    for region in active_regions:
        for sample in patchcore_samples_by_region[region.region_id]:
            tagged_samples.append((region.region_id, sample))

    embeddings_by_region: Dict[str, List[np.ndarray]] = {
        region.region_id: [] for region in active_regions
    }
    extraction_batch_size = max(1, min(32, 128 // len(active_regions)))
    for start in range(0, len(tagged_samples), extraction_batch_size):
        chunk = tagged_samples[start : start + extraction_batch_size]
        chunk_samples = [s for _, s in chunk]
        if extractor is not None:
            chunk_results = extractor.extract_many(chunk_samples)
        else:
            chunk_results = [
                extract_patch_embeddings(
                    image, sample_config,
                    target_mask=target_mask, ignore_mask=ignore_mask,
                )
                for image, target_mask, ignore_mask in chunk_samples
            ]
        for (region_id, _), (embeddings, __) in zip(chunk, chunk_results):
            if len(embeddings) > 0:
                embeddings_by_region[region_id].append(embeddings.astype(np.float32))

    region_summaries: List[Dict[str, Any]] = []
    for region in active_regions:
        raw_embeddings = embeddings_by_region[region.region_id]
        if not raw_embeddings:
            raise ValueError(
                "PatchCore 区域训练前没有可用样本。"
                f" 机位：{camera.camera_id}，区域：{region.region_id}，"
                f"训练目录：{train_dir}，原始图像数：{len(image_paths)}，"
                f"跳过原因：{format_reason_counter(skipped_reason_counter)}。"
            )

        patchcore = PatchCoreTrainer(service.resolve_patchcore_config(camera, region))
        if extractor is not None:
            patchcore.set_feature_extractor(extractor)
        try:
            patchcore_summary = patchcore.fit_from_embeddings(raw_embeddings)
        except ValueError as exc:
            if str(exc) != "PatchCore 没有可用的有效训练样本":
                raise
            patchcore_config = service.resolve_patchcore_config(camera, region)
            raise ValueError(
                "PatchCore 区域样本已通过 ROI 阶段，但有效 patch 数仍为 0。"
                f" 机位：{camera.camera_id}，区域：{region.region_id}，"
                f"ROI 样本数：{len(raw_embeddings)}，"
                f"patch 参数：min_target_coverage={patchcore_config.min_target_coverage}, "
                f"max_ignore_overlap={patchcore_config.max_ignore_overlap}, "
                f"min_valid_patch_ratio={patchcore_config.min_valid_patch_ratio}。"
            ) from exc

        _save_region_model(service, camera, region, seat_model_id, patchcore, patchcore_summary, len(raw_embeddings))
        region_summaries.append(
            {
                "region_id": region.region_id,
                "model_path": region.patchcore_model_path,
                "box": [float(value) for value in region.box],
                "patchcore": patchcore_summary,
                "accepted_region_sample_count": len(raw_embeddings),
            }
        )
    return region_summaries


def _train_regions_independent(
    service: InspectionService,
    *,
    seat_model_id: Optional[str],
    camera: CameraConfig,
    active_regions: List[RegionConfig],
    patchcore_samples_by_region: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    train_dir: Path,
    image_paths: List[Path],
    skipped_reason_counter: Counter[str],
) -> List[Dict[str, Any]]:
    """按区域独立训练，用于区域间特征提取配置不一致时的回退路径。"""
    region_summaries: List[Dict[str, Any]] = []
    for region in active_regions:
        samples = patchcore_samples_by_region[region.region_id]
        if not samples:
            raise ValueError(
                "PatchCore 区域训练前没有可用样本。"
                f" 机位：{camera.camera_id}，区域：{region.region_id}，"
                f"训练目录：{train_dir}，原始图像数：{len(image_paths)}，"
                f"跳过原因：{format_reason_counter(skipped_reason_counter)}。"
            )

        patchcore = PatchCoreTrainer(service.resolve_patchcore_config(camera, region))
        try:
            patchcore_summary = patchcore.fit(samples)
        except ValueError as exc:
            if str(exc) != "PatchCore 没有可用的有效训练样本":
                raise
            patchcore_config = service.resolve_patchcore_config(camera, region)
            raise ValueError(
                "PatchCore 区域样本已通过 ROI 阶段，但有效 patch 数仍为 0。"
                f" 机位：{camera.camera_id}，区域：{region.region_id}，"
                f"ROI 样本数：{len(samples)}，"
                f"patch 参数：min_target_coverage={patchcore_config.min_target_coverage}, "
                f"max_ignore_overlap={patchcore_config.max_ignore_overlap}, "
                f"min_valid_patch_ratio={patchcore_config.min_valid_patch_ratio}。"
            ) from exc

        _save_region_model(service, camera, region, seat_model_id, patchcore, patchcore_summary, len(samples))
        region_summaries.append(
            {
                "region_id": region.region_id,
                "model_path": region.patchcore_model_path,
                "box": [float(value) for value in region.box],
                "patchcore": patchcore_summary,
                "accepted_region_sample_count": len(samples),
            }
        )
    return region_summaries


def _save_region_model(
    service: InspectionService,
    camera: CameraConfig,
    region: "RegionConfig",
    seat_model_id: Optional[str],
    patchcore: "PatchCoreTrainer",
    patchcore_summary: Dict[str, Any],
    accepted_region_sample_count: int,
) -> None:
    """保存单个区域的 PatchCore 模型并写入训练摘要。"""
    patchcore.save(
        region.patchcore_model_path,
        color_profile=None,
    )
    region_summary = {
        "region_id": region.region_id,
        "model_path": region.patchcore_model_path,
        "box": [float(value) for value in region.box],
        "patchcore": patchcore_summary,
        "accepted_region_sample_count": accepted_region_sample_count,
    }
    _write_training_summary(region.patchcore_model_path, region_summary)


def _write_training_summary(model_path: str, summary: Dict[str, Any]) -> None:
    """把训练摘要写到模型文件旁边，便于现场排查。"""
    write_json(Path(model_path).with_suffix(".summary.json"), summary)


def _resolve_training_scope(
    service: "InspectionService",
    seat_model_id: Optional[str],
) -> List[Optional[str]]:
    """Resolve the seat-model routes that should be trained."""
    if seat_model_id is not None:
        return [seat_model_id]
    if service.config.seat_models:
        return [seat_model.seat_model_id for seat_model in service.config.seat_models]
    return [None]
