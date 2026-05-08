"""PatchCore 训练流程。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from seat_defect_core.cvops import split_roi_regions
from seat_defect_core.patchcore import ColorConsistencyService, list_images
from seat_defect_core.util import (
    format_reason_counter,
    select_patchcore_input,
    write_image,
    write_json,
)

from ..config import CameraConfig

if TYPE_CHECKING:
    from .core import CameraPipeline, InspectionService


def train_patchcore_models(
    service: "InspectionService",
    seat_model_id: str | None = None,
) -> list[dict[str, Any]]:
    """按机位训练 PatchCore 模型。"""
    summaries: list[dict[str, Any]] = []
    for candidate_model_id in _resolve_training_scope(service, seat_model_id):
        context = service.resolve_context(candidate_model_id)
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
    pipeline: "CameraPipeline",
) -> dict[str, Any]:
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

    patchcore_samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    color_samples: list[tuple[np.ndarray, np.ndarray]] = []
    skipped_images: list[str] = []
    skipped_reason_counter: Counter[str] = Counter()
    audit_dir = _build_training_audit_dir(camera.patchcore_model_path)
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_records: list[dict[str, Any]] = []

    for image_index, image_path in enumerate(image_paths):
        audit_record: dict[str, Any] = {
            "image_path": str(image_path),
            "status": "skipped",
            "reason": None,
        }
        image = cv2.imread(str(image_path))
        if image is None:
            skipped_images.append(str(image_path))
            skipped_reason_counter["image_read_failed"] += 1
            audit_record["reason"] = "image_read_failed"
            audit_records.append(audit_record)
            continue

        prepared = pipeline.prepare_image(image)
        if prepared.rejection_reason is not None or prepared.roi is None:
            reason = prepared.rejection_reason or "roi_missing"
            skipped_images.append(str(image_path))
            skipped_reason_counter[reason] += 1
            audit_record["reason"] = reason
            if prepared.roi is not None:
                audit_record.update(
                    _write_training_audit_artifacts(
                        audit_dir,
                        image_index=image_index,
                        image_path=image_path,
                        prepared=prepared,
                    )
                )
                audit_record.update(_build_training_audit_metrics(prepared))
            audit_records.append(audit_record)
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
        audit_record["status"] = "accepted"
        audit_record.update(
            _write_training_audit_artifacts(
                audit_dir,
                image_index=image_index,
                image_path=image_path,
                prepared=prepared,
            )
        )
        audit_record.update(_build_training_audit_metrics(prepared))
        audit_records.append(audit_record)

    if not patchcore_samples:
        audit_records_path = _write_training_audit_records(audit_dir, audit_records)
        raise ValueError(
            "PatchCore 训练前没有可用的 ROI 样本。"
            f" 机位：{camera.camera_id}，训练目录：{train_dir}，"
            f"原始图像数：{len(image_paths)}，"
            f"跳过图像数：{len(skipped_images)}，"
            f"跳过原因：{format_reason_counter(skipped_reason_counter)}。"
            f" 训练审计目录：{audit_dir}，记录：{audit_records_path}。"
            " 请优先检查：1) YOLO 是否检出 target_class；"
            "2) 采图亮度/清晰度是否触发质量门控；"
            "3) ROI 精修后的有效区域是否正常。"
        )

    patchcore = service.build_patchcore_service(camera)
    try:
        patchcore_summary = patchcore.fit(patchcore_samples)
    except ValueError as exc:
        if str(exc) != "PatchCore 没有可用的有效训练样本":
            raise
        audit_records_path = _write_training_audit_records(audit_dir, audit_records)
        raise ValueError(
            "PatchCore 训练样本已通过 ROI 阶段，但有效 patch 数仍为 0。"
            f" 机位：{camera.camera_id}，训练目录：{train_dir}，"
            f"ROI 样本数：{len(patchcore_samples)}，"
            f"跳过图像数：{len(skipped_images)}，"
            f"跳过原因：{format_reason_counter(skipped_reason_counter)}，"
            f"训练审计目录：{audit_dir}，记录：{audit_records_path}，"
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

    patchcore_pipeline_context = service.build_patchcore_pipeline_context(camera)
    patchcore_pipeline_signature = service.build_patchcore_pipeline_signature(camera)
    patchcore.save(
        camera.patchcore_model_path,
        color_profile=color_profile,
        pipeline_signature=patchcore_pipeline_signature,
        pipeline_context=patchcore_pipeline_context,
    )
    audit_records_path = _write_training_audit_records(audit_dir, audit_records)
    summary = {
        "seat_model_id": seat_model_id,
        "camera_id": camera.camera_id,
        "model_path": camera.patchcore_model_path,
        "pipeline_signature": patchcore_pipeline_signature,
        "patchcore": patchcore_summary,
        "color_branch": color_summary,
        "train_image_count": len(image_paths),
        "accepted_image_count": len(patchcore_samples),
        "skipped_image_count": len(skipped_images),
        "skipped_reasons": dict(sorted(skipped_reason_counter.items())),
        "training_audit_dir": str(audit_dir),
        "training_audit_records_path": str(audit_records_path),
        "training_audit_records": audit_records,
    }
    _write_training_summary(camera.patchcore_model_path, summary)
    return summary


def _train_one_camera_regions(
    service: InspectionService,
    *,
    seat_model_id: str | None,
    camera: CameraConfig,
    pipeline: "CameraPipeline",
) -> dict[str, Any]:
    """按配置区域分别训练一个机位下的多个 PatchCore 模型。"""
    if not camera.train_good_dir:
        raise ValueError(f"机位 `{camera.camera_id}` 缺少 `train_good_dir` 配置")

    train_dir = Path(camera.train_good_dir)
    image_paths = list_images(train_dir)
    if not image_paths:
        raise FileNotFoundError(f"训练目录中没有图像：{train_dir}")

    active_regions = [region for region in camera.regions if region.enabled]
    patchcore_samples_by_region: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        region.region_id: []
        for region in active_regions
    }
    skipped_images: list[str] = []
    skipped_reason_counter: Counter[str] = Counter()
    audit_dir = _build_training_audit_dir(camera.patchcore_model_path)
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_records: list[dict[str, Any]] = []

    for image_index, image_path in enumerate(image_paths):
        audit_record: dict[str, Any] = {
            "image_path": str(image_path),
            "status": "skipped",
            "reason": None,
            "regions": {},
        }
        image = cv2.imread(str(image_path))
        if image is None:
            skipped_images.append(str(image_path))
            skipped_reason_counter["image_read_failed"] += 1
            audit_record["reason"] = "image_read_failed"
            audit_records.append(audit_record)
            continue

        prepared = pipeline.prepare_image(image)
        if prepared.rejection_reason is not None or prepared.roi is None:
            reason = prepared.rejection_reason or "roi_missing"
            skipped_images.append(str(image_path))
            skipped_reason_counter[reason] += 1
            audit_record["reason"] = reason
            if prepared.roi is not None:
                audit_record.update(
                    _write_training_audit_artifacts(
                        audit_dir,
                        image_index=image_index,
                        image_path=image_path,
                        prepared=prepared,
                    )
                )
                audit_record.update(_build_training_audit_metrics(prepared))
            audit_records.append(audit_record)
            continue

        sample_by_region = {
            sample.region_id: sample
            for sample in split_roi_regions(prepared.roi, active_regions)
        }
        accepted_region_count = 0
        for region in active_regions:
            sample = sample_by_region.get(region.region_id)
            if sample is None:
                audit_record["regions"][region.region_id] = {
                    "status": "skipped",
                    "reason": "region_empty",
                }
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
            audit_record["regions"][region.region_id] = {
                "status": "accepted",
                "box": {
                    "x1": float(sample.box.x1),
                    "y1": float(sample.box.y1),
                    "x2": float(sample.box.x2),
                    "y2": float(sample.box.y2),
                },
                "target_pixel_count": int(np.asarray(sample.target_mask > 0).sum()),
                "valid_pixel_count": int(np.asarray(sample.valid_mask > 0).sum()),
            }

        if accepted_region_count <= 0:
            skipped_images.append(str(image_path))
            skipped_reason_counter["no_region_samples"] += 1
            audit_record["reason"] = "no_region_samples"
        else:
            audit_record["status"] = "accepted"
            audit_record.update(
                _write_training_audit_artifacts(
                    audit_dir,
                    image_index=image_index,
                    image_path=image_path,
                    prepared=prepared,
                )
            )
            audit_record.update(_build_training_audit_metrics(prepared))
        audit_records.append(audit_record)

    region_summaries: list[dict[str, Any]] = []
    for region in active_regions:
        samples = patchcore_samples_by_region[region.region_id]
        if not samples:
            audit_records_path = _write_training_audit_records(audit_dir, audit_records)
            raise ValueError(
                "PatchCore 区域训练前没有可用样本。"
                f" 机位：{camera.camera_id}，区域：{region.region_id}，"
                f"训练目录：{train_dir}，原始图像数：{len(image_paths)}，"
                f"跳过原因：{format_reason_counter(skipped_reason_counter)}，"
                f"训练审计目录：{audit_dir}，记录：{audit_records_path}。"
            )

        patchcore = service.build_patchcore_service(camera, region)
        try:
            patchcore_summary = patchcore.fit(samples)
        except ValueError as exc:
            if str(exc) != "PatchCore 没有可用的有效训练样本":
                raise
            audit_records_path = _write_training_audit_records(audit_dir, audit_records)
            patchcore_config = service.resolve_patchcore_config(camera, region)
            raise ValueError(
                "PatchCore 区域样本已通过 ROI 阶段，但有效 patch 数仍为 0。"
                f" 机位：{camera.camera_id}，区域：{region.region_id}，"
                f"ROI 样本数：{len(samples)}，"
                f"训练审计目录：{audit_dir}，记录：{audit_records_path}，"
                f"patch 参数：min_target_coverage={patchcore_config.min_target_coverage}, "
                f"max_ignore_overlap={patchcore_config.max_ignore_overlap}, "
                f"min_valid_patch_ratio={patchcore_config.min_valid_patch_ratio}。"
            ) from exc

        patchcore_pipeline_context = service.build_region_patchcore_pipeline_context(camera, region)
        patchcore_pipeline_signature = service.build_region_patchcore_pipeline_signature(camera, region)
        patchcore.save(
            region.patchcore_model_path,
            color_profile=None,
            pipeline_signature=patchcore_pipeline_signature,
            pipeline_context=patchcore_pipeline_context,
        )
        region_summary = {
            "region_id": region.region_id,
            "model_path": region.patchcore_model_path,
            "box": [float(value) for value in region.box],
            "pipeline_signature": patchcore_pipeline_signature,
            "patchcore": patchcore_summary,
            "accepted_region_sample_count": len(samples),
        }
        _write_training_summary(region.patchcore_model_path, region_summary)
        region_summaries.append(region_summary)

    audit_records_path = _write_training_audit_records(audit_dir, audit_records)
    summary = {
        "seat_model_id": seat_model_id,
        "camera_id": camera.camera_id,
        "mode": "regions",
        "train_image_count": len(image_paths),
        "accepted_image_count": sum(1 for item in audit_records if item.get("status") == "accepted"),
        "skipped_image_count": len(skipped_images),
        "skipped_reasons": dict(sorted(skipped_reason_counter.items())),
        "regions": region_summaries,
        "training_audit_dir": str(audit_dir),
        "training_audit_records_path": str(audit_records_path),
        "training_audit_records": audit_records,
    }
    _write_training_summary(camera.patchcore_model_path, summary)
    return summary


def _write_training_summary(model_path: str, summary: dict[str, Any]) -> None:
    """把训练摘要写到模型文件旁边，便于现场排查。"""
    write_json(Path(model_path).with_suffix(".summary.json"), summary)


def _build_training_audit_dir(model_path: str) -> Path:
    """Return the artifact directory that documents one PatchCore training run."""
    return Path(model_path).with_suffix(".training_audit")


def _write_training_audit_records(
    audit_dir: Path,
    records: list[dict[str, Any]],
) -> Path:
    """Write per-image audit records independently of final training success."""
    records_path = audit_dir / "records.json"
    write_json(records_path, {"records": records})
    return records_path


def _write_training_audit_artifacts(
    audit_dir: Path,
    *,
    image_index: int,
    image_path: Path,
    prepared,
) -> dict[str, Any]:
    """Persist the exact ROI artifacts used to accept or reject one training image."""
    if prepared.roi is None:
        return {}

    sample_dir = audit_dir / f"{image_index:05d}_{_sanitize_path_stem(image_path.stem)}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    patchcore_input = select_patchcore_input(prepared.roi)
    artifacts = {
        "aligned_roi": sample_dir / "aligned_roi.png",
        "patchcore_input": sample_dir / "patchcore_input.png",
        "target_mask": sample_dir / "target_mask.png",
        "valid_mask": sample_dir / "valid_mask.png",
        "ignore_mask": sample_dir / "ignore_mask.png",
    }
    write_image(artifacts["aligned_roi"], prepared.roi.aligned_roi_image)
    write_image(artifacts["patchcore_input"], patchcore_input)
    _write_mask(artifacts["target_mask"], prepared.roi.target_mask)
    _write_mask(artifacts["valid_mask"], prepared.roi.valid_mask)
    _write_mask(artifacts["ignore_mask"], prepared.roi.ignore_mask)

    return {
        "audit_sample_dir": str(sample_dir),
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }


def _build_training_audit_metrics(prepared) -> dict[str, Any]:
    """Collect compact per-image diagnostics for training set review."""
    roi = prepared.roi
    metrics: dict[str, Any] = {}
    if roi is not None:
        target_pixels = int(np.asarray(roi.target_mask > 0).sum())
        valid_pixels = int(np.asarray(roi.valid_mask > 0).sum())
        ignore_pixels = int(np.asarray(roi.ignore_mask > 0).sum())
        metrics.update(
            {
                "target_pixel_count": target_pixels,
                "valid_pixel_count": valid_pixels,
                "ignore_pixel_count": ignore_pixels,
                "valid_pixel_ratio": (
                    float(valid_pixels) / float(max(1, target_pixels))
                ),
            }
        )
        if roi.crop_box is not None:
            metrics["crop_box"] = {
                "x1": float(roi.crop_box.x1),
                "y1": float(roi.crop_box.y1),
                "x2": float(roi.crop_box.x2),
                "y2": float(roi.crop_box.y2),
            }
    if prepared.quality is not None:
        metrics["quality"] = {
            "accepted": bool(prepared.quality.accepted),
            "reason": prepared.quality.reason,
            "metrics": {
                "laplacian_variance": float(prepared.quality.metrics.laplacian_variance),
                "brightness_mean": float(prepared.quality.metrics.brightness_mean),
                "overexposed_ratio": float(prepared.quality.metrics.overexposed_ratio),
                "underexposed_ratio": float(prepared.quality.metrics.underexposed_ratio),
                "is_black_frame": bool(prepared.quality.metrics.is_black_frame),
                "is_white_frame": bool(prepared.quality.metrics.is_white_frame),
            },
        }
    if prepared.detection is not None and prepared.detection.target is not None:
        target = prepared.detection.target
        metrics["detection"] = {
            "target_label": target.label,
            "target_confidence": float(target.confidence),
            "used_fallback": bool(prepared.detection.used_fallback),
            "has_segmentation_mask": target.segmentation_mask is not None,
        }
    return metrics


def _write_mask(path: Path, mask: np.ndarray) -> None:
    """Write a binary or soft mask as a readable uint8 PNG."""
    array = np.asarray(mask)
    if array.dtype == np.uint8:
        normalized = np.where(array > 0, 255, 0).astype(np.uint8)
    else:
        normalized = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    write_image(path, normalized)


def _sanitize_path_stem(value: str) -> str:
    """Keep audit folders stable and filesystem-safe."""
    sanitized = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value
    ).strip("_")
    return sanitized or "image"


def _resolve_training_scope(
    service: "InspectionService",
    seat_model_id: str | None,
) -> list[str | None]:
    """Resolve the seat-model routes that should be trained."""
    if seat_model_id is not None:
        return [seat_model_id]
    if service.config.seat_models:
        return [seat_model.seat_model_id for seat_model in service.config.seat_models]
    return [None]
