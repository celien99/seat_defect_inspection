from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from demo_utils import write_image, write_mask
from seat_defect_inspection.config import (
    AlignmentConfig,
    CameraConfig,
    DetectionConfig,
    PatchCoreConfig,
    PreprocessConfig,
    QualityGuardConfig,
    RoiRefineConfig,
)
from seat_defect_inspection.patchcore import PatchCoreService
from seat_defect_inspection.schemas import BoundingBox
from seat_defect_inspection.service import _CameraPipeline, _overlay_heatmap, _render_detections

# 这个脚本只做：
# 1. 单张图片进入当前项目里的 YOLO + OpenCV + ROI 链路
# 2. 把处理后的结果交给 PatchCore 模型
# 3. 如果本地没有 PatchCore 模型，就先用少量样本拟合一个 demo 模型
IMAGE_PATH = "runs/segment/outputs/yolo_debug/demo_preprocessed/preprocessed.jpg"
FALLBACK_IMAGE_PATHS = [
    "runs/segment/outputs/yolo_debug/demo_preprocessed/preprocessed.jpg",
]
YOLO_MODEL_PATH: str | None = "best.pt"
PATCHCORE_MODEL_PATH = "outputs/patchcore_single_image_inference/patchcore_demo_model.npz"
TRAIN_IMAGE_PATHS = [
    "datasets/seat_defect/images/train/1.png",
    "datasets/seat_defect/images/train/2.png",
    "datasets/seat_defect/images/train/3.png",
]
FORCE_RETRAIN_PATCHCORE = False
DEVICE = "cpu"
OUTPUT_DIR = "outputs/patchcore_single_image_inference"

QUALITY = {
    "min_laplacian_variance": 80.0,
    "min_brightness_mean": 30.0,
    "max_brightness_mean": 225.0,
    "max_overexposed_ratio": 0.25,
    "max_underexposed_ratio": 0.35,
}

PREPROCESS = {
    "denoise_method": "gaussian",
    "gaussian_kernel_size": 5,
    "white_balance_method": "gray_world",
    "max_white_balance_gain": 1.2,
    "apply_illumination_correction": True,
    "illumination_blur_kernel_size": 51,
    "illumination_strength": 0.65,
    "apply_clahe": True,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": 8,
    "sharpen": False,
}

DETECTION = {
    "model_path": YOLO_MODEL_PATH,
    "target_class": "seat",
    "ignore_classes": [
        "tooling",
        "worker_hand",
        "wire",
        "foreign_object",
    ],
    "confidence": 0.5,
    "iou": 0.45,
    "device": DEVICE,
    "fallback_box": {
        "x1": 1116.0,
        "y1": 332.0,
        "x2": 2722.0,
        "y2": 2911.0,
    },
}

ROI = {
    "crop_expand_ratio": 0.02,
    "crop_shrink_ratio": 0.0,
    "mask_mode": "full",
    "morphology_kernel_size": 5,
    "ignore_dilate_kernel_size": 9,
    "edge_ignore_pixels": 4,
    "apply_texture_clahe": True,
    "texture_denoise_method": "bilateral",
    "texture_illumination_correction": True,
    "texture_illumination_blur_kernel_size": 41,
    "texture_illumination_strength": 0.85,
    "mask_feather_kernel_size": 15,
    "edge_enhance_method": "scharr",
    "edge_enhance_weight": 0.18,
    "suppress_background": True,
    "background_fill_mode": "median",
    "alignment": {
        "enabled": False,
        "method": "resize",
        "output_width": 256,
        "output_height": 256,
    },
}

PATCHCORE = {
    "backend": "handcrafted",
    "image_size": 256,
    "patch_size": 32,
    "stride": 16,
    "max_memory": 512,
    "threshold_quantile": 0.99,
    "coreset_sampling_ratio": 0.1,
    "texture_input": "lab_l",
    "min_target_coverage": 0.6,
    "max_ignore_overlap": 0.1,
    "min_valid_patch_ratio": 0.4,
}


def _build_camera() -> CameraConfig:
    detection = DetectionConfig(
        **{
            **DETECTION,
            "fallback_box": BoundingBox(**DETECTION["fallback_box"]),
        }
    )
    roi = RoiRefineConfig(
        **{
            **ROI,
            "alignment": AlignmentConfig(**ROI["alignment"]),
        }
    )
    return CameraConfig(
        camera_id="cam_demo",
        source=IMAGE_PATH,
        patchcore_model_path=PATCHCORE_MODEL_PATH,
        quality=QualityGuardConfig(**QUALITY),
        preprocess=PreprocessConfig(**PREPROCESS),
        detection=detection,
        roi=roi,
        patchcore=PatchCoreConfig(**PATCHCORE),
    )


def _resolve_image_path() -> Path:
    candidate_paths = [IMAGE_PATH, *FALLBACK_IMAGE_PATHS]
    for candidate in candidate_paths:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"读取图片失败，候选路径都不存在：{candidate_paths}")


def _resolve_patchcore_service(
    pipeline: _CameraPipeline,
    camera: CameraConfig,
) -> tuple[PatchCoreService, bool, dict[str, object] | None]:
    model_path = Path(camera.patchcore_model_path)
    if model_path.exists() and not FORCE_RETRAIN_PATCHCORE:
        return PatchCoreService.load_bundle(model_path).patchcore, False, None

    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    used_image_paths: list[str] = []
    skipped_image_paths: list[str] = []

    for train_image_path_str in TRAIN_IMAGE_PATHS:
        train_image_path = Path(train_image_path_str)
        train_image = cv2.imread(str(train_image_path))
        if train_image is None:
            skipped_image_paths.append(str(train_image_path))
            continue

        prepared = pipeline.prepare_image(train_image)
        if prepared.rejection_reason is not None or prepared.roi is None:
            skipped_image_paths.append(f"{train_image_path}:{prepared.rejection_reason}")
            continue

        patchcore_input = (
            prepared.roi.texture_ready_image
            if prepared.roi.texture_ready_image is not None
            else prepared.roi.aligned_roi_image
        )
        samples.append(
            (
                patchcore_input,
                prepared.roi.valid_mask,
                np.zeros_like(prepared.roi.valid_mask, dtype=np.uint8),
            )
        )
        used_image_paths.append(str(train_image_path))

    if not samples:
        raise RuntimeError(
            "没有可用于拟合 demo PatchCore 模型的样本。"
            f" 请检查 TRAIN_IMAGE_PATHS: {TRAIN_IMAGE_PATHS}"
        )

    service = PatchCoreService(camera.patchcore)
    training_summary = service.fit(samples)
    service.save(model_path)
    return (
        service,
        True,
        {
            "model_path": str(model_path),
            "used_image_paths": used_image_paths,
            "skipped_image_paths": skipped_image_paths,
            "training_summary": training_summary,
        },
    )


def main() -> None:
    camera = _build_camera()
    image_path = _resolve_image_path()
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"读取图片失败：{image_path}")

    pipeline = _CameraPipeline(camera)
    prepared = pipeline.prepare_image(image)
    if prepared.rejection_reason is not None or prepared.roi is None:
        raise RuntimeError(f"当前图片在项目链路中被拒绝：{prepared.rejection_reason}")

    patchcore = _resolve_patchcore_service(pipeline, camera)
    patchcore_service, trained_this_run, training_details = patchcore

    patchcore_input = (
        prepared.roi.texture_ready_image
        if prepared.roi.texture_ready_image is not None
        else prepared.roi.aligned_roi_image
    )
    result = patchcore_service.predict(
        patchcore_input,
        prepared.roi.valid_mask,
        np.zeros_like(prepared.roi.valid_mask, dtype=np.uint8),
    )

    sample_dir = Path(OUTPUT_DIR) / image_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    write_image(sample_dir / "raw.png", image)
    if prepared.preprocessed_image is not None:
        write_image(sample_dir / "preprocessed.png", prepared.preprocessed_image)
        write_image(
            sample_dir / "detections.png",
            _render_detections(prepared.preprocessed_image, prepared.detection),
        )
    write_image(sample_dir / "roi.png", prepared.roi.aligned_roi_image)
    write_image(sample_dir / "patchcore_input.png", patchcore_input)
    write_mask(sample_dir / "valid_mask.png", prepared.roi.valid_mask)

    heatmap = np.uint8(np.clip(result.heatmap, 0.0, 1.0) * 255)
    write_image(sample_dir / "heatmap.png", cv2.applyColorMap(heatmap, cv2.COLORMAP_JET))
    write_image(
        sample_dir / "overlay.png",
        _overlay_heatmap(prepared.roi.aligned_roi_image, result.heatmap),
    )

    summary = {
        "image_path": str(image_path),
        "patchcore_model_path": camera.patchcore_model_path,
        "trained_this_run": trained_this_run,
        "score": float(result.score),
        "threshold": float(result.threshold),
        "valid_patch_ratio": float(result.valid_patch_ratio),
        "is_anomaly": bool(result.is_anomaly),
        "training_details": training_details,
    }
    (sample_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"图片路径: {image_path}")
    print(f"输出目录: {sample_dir}")
    print(f"PatchCore 模型: {camera.patchcore_model_path}")
    print(f"本次是否重新拟合模型: {trained_this_run}")
    print(f"PatchCore 分数: {result.score:.6f}")
    print(f"PatchCore 阈值: {result.threshold:.6f}")
    print(f"是否异常: {result.is_anomaly}")


if __name__ == "__main__":
    main()
