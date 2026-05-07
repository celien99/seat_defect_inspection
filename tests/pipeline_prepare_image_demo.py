from __future__ import annotations

from pathlib import Path

import cv2

from demo_utils import ensure_raw_input_path, write_image, write_mask
from seat_defect_core.config import (
    AlignmentConfig,
    DetectionConfig,
    PreprocessConfig,
    QualityGuardConfig,
    RoiRefineConfig,
)
from seat_defect_core.cvops.debug_artifacts import _render_detections
from seat_defect_core.schemas import BoundingBox
from seat_defect_core.service.core import CameraPipeline
from seat_defect_inspection.config import CameraConfig

# 这个脚本只看：
# 1. 单张图片进入当前项目链路
# 2. YOLO 识别成功后，OpenCV 中间层处理结果是什么
# 3. ROI 精修后输出了哪些中间结果
IMAGE_PATH = "datasets/seat_defect/images/val/1.png"
YOLO_MODEL_PATH: str | None = "best.pt"
DEVICE = "cpu"
OUTPUT_DIR = "outputs/pipeline_prepare_image_demo"

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
    "edge_ignore_pixels": 4,
    "alignment": {
        "output_width": 256,
        "output_height": 256,
    },
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
        patchcore_model_path="unused.npz",
        quality=QualityGuardConfig(**QUALITY),
        preprocess=PreprocessConfig(**PREPROCESS),
        detection=detection,
        roi=roi,
    )


def main() -> None:
    image_path = ensure_raw_input_path(Path(IMAGE_PATH))
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"读取图片失败：{image_path}")

    pipeline = CameraPipeline(_build_camera())
    prepared = pipeline.prepare_image(image)

    sample_dir = Path(OUTPUT_DIR) / image_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    write_image(sample_dir / "raw.png", image)
    if prepared.preprocessed_image is not None:
        write_image(sample_dir / "preprocessed.png", prepared.preprocessed_image)
        write_image(
            sample_dir / "detections.png",
            _render_detections(prepared.preprocessed_image, prepared.detection),
        )

    if prepared.roi is not None:
        write_image(sample_dir / "roi.png", prepared.roi.aligned_roi_image)
        if prepared.roi.texture_ready_image is not None:
            write_image(sample_dir / "roi_texture.png", prepared.roi.texture_ready_image)
        patchcore_input = (
            prepared.roi.texture_ready_image
            if prepared.roi.texture_ready_image is not None
            else prepared.roi.aligned_roi_image
        )
        write_image(sample_dir / "patchcore_input.png", patchcore_input)
        write_mask(sample_dir / "target_mask.png", prepared.roi.target_mask)
        write_mask(sample_dir / "valid_mask.png", prepared.roi.valid_mask)

    print(f"图片路径: {image_path}")
    print(f"输出目录: {sample_dir}")
    print(f"拒绝原因: {prepared.rejection_reason}")


if __name__ == "__main__":
    main()
