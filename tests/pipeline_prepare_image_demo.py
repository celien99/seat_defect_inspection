from __future__ import annotations

from pathlib import Path

import cv2

from demo_utils import load_camera, resolve_image_path, write_image, write_mask
from seat_defect_inspection.service import _CameraPipeline, _render_detections

# 这个脚本只看：
# 1. 单张图片进入当前项目链路
# 2. YOLO 识别成功后，OpenCV 中间层处理结果是什么

CONFIG_PATH = "configs/seat_defect_inspection.mvs.json"
CAMERA_ID = "cam_0"
SEAT_MODEL_ID: str | None = None
YOLO_MODEL_PATH: str | None = "best.pt"
DEVICE: str | None = "cpu"
IMAGE_DIR = "datasets/seat_defect/images/val"
IMAGE_PATH: str | None = None
OUTPUT_DIR = "outputs/pipeline_prepare_image_demo"


def main() -> None:
    camera = load_camera(
        config_path=CONFIG_PATH,
        camera_id=CAMERA_ID,
        seat_model_id=SEAT_MODEL_ID,
        yolo_model_path=YOLO_MODEL_PATH,
        device=DEVICE,
    )
    image_path = resolve_image_path(image_path=IMAGE_PATH, image_dir=IMAGE_DIR)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"读取图片失败：{image_path}")

    pipeline = _CameraPipeline(camera)
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
        write_mask(sample_dir / "target_mask.png", prepared.roi.target_mask)
        write_mask(sample_dir / "ignore_mask.png", prepared.roi.ignore_mask)
        write_mask(sample_dir / "valid_mask.png", prepared.roi.valid_mask)

    print(f"图片路径: {image_path}")
    print(f"输出目录: {sample_dir}")
    print(f"拒绝原因: {prepared.rejection_reason}")


if __name__ == "__main__":
    main()
