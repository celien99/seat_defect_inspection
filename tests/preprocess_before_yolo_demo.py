from __future__ import annotations

from pathlib import Path

import cv2

from demo_utils import load_camera, resolve_image_path, write_image
from seat_defect_inspection.preprocess import PreprocessEngine

# 这个脚本只做当前项目流程里的第一步：
# 原图 -> OpenCV 预处理 -> 保存结果
# 也就是进入 YOLO 之前的那一步。

CONFIG_PATH = "configs/seat_defect_inspection.mvs.json"
CAMERA_ID = "cam_0"
SEAT_MODEL_ID: str | None = None
IMAGE_DIR = "datasets/seat_defect/images/val"
IMAGE_PATH: str | None = None
OUTPUT_DIR = "outputs/preprocess_before_yolo_demo"


def main() -> None:
    camera = load_camera(
        config_path=CONFIG_PATH,
        camera_id=CAMERA_ID,
        seat_model_id=SEAT_MODEL_ID,
    )
    image_path = resolve_image_path(image_path=IMAGE_PATH, image_dir=IMAGE_DIR)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"读取图片失败：{image_path}")

    preprocess_engine = PreprocessEngine(camera.preprocess)
    preprocessed = preprocess_engine.process(image)

    sample_dir = Path(OUTPUT_DIR) / image_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    write_image(sample_dir / "raw.png", image)
    write_image(sample_dir / "preprocessed.png", preprocessed)

    print(f"图片路径: {image_path}")
    print(f"输出目录: {sample_dir}")
    print("说明: raw.png 是原图，preprocessed.png 是进入 YOLO 之前的 OpenCV 预处理结果。")


if __name__ == "__main__":
    main()
