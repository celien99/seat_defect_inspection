from __future__ import annotations

from pathlib import Path

import cv2

from demo_utils import write_image
from seat_defect_inspection.preprocess import PreprocessEngine

# 这个脚本只做当前项目流程里的第一步：
# 原图 -> OpenCV 预处理 -> 保存结果
# 也就是进入 YOLO 之前的那一步。
OUTPUT_DIR = "outputs/demo"
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
    "sharpen": False
}


def main() -> None:
    image_path = Path("datasets/seat_defect/images/val/1.png")
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"读取图片失败：{image_path}")
 
    preprocess_engine = PreprocessEngine(PREPROCESS)
    preprocessed = preprocess_engine.process(image)
    

    sample_dir = Path(OUTPUT_DIR) / "preprocessed"
    sample_dir.mkdir(parents=True, exist_ok=True)

    write_image(sample_dir / "raw.png", image)
    write_image(sample_dir / "preprocessed.png", preprocessed)

    print(f"图片路径: {image_path}")
    # print(f"输出目录: {sample_dir}")
    print("说明: raw.png 是原图,preprocessed.png 是进入 YOLO 之前的 OpenCV 预处理结果。")


if __name__ == "__main__":
    main()
