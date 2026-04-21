from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from demo_utils import load_camera, resolve_image_path, write_image, write_mask
from seat_defect_inspection.patchcore import PatchCoreService
from seat_defect_inspection.service import _CameraPipeline, _overlay_heatmap, _render_detections

# 这个脚本只做：
# 1. 选一张图片
# 2. 走当前项目里的 YOLO + OpenCV + ROI 链路
# 3. 把处理后的结果交给训练好的 PatchCore 模型

CONFIG_PATH = "configs/seat_defect_inspection.mvs.json"
CAMERA_ID = "cam_0"
SEAT_MODEL_ID: str | None = None
YOLO_MODEL_PATH: str | None = "best.pt"
PATCHCORE_MODEL_PATH: str | None = None
DEVICE: str | None = "cpu"
IMAGE_DIR = "datasets/seat_defect/images/val"
IMAGE_PATH: str | None = None
OUTPUT_DIR = "outputs/patchcore_single_image_inference"


def main() -> None:
    camera = load_camera(
        config_path=CONFIG_PATH,
        camera_id=CAMERA_ID,
        seat_model_id=SEAT_MODEL_ID,
        yolo_model_path=YOLO_MODEL_PATH,
        patchcore_model_path=PATCHCORE_MODEL_PATH,
        device=DEVICE,
    )
    image_path = resolve_image_path(image_path=IMAGE_PATH, image_dir=IMAGE_DIR)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"读取图片失败：{image_path}")

    pipeline = _CameraPipeline(camera)
    prepared = pipeline.prepare_image(image)
    if prepared.rejection_reason is not None or prepared.roi is None:
        raise RuntimeError(f"当前图片在项目链路中被拒绝：{prepared.rejection_reason}")

    patchcore_input = (
        prepared.roi.texture_ready_image
        if prepared.roi.texture_ready_image is not None
        else prepared.roi.aligned_roi_image
    )
    result = PatchCoreService.load_bundle(camera.patchcore_model_path).patchcore.predict(
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
        "score": float(result.score),
        "threshold": float(result.threshold),
        "valid_patch_ratio": float(result.valid_patch_ratio),
        "is_anomaly": bool(result.is_anomaly),
    }
    (sample_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"图片路径: {image_path}")
    print(f"输出目录: {sample_dir}")
    print(f"PatchCore 分数: {result.score:.6f}")
    print(f"PatchCore 阈值: {result.threshold:.6f}")
    print(f"是否异常: {result.is_anomaly}")


if __name__ == "__main__":
    main()
