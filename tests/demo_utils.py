from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from seat_defect_inspection.patchcore import list_images
from seat_defect_inspection.runtime_config import load_config


def load_camera(
    *,
    config_path: str,
    camera_id: str,
    seat_model_id: str | None = None,
    yolo_model_path: str | None = None,
    patchcore_model_path: str | None = None,
    device: str | None = None,
):
    # 从项目配置里取一个机位，后续直接复用正式链路。
    config = load_config(config_path)

    if config.seat_models:
        resolved_seat_model_id = (
            seat_model_id
            or config.default_seat_model_id
            or config.seat_models[0].seat_model_id
        )
        seat_model = next(
            item for item in config.seat_models if item.seat_model_id == resolved_seat_model_id
        )
        cameras = [camera for camera in seat_model.cameras if camera.enabled]
    else:
        cameras = [camera for camera in config.cameras if camera.enabled]

    camera = next(camera for camera in cameras if camera.camera_id == camera_id)
    if yolo_model_path is not None or device is not None:
        camera = replace(
            camera,
            detection=replace(
                camera.detection,
                model_path=str(Path(yolo_model_path).resolve()) if yolo_model_path else camera.detection.model_path,
                device=device or camera.detection.device,
            ),
        )
    if patchcore_model_path is not None:
        camera = replace(
            camera,
            patchcore_model_path=str(Path(patchcore_model_path).resolve()),
        )
    return camera


def resolve_image_path(*, image_path: str | None, image_dir: str) -> Path:
    # 不指定单张图片时，默认取目录里的第一张。
    if image_path:
        return ensure_raw_input_path(Path(image_path))

    image_paths = list_images(Path(image_dir))
    if not image_paths:
        raise FileNotFoundError(f"没有找到图片：{image_dir}")
    return ensure_raw_input_path(image_paths[0])


def ensure_raw_input_path(image_path: Path | str) -> Path:
    # demo 只能接原始输入图，不能直接吃 yolo_debug 这类已经叠加可视化的产物。
    path = Path(image_path)
    if "yolo_debug" in path.as_posix():
        raise ValueError(
            f"禁止把 YOLO 调试可视化结果当作链路输入：{path}。"
            " 请改用相机原图或数据集原图。"
        )
    return path


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"写入图片失败：{path}")


def write_mask(path: Path, mask: np.ndarray) -> None:
    normalized = mask
    if normalized.dtype != np.uint8:
        normalized = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    else:
        normalized = np.where(normalized > 0, 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(path), normalized):
        raise OSError(f"写入掩膜失败：{path}")
