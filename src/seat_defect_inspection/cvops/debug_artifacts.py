"""调试图输出与可视化辅助。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..debug_artifacts import DEFAULT_DEBUG_ARTIFACT_MODE, resolve_debug_artifact_names
from ..util import build_model_scoped_root, select_patchcore_input, write_image

def save_debug_artifacts(
    *,
    enabled: bool,
    debug_dir: str,
    debug_artifact_mode: str | None,
    frame_packet: Any,
    prepared: Any,
    texture_result: Any | None,
    seat_model_id: str | None,
) -> dict[str, str]:
    """按配置把当前机位调试产物写入磁盘并返回路径字典。"""
    if not enabled:
        return {}
    selected_artifacts = resolve_debug_artifact_names(debug_artifact_mode)

    camera_dir = (
        build_model_scoped_root(Path(debug_dir), seat_model_id)
        / frame_packet.part_id
        / frame_packet.camera_id
        / frame_packet.frame_id
    )
    camera_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, str] = {}

    _save_selected_image(
        artifact_paths,
        selected_artifacts,
        "raw",
        camera_dir / "raw.png",
        frame_packet.image,
    )

    if prepared.preprocessed_image is not None:
        _save_selected_image(
            artifact_paths,
            selected_artifacts,
            "preprocessed",
            camera_dir / "preprocessed.png",
            prepared.preprocessed_image,
        )
        if "detections" in selected_artifacts:
            _save_selected_image(
                artifact_paths,
                selected_artifacts,
                "detections",
                camera_dir / "detections.png",
                _render_detections(prepared.preprocessed_image, prepared.detection),
            )

    if prepared.roi is not None:
        _save_selected_image(
            artifact_paths,
            selected_artifacts,
            "roi",
            camera_dir / "roi.png",
            prepared.roi.aligned_roi_image,
        )
        if prepared.roi.texture_ready_image is not None:
            _save_selected_image(
                artifact_paths,
                selected_artifacts,
                "roi_texture",
                camera_dir / "roi_texture.png",
                prepared.roi.texture_ready_image,
            )
        patchcore_input = select_patchcore_input(prepared.roi)
        _save_selected_image(
            artifact_paths,
            selected_artifacts,
            "patchcore_input",
            camera_dir / "patchcore_input.png",
            patchcore_input,
        )
        if prepared.roi.foreground_weight is not None:
            _save_selected_mask(
                artifact_paths,
                selected_artifacts,
                "foreground_weight",
                camera_dir / "foreground_weight.png",
                prepared.roi.foreground_weight,
            )
        _save_selected_mask(
            artifact_paths,
            selected_artifacts,
            "target_mask",
            camera_dir / "target_mask.png",
            prepared.roi.target_mask,
        )
        _save_selected_mask(
            artifact_paths,
            selected_artifacts,
            "ignore_mask",
            camera_dir / "ignore_mask.png",
            prepared.roi.ignore_mask,
        )
        _save_selected_mask(
            artifact_paths,
            selected_artifacts,
            "valid_mask",
            camera_dir / "valid_mask.png",
            prepared.roi.valid_mask,
        )

    if texture_result is not None and prepared.roi is not None:
        _save_selected_mask(
            artifact_paths,
            selected_artifacts,
            "heatmap",
            camera_dir / "heatmap.png",
            texture_result.heatmap,
        )
        if "overlay" in selected_artifacts:
            _save_selected_image(
                artifact_paths,
                selected_artifacts,
                "overlay",
                camera_dir / "overlay.png",
                _overlay_heatmap(
                    select_patchcore_input(prepared.roi),
                    texture_result.heatmap,
                    texture_result=texture_result,
                ),
            )

    return artifact_paths


def _save_selected_image(
    artifact_paths: dict[str, str],
    selected_artifacts: set[str],
    key: str,
    path: Path,
    image: Any | None,
) -> None:
    """仅在当前档位启用时保存图像。"""
    if image is None or key not in selected_artifacts:
        return
    write_image(path, image)
    artifact_paths[key] = str(path)


def _save_selected_mask(
    artifact_paths: dict[str, str],
    selected_artifacts: set[str],
    key: str,
    path: Path,
    mask: np.ndarray | None,
) -> None:
    """仅在当前档位启用时保存掩膜。"""
    if mask is None or key not in selected_artifacts:
        return
    _write_mask(path, mask)
    artifact_paths[key] = str(path)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    """把浮点或二值掩膜统一归一化后写盘。"""
    if mask.dtype != np.uint8:
        normalized = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    else:
        normalized = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(path), normalized):
        raise OSError(f"Failed to write mask: {path}")


def _render_detections(image: Any, detection) -> Any:
    """渲染检测结果，便于快速核对 YOLO 命中区域。"""
    if detection is None:
        return image.copy()
    canvas = image.copy()
    if detection.target is not None:
        _draw_detection(canvas, detection.target, (0, 255, 0))
    for item in detection.ignores:
        _draw_detection(canvas, item, (0, 0, 255))
    return canvas


def _draw_detection(image: Any, detection, color: tuple[int, int, int]) -> None:
    """绘制检测框与可选分割轮廓。"""
    if getattr(detection, "segmentation_mask", None) is not None:
        _draw_segmentation_mask(image, detection.segmentation_mask, color)
    _draw_box(image, detection.bounding_box, color, detection.label)


def _draw_box(image: Any, box, color: tuple[int, int, int], label: str) -> None:
    """绘制检测框。"""
    x1 = int(round(box.x1))
    y1 = int(round(box.y1))
    x2 = int(round(box.x2))
    y2 = int(round(box.y2))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_segmentation_mask(image: Any, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    """绘制分割区域填充与轮廓。"""
    normalized = np.asarray(mask)
    if normalized.ndim != 2:
        return
    if normalized.shape[:2] != image.shape[:2]:
        normalized = cv2.resize(
            normalized.astype(np.float32),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    binary_mask = (normalized > 0).astype(np.uint8)
    if binary_mask.sum() == 0:
        return

    overlay = image.copy()
    overlay[binary_mask > 0] = (
        0.82 * overlay[binary_mask > 0] + 0.18 * np.asarray(color, dtype=np.float32)
    ).astype(np.uint8)
    image[:] = overlay

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)


def _overlay_heatmap(
    image: Any,
    heatmap: np.ndarray,
    *,
    texture_result: Any | None = None,
) -> Any:
    """把热力图叠加到 ROI 或纹理输入图上。"""
    base_image = _ensure_color_image(image)
    color_map = cv2.applyColorMap(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base_image, 0.65, color_map, 0.35, 0.0)
    if texture_result is not None:
        _draw_texture_metrics(overlay, texture_result)
    return overlay


def _ensure_color_image(image: Any) -> np.ndarray:
    """统一把单通道图转成 BGR，便于叠加彩色热力图与文字。"""
    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    return array.copy()


def _draw_texture_metrics(image: np.ndarray, texture_result: Any) -> None:
    """把关键判定数值直接写到调试叠图上，避免只看颜色误判。"""
    lines = [
        (
            f"score={float(getattr(texture_result, 'score', 0.0)):.3f} "
            f"thr={float(getattr(texture_result, 'threshold', 0.0)):.3f} "
            f"dec={float(getattr(texture_result, 'decision_threshold', 0.0)):.3f}"
        ),
        (
            f"peak={float(getattr(texture_result, 'peak_patch_score', 0.0)):.3f} "
            f"strong={int(getattr(texture_result, 'strong_patch_count', 0))} "
            f"comp={int(getattr(texture_result, 'largest_component_patch_count', 0))}"
        ),
        (
            f"mode={str(getattr(texture_result, 'decision_mode', 'none'))} "
            f"anom={bool(getattr(texture_result, 'is_anomaly', False))}"
        ),
    ]
    line_height = 24
    block_height = 10 + line_height * len(lines)
    cv2.rectangle(image, (8, 8), (min(image.shape[1] - 8, 540), 8 + block_height), (0, 0, 0), -1)

    for index, text in enumerate(lines):
        y = 30 + index * line_height
        cv2.putText(
            image,
            text,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
