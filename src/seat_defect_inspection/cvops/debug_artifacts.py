"""Debug artifact saving and visualization helpers."""

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
    """Persist the selected debug artifacts for one camera result."""
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
            "valid_mask",
            camera_dir / "valid_mask.png",
            prepared.roi.valid_mask,
        )
        _save_selected_mask(
            artifact_paths,
            selected_artifacts,
            "ignore_mask",
            camera_dir / "ignore_mask.png",
            prepared.roi.ignore_mask,
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
    """Write an image artifact when the current mode enables it."""
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
    """Write a mask artifact when the current mode enables it."""
    if mask is None or key not in selected_artifacts:
        return
    _write_mask(path, mask)
    artifact_paths[key] = str(path)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    """Normalize a mask-like array to uint8 before saving."""
    if mask.dtype != np.uint8:
        normalized = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    else:
        normalized = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(path), normalized):
        raise OSError(f"Failed to write mask: {path}")


def _render_detections(image: Any, detection) -> Any:
    """Render YOLO detections for quick inspection."""
    if detection is None:
        return image.copy()
    canvas = image.copy()
    drawn_ids: set[int] = set()
    if detection.target is not None:
        _draw_detection(canvas, detection.target, (0, 255, 0))
        drawn_ids.add(id(detection.target))
    for item in detection.all_objects:
        if id(item) in drawn_ids:
            continue
        _draw_detection(canvas, item, (0, 165, 255))
    return canvas


def _draw_detection(image: Any, detection, color: tuple[int, int, int]) -> None:
    """Draw one detection box and optional segmentation mask."""
    if getattr(detection, "segmentation_mask", None) is not None:
        _draw_segmentation_mask(image, detection.segmentation_mask, color)
    _draw_box(image, detection.bounding_box, color, detection.label)


def _draw_box(image: Any, box, color: tuple[int, int, int], label: str) -> None:
    """Draw one bounding box."""
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
    """Draw segmentation fill and contours."""
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
) -> Any:
    """Overlay the heatmap without tinting the whole ROI background."""
    base_image = _ensure_color_image(image)
    clipped = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    if clipped.shape != base_image.shape[:2]:
        clipped = cv2.resize(
            clipped,
            (base_image.shape[1], base_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    if float(clipped.max()) <= 1e-6:
        return base_image

    color_map = cv2.applyColorMap(np.uint8(clipped * 255), cv2.COLORMAP_JET).astype(np.float32)
    base_float = base_image.astype(np.float32)
    alpha = np.power(clipped, 1.35)[..., None] * 0.75
    overlay = base_float * (1.0 - alpha) + color_map * alpha
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def _ensure_color_image(image: Any) -> np.ndarray:
    """Ensure the heatmap overlay base is a BGR image."""
    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    return array.copy()
