"""Debug artifact saving and visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

import cv2
import numpy as np

from ..util import build_model_scoped_root, select_patchcore_input, write_image

DEFAULT_DEBUG_ARTIFACT_NAMES: FrozenSet[str] = frozenset(
    {
        "overlay",
    }
)


def generate_overlay_image(
    frame_packet: Any,
    prepared: Any,
    texture_result: Optional[Any] = None,
    region_results: Optional[Any] = None,
) -> Optional[np.ndarray]:
    """Generate the BGR overlay image for one camera result.

    Returns None when no heatmap data is available (e.g. REJECT/error paths).
    """
    if prepared.roi is None:
        return None
    if texture_result is None and not region_results:
        return None
    heatmap = (
        texture_result.heatmap
        if texture_result is not None
        else _stitch_region_heatmap(prepared.roi, region_results)
    )
    return _overlay_heatmap_on_frame(frame_packet.image, prepared.roi, heatmap)


def save_debug_artifacts(
    *,
    debug_dir: str,
    artifact_names: Union[List[str], Tuple[str, ...], FrozenSet[str], None] = None,
    frame_packet: Any,
    prepared: Any,
    texture_result: Optional[Any],
    seat_model_id: Optional[str],
    region_results: Optional[Any] = None,
) -> Dict[str, str]:
    """Persist the selected debug artifacts for one camera result."""
    selected_artifacts = _normalize_artifact_names(artifact_names)
    if not selected_artifacts:
        return {}

    camera_dir = (
        build_model_scoped_root(Path(debug_dir), seat_model_id)
        / frame_packet.part_id
        / frame_packet.camera_id
        / frame_packet.frame_id
    )
    camera_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: Dict[str, str] = {}

    if prepared.roi is not None and (
        texture_result is not None or region_results
    ):
        overlay = generate_overlay_image(
            frame_packet,
            prepared,
            texture_result=texture_result,
            region_results=region_results,
        )
        if overlay is not None and "overlay" in selected_artifacts:
            _save_artifact_image(
                artifact_paths,
                "overlay",
                camera_dir / "overlay.png",
                overlay,
            )

    return artifact_paths


def _normalize_artifact_names(
    artifact_names: Union[List[str], Tuple[str, ...], FrozenSet[str], None],
) -> FrozenSet[str]:
    if artifact_names is None:
        return DEFAULT_DEBUG_ARTIFACT_NAMES
    requested = [str(name).strip() for name in artifact_names if str(name).strip()]
    unexpected = sorted(set(requested) - DEFAULT_DEBUG_ARTIFACT_NAMES)
    if unexpected:
        formatted = ", ".join(f"`{item}`" for item in unexpected)
        raise ValueError(f"debug_artifact_names 包含不支持的调试产物: {formatted}")
    return frozenset(requested)


def _stitch_region_heatmap(roi, region_results) -> np.ndarray:
    """Merge region-level PatchCore heatmaps back into full ROI coordinates."""
    height, width = select_patchcore_input(roi).shape[:2]
    stitched = np.zeros((height, width), dtype=np.float32)
    for region_result in region_results or []:
        texture_result = getattr(region_result, "texture_result", None)
        if texture_result is None:
            continue
        x1, y1, x2, y2 = _region_box_to_pixels(region_result.box, width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        region_heatmap = np.clip(
            np.asarray(texture_result.heatmap, dtype=np.float32),
            0.0,
            1.0,
        )
        if region_heatmap.shape != (y2 - y1, x2 - x1):
            region_heatmap = cv2.resize(
                region_heatmap,
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_LINEAR,
            )
        stitched[y1:y2, x1:x2] = np.maximum(
            stitched[y1:y2, x1:x2],
            region_heatmap,
        )
    return stitched


def _region_box_to_pixels(box, width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = int(round(float(box.x1)))
    y1 = int(round(float(box.y1)))
    x2 = int(round(float(box.x2)))
    y2 = int(round(float(box.y2)))
    return (
        min(max(x1, 0), width),
        min(max(y1, 0), height),
        min(max(x2, 0), width),
        min(max(y2, 0), height),
    )


def _save_artifact_image(
    artifact_paths: Dict[str, str],
    key: str,
    path: Path,
    image: Optional[Any],
) -> None:
    """Write one final per-camera artifact."""
    if image is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_image(path, image)
    artifact_paths[key] = str(path)


def _overlay_heatmap_on_frame(
    frame_image: Any,
    roi,
    heatmap: np.ndarray,
) -> np.ndarray:
    """Overlay a canonical ROI heatmap back onto the original frame size."""
    frame_base = _ensure_color_image(frame_image)
    x1, y1, x2, y2 = _box_to_frame_pixels(roi.crop_box, frame_base.shape[:2])
    if x2 <= x1 or y2 <= y1:
        return frame_base

    crop_base = frame_base[y1:y2, x1:x2]
    crop_heatmap = _restore_heatmap_to_crop(roi, heatmap, crop_base.shape[:2])
    frame_base[y1:y2, x1:x2] = _overlay_heatmap(crop_base, crop_heatmap)
    return frame_base


def _restore_heatmap_to_crop(
    roi,
    heatmap: np.ndarray,
    crop_shape: Tuple[int, int],
) -> np.ndarray:
    canonical_shape = select_patchcore_input(roi).shape[:2]
    clipped = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    if clipped.shape != canonical_shape:
        clipped = cv2.resize(
            clipped,
            (canonical_shape[1], canonical_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    crop_height, crop_width = crop_shape
    if crop_height <= 0 or crop_width <= 0:
        return np.zeros((0, 0), dtype=np.float32)

    canonical_height, canonical_width = canonical_shape
    scale = min(
        float(canonical_width) / float(crop_width),
        float(canonical_height) / float(crop_height),
    )
    content_width = max(1, int(round(crop_width * scale)))
    content_height = max(1, int(round(crop_height * scale)))
    offset_x = max(0, (canonical_width - content_width) // 2)
    offset_y = max(0, (canonical_height - content_height) // 2)

    content = clipped[
        offset_y : min(canonical_height, offset_y + content_height),
        offset_x : min(canonical_width, offset_x + content_width),
    ]
    if content.size == 0:
        return np.zeros((crop_height, crop_width), dtype=np.float32)
    return cv2.resize(
        content,
        (crop_width, crop_height),
        interpolation=cv2.INTER_LINEAR,
    )


def _box_to_frame_pixels(box, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    height, width = frame_shape
    x1 = int(round(float(box.x1)))
    y1 = int(round(float(box.y1)))
    x2 = int(round(float(box.x2)))
    y2 = int(round(float(box.y2)))
    return (
        min(max(x1, 0), width),
        min(max(y1, 0), height),
        min(max(x2, 0), width),
        min(max(y2, 0), height),
    )


def _overlay_heatmap(
    image: Any,
    heatmap: np.ndarray,
) -> np.ndarray:
    """Overlay the heatmap on the ROI while preserving cool areas."""
    base_image, clipped = _prepare_heatmap_layers(image, heatmap)
    if float(clipped.max()) <= 1e-6:
        return base_image

    color_map = cv2.applyColorMap(np.uint8(clipped * 255), cv2.COLORMAP_JET).astype(np.float32)
    base_float = base_image.astype(np.float32)
    alpha = np.power(clipped, 1.35)[..., None] * 0.75
    overlay = base_float * (1.0 - alpha) + color_map * alpha
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def _prepare_heatmap_layers(
    image: Any,
    heatmap: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize heatmap geometry and value range for visualization helpers."""
    base_image = _ensure_color_image(image)
    clipped = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    if clipped.shape != base_image.shape[:2]:
        clipped = cv2.resize(
            clipped,
            (base_image.shape[1], base_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return base_image, clipped


def _ensure_color_image(image: Any) -> np.ndarray:
    """Ensure the heatmap base is a BGR image."""
    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim == 3 and array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    return array.copy()
