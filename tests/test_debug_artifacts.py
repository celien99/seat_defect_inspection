from __future__ import annotations

import numpy as np

from seat_defect_core.debug_artifacts import get_debug_artifact_names
from seat_defect_core.cvops.debug_artifacts import _overlay_heatmap, _render_heatmap


def test_unified_artifact_set_contains_expected_outputs() -> None:
    assert get_debug_artifact_names() == {
        "raw",
        "detections",
        "heatmap",
        "overlay",
    }


def test_render_heatmap_uses_grayscale_base_when_heat_is_zero() -> None:
    base_image = np.full((32, 32, 3), (10, 70, 150), dtype=np.uint8)
    heatmap = np.zeros((32, 32), dtype=np.float32)

    rendered = _render_heatmap(base_image, heatmap)

    assert rendered.shape == base_image.shape
    assert np.array_equal(rendered[:, :, 0], rendered[:, :, 1])
    assert np.array_equal(rendered[:, :, 1], rendered[:, :, 2])


def test_render_heatmap_emphasizes_hotspot_on_roi_structure() -> None:
    base_image = np.full((32, 32, 3), 64, dtype=np.uint8)
    heatmap = np.zeros((32, 32), dtype=np.float32)
    heatmap[12:20, 12:20] = 1.0

    rendered = _render_heatmap(base_image, heatmap)

    assert np.array_equal(rendered[2, 2][0], rendered[2, 2][1])
    assert np.array_equal(rendered[2, 2][1], rendered[2, 2][2])
    assert not np.array_equal(rendered[16, 16], rendered[2, 2])
    assert not (
        rendered[16, 16][0] == rendered[16, 16][1] == rendered[16, 16][2]
    )


def test_overlay_heatmap_keeps_base_image_when_heat_is_zero() -> None:
    base_image = np.full((32, 32, 3), 42, dtype=np.uint8)
    heatmap = np.zeros((32, 32), dtype=np.float32)

    overlay = _overlay_heatmap(base_image, heatmap)

    assert overlay.shape == base_image.shape
    assert np.array_equal(overlay, base_image)


def test_overlay_heatmap_emphasizes_hotspot_without_tinting_everything() -> None:
    base_image = np.full((32, 32, 3), 64, dtype=np.uint8)
    heatmap = np.zeros((32, 32), dtype=np.float32)
    heatmap[12:20, 12:20] = 1.0

    overlay = _overlay_heatmap(base_image, heatmap)

    assert not np.array_equal(overlay[16, 16], base_image[16, 16])
    assert np.array_equal(overlay[2, 2], base_image[2, 2])
