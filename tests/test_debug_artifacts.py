from __future__ import annotations

import numpy as np

from seat_defect_inspection.cvops import resolve_debug_artifact_names
from seat_defect_inspection.cvops.debug_artifacts import _overlay_heatmap


def test_standard_artifact_mode_keeps_only_core_outputs() -> None:
    assert resolve_debug_artifact_names("standard") == {
        "raw",
        "detections",
        "roi",
        "patchcore_input",
        "overlay",
    }


def test_full_artifact_mode_keeps_all_debug_outputs() -> None:
    assert resolve_debug_artifact_names("full") == {
        "raw",
        "preprocessed",
        "detections",
        "roi",
        "roi_texture",
        "patchcore_input",
        "foreground_weight",
        "target_mask",
        "valid_mask",
        "ignore_mask",
        "heatmap",
        "overlay",
    }


def test_artifact_mode_aliases_are_supported() -> None:
    assert resolve_debug_artifact_names("core") == resolve_debug_artifact_names("standard")
    assert resolve_debug_artifact_names("all") == resolve_debug_artifact_names("full")


def test_unknown_artifact_mode_raises() -> None:
    try:
        resolve_debug_artifact_names("unexpected")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unexpected debug_artifact_mode")


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
