from __future__ import annotations

import numpy as np

from seat_defect_core.cvops.debug_artifacts import (
    DEFAULT_DEBUG_ARTIFACT_NAMES,
    _overlay_heatmap,
    _render_heatmap,
    save_debug_artifacts,
)
from seat_defect_core.service.core import PreparedCameraSample
from seat_defect_core.types import BoundingBox, FramePacket, RoiRefineResult, TextureAnomalyResult


def test_unified_artifact_set_contains_expected_outputs() -> None:
    assert DEFAULT_DEBUG_ARTIFACT_NAMES == {
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


def test_save_debug_artifacts_can_emit_only_heatmap(tmp_path) -> None:
    target_mask = np.ones((16, 16), dtype=np.uint8)
    roi = RoiRefineResult(
        crop_box=BoundingBox(0.0, 0.0, 16.0, 16.0),
        roi_image=np.zeros((16, 16, 3), dtype=np.uint8),
        aligned_roi_image=np.zeros((16, 16, 3), dtype=np.uint8),
        texture_ready_image=np.zeros((16, 16, 3), dtype=np.uint8),
        target_mask=target_mask,
        valid_mask=target_mask,
        ignore_mask=np.zeros((16, 16), dtype=np.uint8),
        foreground_weight=None,
    )
    frame_packet = FramePacket(
        camera_id="cam_0",
        frame_id="frame_0",
        part_id="part_0",
        source="unit",
        source_kind="test",
        timestamp="2026-05-11T00:00:00+08:00",
        image=np.zeros((16, 16, 3), dtype=np.uint8),
    )
    texture_result = TextureAnomalyResult(
        score=0.0,
        threshold=1.0,
        is_anomaly=False,
        heatmap=np.ones((16, 16), dtype=np.float32),
        valid_patch_ratio=1.0,
        valid_patch_count=1,
        total_patch_count=1,
    )

    paths = save_debug_artifacts(
        debug_dir=str(tmp_path),
        artifact_names=["heatmap"],
        frame_packet=frame_packet,
        prepared=PreparedCameraSample(quality=None, roi=roi),
        texture_result=texture_result,
        seat_model_id=None,
    )

    assert set(paths) == {"heatmap"}
    assert (tmp_path / "part_0" / "cam_0" / "frame_0" / "heatmap.png").is_file()
    assert not (tmp_path / "part_0" / "cam_0" / "frame_0" / "raw.png").exists()
