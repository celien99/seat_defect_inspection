from __future__ import annotations

import numpy as np
import pytest
import cv2

from seat_defect_core.cvops.debug_artifacts import (
    DEFAULT_DEBUG_ARTIFACT_NAMES,
    _overlay_heatmap,
    save_debug_artifacts,
)
from seat_defect_core.service.core import PreparedCameraSample
from seat_defect_core.types import (
    BoundingBox,
    FramePacket,
    RegionPatchCoreResult,
    RoiRefineResult,
    TextureAnomalyResult,
)


def test_unified_artifact_set_contains_expected_outputs() -> None:
    assert DEFAULT_DEBUG_ARTIFACT_NAMES == {"overlay"}


def test_save_debug_artifacts_rejects_removed_artifact_names(tmp_path) -> None:
    frame_packet = FramePacket(
        camera_id="cam_0",
        frame_id="frame_0",
        part_id="part_0",
        source="unit",
        source_kind="test",
        timestamp="2026-05-12T00:00:00+08:00",
        image=np.zeros((16, 16, 3), dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="`raw`"):
        save_debug_artifacts(
            debug_dir=str(tmp_path),
            artifact_names=["raw"],
            frame_packet=frame_packet,
            prepared=PreparedCameraSample(quality=None, roi=None),
            texture_result=None,
            seat_model_id=None,
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


def test_save_region_debug_artifacts_stitches_overlay_per_camera(tmp_path) -> None:
    target_mask = np.ones((10, 12), dtype=np.uint8)
    roi = RoiRefineResult(
        crop_box=BoundingBox(4.0, 5.0, 16.0, 15.0),
        roi_image=np.zeros((10, 12, 3), dtype=np.uint8),
        aligned_roi_image=np.zeros((10, 12, 3), dtype=np.uint8),
        texture_ready_image=np.zeros((10, 12, 3), dtype=np.uint8),
        target_mask=target_mask,
        valid_mask=target_mask,
        ignore_mask=np.zeros((10, 12), dtype=np.uint8),
        foreground_weight=None,
    )
    frame_packet = FramePacket(
        camera_id="cam_0",
        frame_id="frame_0",
        part_id="part_0",
        source="unit",
        source_kind="test",
        timestamp="2026-05-12T00:00:00+08:00",
        image=np.zeros((24, 32, 3), dtype=np.uint8),
    )
    region_results = [
        RegionPatchCoreResult(
            region_id="front",
            status="OK",
            reason="all_checks_passed",
            box=BoundingBox(0.0, 0.0, 4.0, 5.0),
            texture_result=TextureAnomalyResult(
                score=0.0,
                threshold=1.0,
                is_anomaly=False,
                heatmap=np.ones((5, 4), dtype=np.float32),
                valid_patch_ratio=1.0,
                valid_patch_count=1,
                total_patch_count=1,
            ),
        ),
        RegionPatchCoreResult(
            region_id="rear",
            status="NG",
            reason="texture_anomaly",
            box=BoundingBox(8.0, 5.0, 12.0, 10.0),
            texture_result=TextureAnomalyResult(
                score=2.0,
                threshold=1.0,
                is_anomaly=True,
                heatmap=np.full((5, 4), 0.8, dtype=np.float32),
                valid_patch_ratio=1.0,
                valid_patch_count=1,
                total_patch_count=1,
            ),
        ),
    ]

    paths = save_debug_artifacts(
        debug_dir=str(tmp_path),
        artifact_names=["overlay"],
        frame_packet=frame_packet,
        prepared=PreparedCameraSample(quality=None, roi=roi),
        texture_result=None,
        region_results=region_results,
        seat_model_id=None,
    )

    camera_dir = tmp_path / "part_0" / "cam_0" / "frame_0"
    assert set(paths) == {"overlay"}
    assert (camera_dir / "overlay.png").is_file()
    assert not (camera_dir / "regions").exists()
    overlay = cv2.imread(paths["overlay"])
    assert overlay is not None
    assert overlay.shape[:2] == frame_packet.image.shape[:2]


def test_save_debug_artifacts_can_emit_only_overlay(tmp_path) -> None:
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
        artifact_names=["overlay"],
        frame_packet=frame_packet,
        prepared=PreparedCameraSample(quality=None, roi=roi),
        texture_result=texture_result,
        seat_model_id=None,
    )

    assert set(paths) == {"overlay"}
    assert (tmp_path / "part_0" / "cam_0" / "frame_0" / "overlay.png").is_file()
    assert not (tmp_path / "part_0" / "cam_0" / "frame_0" / "heatmap.png").exists()
    assert not (tmp_path / "part_0" / "cam_0" / "frame_0" / "raw.png").exists()


def test_save_debug_artifacts_default_emits_overlay(tmp_path) -> None:
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
        artifact_names=None,
        frame_packet=frame_packet,
        prepared=PreparedCameraSample(quality=None, roi=roi),
        texture_result=texture_result,
        seat_model_id=None,
    )

    camera_dir = tmp_path / "part_0" / "cam_0" / "frame_0"
    assert set(paths) == {"overlay"}
    assert (camera_dir / "overlay.png").is_file()
    assert not (camera_dir / "heatmap.png").exists()
