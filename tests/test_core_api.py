from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from seat_defect_core.api import frames_from_paths


def test_frames_from_paths_reads_images_and_reports_failures(tmp_path: Path) -> None:
    image_path = tmp_path / "cam_0.png"
    cv2.imwrite(str(image_path), np.full((8, 8, 3), 127, dtype=np.uint8))

    frames = frames_from_paths(
        {
            "cam_0": image_path,
            "cam_1": tmp_path / "missing.png",
        },
        frame_id="frame_001",
        timestamp="2026-05-09T00:00:00+08:00",
    )

    assert frames[0].camera_id == "cam_0"
    assert frames[0].image is not None
    assert frames[0].source == str(image_path)
    assert frames[0].source_kind == "image_path"
    assert frames[0].error_reason is None
    assert frames[1].camera_id == "cam_1"
    assert frames[1].image is None
    assert frames[1].error_reason == "image_read_failed"
