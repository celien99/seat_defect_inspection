from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from seat_defect_inspection.service import benchmark as benchmark_module


class _FakeCamera:
    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self.enabled = True
        self.source = f"source://{camera_id}"


class _FakeContext:
    def __init__(self, cameras) -> None:
        self.cameras = cameras


class _FakeService:
    def __init__(self, cameras) -> None:
        self._context = _FakeContext(cameras)

    def resolve_context(self, seat_model_id):
        return self._context


def test_benchmark_exports_flat_overlay_artifact(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    benchmark_data = tmp_path / "benchmark_data" / "good" / "cam_0"
    benchmark_data.mkdir(parents=True)

    image_path = benchmark_data / "sample_001.png"
    source_image = np.zeros((10, 10, 3), dtype=np.uint8)
    source_image[:, :] = (20, 40, 60)
    cv2.imwrite(str(image_path), source_image)

    cameras = [_FakeCamera("cam_0")]
    service = _FakeService(cameras)

    def fake_inspect_frames(_service, frames, *, part_id=None):
        heatmap = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
        texture_result = SimpleNamespace(heatmap=heatmap, score=1.0, threshold=0.5, is_anomaly=True)
        cam_result = SimpleNamespace(
            camera_id="cam_0",
            status="NG",
            source=frames[0].source,
            crop_box=SimpleNamespace(x1=1.0, y1=1.0, x2=9.0, y2=9.0),
            texture_result=texture_result,
            region_results=[],
            overlay_image=np.full((10, 10, 3), 180, dtype=np.uint8),
        )
        return SimpleNamespace(
            status="NG",
            decision_reason="ng_from_cam_0",
            camera_results=[cam_result],
        )

    monkeypatch.setattr(benchmark_module, "BENCHMARK_DATA_DIR", tmp_path / "benchmark_data")
    monkeypatch.setattr(benchmark_module, "inspect_frames", fake_inspect_frames)

    results = benchmark_module.run_benchmark(
        service,
        rounds=("good",),
        camera_ids=["cam_0"],
        artifacts_dir=tmp_path / "benchmark_artifacts",
    )

    captured = capsys.readouterr().out
    assert str(image_path) not in captured
    assert "reason=ng_from_cam_0" in captured

    overlay_path = (
        tmp_path / "benchmark_artifacts" / "good_good_0000_cam_0_overlay.png"
    )
    assert overlay_path.is_file()
    assert not (tmp_path / "benchmark_artifacts" / "good").exists()

    overlay_image = cv2.imread(str(overlay_path))
    assert overlay_image is not None and overlay_image.shape[:2] == (10, 10)

    camera_artifacts = results["good"]["records"][0]["camera_results"][0]["artifact_paths"]
    assert set(camera_artifacts) == {"overlay"}
    assert camera_artifacts["overlay"] == str(overlay_path)
