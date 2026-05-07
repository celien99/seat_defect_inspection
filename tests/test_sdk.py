from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from seat_defect_core.config import CameraConfig, FusionConfig, InspectionConfig
from seat_defect_core.schemas import CameraInspectionResult
from seat_defect_sdk import CameraFrame, SeatDefectInspector, inspect_once


def _config(tmp_path: Path) -> InspectionConfig:
    return InspectionConfig(
        cameras=[
            CameraConfig(camera_id="cam_0", source="mvs://cam_0", patchcore_model_path="model_0.npz"),
            CameraConfig(camera_id="cam_1", source="mvs://cam_1", patchcore_model_path="model_1.npz"),
        ],
        output_json_path=str(tmp_path / "results.json"),
        debug_dir=str(tmp_path / "debug"),
        part_id="seat_demo",
        fusion=FusionConfig(),
    )


def test_sdk_inspect_uses_external_frames_without_capture(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    service_instances: list[object] = []
    inspected: list[tuple[str, object]] = []

    class _ForbiddenAcquisition:
        def capture(self, *_args, **_kwargs):
            raise AssertionError("SDK inspect must not call capture")

    class _FakeService:
        def __init__(self, config_arg):
            self.config = config_arg
            self.acquisition = _ForbiddenAcquisition()
            service_instances.append(self)

        def resolve_context(self, _seat_model_id):
            return SimpleNamespace(
                seat_model_id=None,
                cameras=config.cameras,
                pipelines={camera.camera_id: object() for camera in config.cameras},
            )

    def fake_inspect_one_camera(service, frame_packet, camera, _pipeline, seat_model_id):
        inspected.append((camera.camera_id, frame_packet.image))
        return CameraInspectionResult(
            camera_id=camera.camera_id,
            frame_id=frame_packet.frame_id,
            source=frame_packet.source,
            source_kind=frame_packet.source_kind,
            status="OK",
            reason="all_checks_passed",
            seat_model_id=seat_model_id,
            artifact_paths={
                "raw": f"{service.config.debug_dir}/{camera.camera_id}/raw.png",
                "overlay": f"{service.config.debug_dir}/{camera.camera_id}/overlay.png",
            },
        )

    monkeypatch.setattr("seat_defect_sdk.client._create_service", lambda config_arg: _FakeService(config_arg))
    monkeypatch.setattr("seat_defect_sdk.client._inspect_external_camera", fake_inspect_one_camera)

    image_0 = np.zeros((8, 8, 3), dtype=np.uint8)
    image_1 = np.ones((8, 8, 3), dtype=np.uint8)
    response = inspect_once(
        config,
        [
            {
                "camera_id": "cam_0",
                "image": image_0,
                "source": "memory://cam_0",
                "frame_id": "frame_001",
            },
            CameraFrame(
                camera_id="cam_1",
                image=image_1,
                source="memory://cam_1",
                frame_id="frame_001",
            ),
        ],
        part_id="seat_001",
    )

    assert len(service_instances) == 1
    assert inspected[0][0] == "cam_0"
    assert inspected[0][1] is image_0
    assert inspected[1][0] == "cam_1"
    assert inspected[1][1] is image_1
    assert response.status == "OK"
    assert response.report_path == str(tmp_path / "results.json")
    assert response.archive_report_path.endswith("results_history/default/seat_001/frame_001.json")
    assert response.artifact_paths["cam_0"]["overlay"].endswith("cam_0/overlay.png")
    assert response.to_dict()["artifact_paths"]["cam_1"]["raw"].endswith("cam_1/raw.png")


def test_sdk_reuses_service_and_reports_missing_frame(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    service_instances: list[object] = []

    class _FakeService:
        def __init__(self, config_arg):
            self.config = config_arg
            service_instances.append(self)

        def resolve_context(self, _seat_model_id):
            return SimpleNamespace(
                seat_model_id=None,
                cameras=config.cameras,
                pipelines={camera.camera_id: object() for camera in config.cameras},
            )

    monkeypatch.setattr("seat_defect_sdk.client._create_service", lambda config_arg: _FakeService(config_arg))
    monkeypatch.setattr(
        "seat_defect_sdk.client._inspect_external_camera",
        lambda *_args, **_kwargs: CameraInspectionResult(
            camera_id="cam_0",
            frame_id="frame_001",
            source="memory://cam_0",
            source_kind="external_image",
            status="OK",
            reason="all_checks_passed",
        ),
    )

    inspector = SeatDefectInspector(config)
    response = inspector.inspect(
        [CameraFrame(camera_id="cam_0", image=np.zeros((8, 8, 3), dtype=np.uint8), frame_id="frame_001")],
        part_id="seat_001",
    )
    inspector.inspect(
        [CameraFrame(camera_id="cam_0", image=np.zeros((8, 8, 3), dtype=np.uint8), frame_id="frame_002")],
        part_id="seat_002",
    )

    assert len(service_instances) == 1
    assert response.status == "REJECT"
    missing = response.result.camera_results[1]
    assert missing.camera_id == "cam_1"
    assert missing.reason == "missing_external_frame"


def test_sdk_rejects_duplicate_camera_frames(tmp_path: Path) -> None:
    config = _config(tmp_path)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    try:
        inspect_once(
            config,
            [
                CameraFrame(camera_id="cam_0", image=image),
                CameraFrame(camera_id="cam_0", image=image),
            ]
        )
    except ValueError as exc:
        assert "重复 camera_id" in str(exc)
        return
    raise AssertionError("expected duplicate camera_id rejection")


def test_sdk_rejects_unknown_camera_frame(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)

    class _FakeService:
        def __init__(self, config_arg):
            self.config = config_arg

        def resolve_context(self, _seat_model_id):
            return SimpleNamespace(
                seat_model_id=None,
                cameras=config.cameras,
                pipelines={camera.camera_id: object() for camera in config.cameras},
            )

    monkeypatch.setattr("seat_defect_sdk.client._create_service", lambda config_arg: _FakeService(config_arg))

    try:
        inspect_once(
            config,
            [CameraFrame(camera_id="unknown_cam", image=np.zeros((8, 8, 3), dtype=np.uint8))],
        )
    except ValueError as exc:
        assert "未配置或未启用" in str(exc)
        assert "unknown_cam" in str(exc)
        return
    raise AssertionError("expected unknown camera_id rejection")
