from __future__ import annotations

import json
from pathlib import Path

from seat_defect_core.reporting import export_inspection_report
from seat_defect_core.types import BoundingBox, CameraInspectionResult, DetectionObject, DetectionResult, InspectionError, InspectionResponse, InspectionResult, TextureAnomalyResult


def test_export_inspection_report_writes_latest_only(tmp_path: Path) -> None:
    result = InspectionResult(
        part_id="seat_001",
        frame_id="20260422_101010_000001",
        timestamp="2026-04-22T10:10:10+08:00",
        status="OK",
        decision_reason="all_cameras_ok",
        seat_model_id="seat_model_a",
        camera_results=[],
    )

    latest_path = export_inspection_report(result, str(tmp_path / "results.json"))

    assert latest_path == tmp_path / "results.json"
    assert latest_path.exists()
    assert not (tmp_path / "results_history").exists()


def test_export_inspection_report_writes_target_box_and_crop_box(tmp_path: Path) -> None:
    result = InspectionResult(
        part_id="seat_001",
        frame_id="frame_001",
        timestamp="2026-04-22T10:10:10+08:00",
        status="NG",
        decision_reason="ng_from_cam_0",
        seat_model_id=None,
        camera_results=[
            CameraInspectionResult(
                camera_id="cam_0",
                frame_id="frame_001",
                source="cam0.png",
                source_kind="image",
                status="NG",
                reason="texture_anomaly",
                timings_ms={"total": 12.5},
                error=InspectionError(
                    code="pipeline_failed",
                    message="failed",
                    stage="camera_pipeline",
                ),
                detection=DetectionResult(
                    target=DetectionObject(
                        label="seat",
                        confidence=0.98,
                        bounding_box=BoundingBox(10.0, 20.0, 30.0, 40.0),
                    )
                ),
                crop_box=BoundingBox(8.0, 18.0, 32.0, 42.0),
            )
        ],
    )

    latest_path = export_inspection_report(result, str(tmp_path / "results.json"))
    payload = json.loads(latest_path.read_text(encoding="utf-8"))

    camera_payload = payload["camera_results"][0]
    assert camera_payload["target_box"] == {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0}
    assert camera_payload["crop_box"] == {"x1": 8.0, "y1": 18.0, "x2": 32.0, "y2": 42.0}
    assert camera_payload["timings_ms"]["total"] == 12.5
    assert camera_payload["error"]["code"] == "pipeline_failed"


def test_inspection_response_to_dict_uses_full_result_serializer() -> None:
    response = InspectionResponse(
        result=InspectionResult(
            part_id="seat_001",
            frame_id="frame_001",
            timestamp="2026-04-22T10:10:10+08:00",
            status="NG",
            decision_reason="ng_from_cam_0",
            camera_results=[
                CameraInspectionResult(
                    camera_id="cam_0",
                    frame_id="frame_001",
                    source="cam0.png",
                    source_kind="image",
                    status="NG",
                    reason="texture_anomaly",
                    detection=DetectionResult(
                        target=DetectionObject(
                            label="seat",
                            confidence=0.98,
                            bounding_box=BoundingBox(10.0, 20.0, 30.0, 40.0),
                        )
                    ),
                    crop_box=BoundingBox(8.0, 18.0, 32.0, 42.0),
                    texture_result=TextureAnomalyResult(
                        score=2.0,
                        threshold=1.0,
                        is_anomaly=True,
                        heatmap=None,
                        valid_patch_ratio=1.0,
                        valid_patch_count=4,
                        total_patch_count=4,
                        decision_mode="normal_rule",
                    ),
                )
            ],
        ),
        report_path="results.json",
        artifact_paths={},
    )

    payload = response.to_dict()
    camera_payload = payload["camera_results"][0]

    assert camera_payload["target_box"] == {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0}
    assert camera_payload["crop_box"] == {"x1": 8.0, "y1": 18.0, "x2": 32.0, "y2": 42.0}
    assert camera_payload["texture_result"]["decision_mode"] == "normal_rule"
    assert payload["report_path"] == "results.json"
