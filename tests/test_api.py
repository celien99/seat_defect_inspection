from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from seat_defect_inspection.api import (
    SeatDefectInspector,
    inspect_folder_once,
    inspect_once,
)
from seat_defect_inspection.config import InspectionConfig
from seat_defect_inspection.schemas import InspectionResult


def _result(part_id: str | None, seat_model_id: str | None) -> InspectionResult:
    return InspectionResult(
        part_id=part_id or "seat_demo",
        frame_id="frame_001",
        timestamp="2026-05-07T12:00:00+08:00",
        status="OK",
        decision_reason="all_checks_passed",
        seat_model_id=seat_model_id,
    )


def test_inspect_once_accepts_config_path(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    loaded_config = InspectionConfig(part_id="loaded_part")
    observed: dict[str, object] = {}

    def fake_load_config(path: str) -> InspectionConfig:
        observed["config_path"] = path
        return loaded_config

    def fake_create_service(config: InspectionConfig):
        observed["service_config"] = config
        return SimpleNamespace(config=config)

    def fake_run_online_inspection(service, *, part_id, seat_model_id):
        observed["service"] = service
        observed["part_id"] = part_id
        observed["seat_model_id"] = seat_model_id
        return _result(part_id, seat_model_id)

    monkeypatch.setattr("seat_defect_inspection.api.load_config", fake_load_config)
    monkeypatch.setattr("seat_defect_inspection.api._create_service", fake_create_service)
    monkeypatch.setattr(
        "seat_defect_inspection.api._run_online_inspection",
        fake_run_online_inspection,
    )

    result = inspect_once(
        config_path,
        part_id="seat_001",
        seat_model_id="model_a",
    )

    assert observed["config_path"] == str(config_path)
    assert observed["service_config"] is loaded_config
    assert observed["part_id"] == "seat_001"
    assert observed["seat_model_id"] == "model_a"
    assert result.status == "OK"


def test_seat_defect_inspector_reuses_service_between_inspections(monkeypatch) -> None:
    config = InspectionConfig()
    created_services: list[object] = []

    def fake_create_service(config_arg: InspectionConfig):
        service = SimpleNamespace(config=config_arg, index=len(created_services))
        created_services.append(service)
        return service

    def fake_run_online_inspection(service, *, part_id, seat_model_id):
        return _result(f"{part_id}:{service.index}", seat_model_id)

    monkeypatch.setattr("seat_defect_inspection.api._create_service", fake_create_service)
    monkeypatch.setattr(
        "seat_defect_inspection.api._run_online_inspection",
        fake_run_online_inspection,
    )

    inspector = SeatDefectInspector(config)
    first = inspector.inspect(part_id="seat_001")
    second = inspector.inspect(part_id="seat_002")

    assert len(created_services) == 1
    assert first.part_id == "seat_001:0"
    assert second.part_id == "seat_002:0"


def test_inspect_folder_once_routes_to_offline_api(monkeypatch) -> None:
    config = InspectionConfig()
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "seat_defect_inspection.api._create_service",
        lambda config_arg: SimpleNamespace(config=config_arg),
    )

    def fake_run_offline_folder_inspection(
        service,
        *,
        input_dir,
        seat_model_id,
        output_dir,
        part_id,
    ):
        observed["config"] = service.config
        observed["input_dir"] = input_dir
        observed["seat_model_id"] = seat_model_id
        observed["output_dir"] = output_dir
        observed["part_id"] = part_id
        return {"sample_count": 1, "ok_count": 1}

    monkeypatch.setattr(
        "seat_defect_inspection.api._run_offline_folder_inspection",
        fake_run_offline_folder_inspection,
    )

    summary = inspect_folder_once(
        config,
        input_dir="offline_samples",
        output_dir="outputs/offline",
        part_id="seat_001",
        seat_model_id="model_a",
    )

    assert observed["config"] is config
    assert observed["input_dir"] == "offline_samples"
    assert observed["output_dir"] == "outputs/offline"
    assert observed["part_id"] == "seat_001"
    assert observed["seat_model_id"] == "model_a"
    assert summary["sample_count"] == 1
