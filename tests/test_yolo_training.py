from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import yaml

from seat_defect_core.config import PreprocessConfig
from seat_defect_inspection.config import YoloTrainingConfig
from seat_defect_inspection.yolo.training import (
    YOLO_SEGMENT_MODEL,
    YOLO_SEGMENT_MODEL_YAML,
    _import_ultralytics_yolo,
    _load_yolo_model,
    _prepare_preprocessed_dataset,
    _prepare_training_dataset,
)


def _write_dataset(
    tmp_path: Path,
    *,
    train_label: str,
    val_label: str,
    task: str | None = None,
    declare_test: bool = False,
    create_test: bool = False,
    test_label: str | None = None,
) -> Path:
    dataset_root = tmp_path / "dataset"
    splits = [("train", train_label), ("val", val_label)]
    if create_test:
        splits.append(("test", test_label or train_label))

    for split_name, label_content in splits:
        image_dir = dataset_root / "images" / split_name
        label_dir = dataset_root / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / f"{split_name}_0.png").write_bytes(b"placeholder")
        (label_dir / f"{split_name}_0.txt").write_text(label_content, encoding="utf-8")

    task_line = f"task: {task}\n" if task else ""
    test_line = "test: images/test\n" if declare_test else ""
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text(
        f"{task_line}"
        "path: dataset\n"
        "train: images/train\n"
        "val: images/val\n"
        f"{test_line}"
        "names:\n"
        "  0: seat\n",
        encoding="utf-8",
    )
    return dataset_yaml


def _build_training_config(tmp_path: Path) -> YoloTrainingConfig:
    return YoloTrainingConfig(
        data_config_path=str(tmp_path / "dataset.yaml"),
        project=str(tmp_path / "runs"),
        name="seat_defect",
        preprocess=PreprocessConfig(),
    )


def test_prepare_training_dataset_accepts_polygon_labels_for_seg_model(tmp_path: Path) -> None:
    polygon_label = "0 0.10 0.10 0.80 0.10 0.80 0.85 0.12 0.88\n"
    dataset_yaml = _write_dataset(
        tmp_path,
        train_label=polygon_label,
        val_label=polygon_label,
    )

    dataset_root, resolved_path = _prepare_training_dataset(
        dataset_yaml,
        tmp_path / "runs",
    )

    assert dataset_root == (tmp_path / "dataset").resolve()
    assert resolved_path.exists()
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    assert resolved["task"] == "segment"


def test_prepare_training_dataset_rejects_box_labels(tmp_path: Path) -> None:
    dataset_yaml = _write_dataset(
        tmp_path,
        train_label="0 0.5 0.5 0.4 0.6\n",
        val_label="0 0.4 0.4 0.3 0.5\n",
    )

    try:
        _prepare_training_dataset(
            dataset_yaml,
            tmp_path / "runs",
        )
    except ValueError as exc:
        assert "多边形" in str(exc)
        return
    raise AssertionError("expected ValueError for box labels")


def test_prepare_training_dataset_rejects_detect_task_in_dataset_yaml(tmp_path: Path) -> None:
    polygon_label = "0 0.10 0.10 0.80 0.10 0.80 0.85 0.12 0.88\n"
    dataset_yaml = _write_dataset(
        tmp_path,
        train_label=polygon_label,
        val_label=polygon_label,
        task="detect",
    )

    try:
        _prepare_training_dataset(dataset_yaml, tmp_path / "runs")
    except ValueError as exc:
        assert "task" in str(exc)
        assert "segment" in str(exc)
        return
    raise AssertionError("expected ValueError for detect task")


def test_prepare_training_dataset_rejects_unknown_class_id(tmp_path: Path) -> None:
    dataset_yaml = _write_dataset(
        tmp_path,
        train_label="1 0.10 0.10 0.80 0.10 0.80 0.85 0.12 0.88\n",
        val_label="1 0.10 0.10 0.80 0.10 0.80 0.85 0.12 0.88\n",
    )

    try:
        _prepare_training_dataset(
            dataset_yaml,
            tmp_path / "runs",
        )
    except ValueError as exc:
        assert "类别 id" in str(exc)
        assert "names" in str(exc)
        return
    raise AssertionError("expected ValueError for unknown class id")


def test_prepare_training_dataset_rejects_out_of_range_coordinates(tmp_path: Path) -> None:
    dataset_yaml = _write_dataset(
        tmp_path,
        train_label="0 0.10 0.10 1.20 0.10 0.80 0.85 0.12 0.88\n",
        val_label="0 0.10 0.10 1.20 0.10 0.80 0.85 0.12 0.88\n",
    )

    try:
        _prepare_training_dataset(
            dataset_yaml,
            tmp_path / "runs",
        )
    except ValueError as exc:
        assert "[0, 1]" in str(exc)
        return
    raise AssertionError("expected ValueError for out-of-range coordinates")


def test_prepare_training_dataset_rejects_non_finite_coordinates(tmp_path: Path) -> None:
    dataset_yaml = _write_dataset(
        tmp_path,
        train_label="0 0.10 0.10 inf 0.10 0.80 0.85 0.12 0.88\n",
        val_label="0 0.10 0.10 inf 0.10 0.80 0.85 0.12 0.88\n",
    )

    try:
        _prepare_training_dataset(
            dataset_yaml,
            tmp_path / "runs",
        )
    except ValueError as exc:
        assert "非有限" in str(exc)
        return
    raise AssertionError("expected ValueError for non-finite coordinates")


def test_prepare_preprocessed_dataset_copies_only_referenced_splits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    polygon_label = "0 0.10 0.10 0.80 0.10 0.80 0.85 0.12 0.88\n"
    dataset_yaml = _write_dataset(
        tmp_path,
        train_label=polygon_label,
        val_label=polygon_label,
    )
    dataset_root = tmp_path / "dataset"
    (dataset_root / "README.md").write_text("ignore me", encoding="utf-8")
    (dataset_root / "images" / "unused").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "unused").mkdir(parents=True, exist_ok=True)

    prepared_root, resolved_path = _prepare_training_dataset(dataset_yaml, tmp_path / "runs")
    config = _build_training_config(tmp_path)
    expected_target_root = tmp_path / "runs" / "_preprocessed_fixture"
    processed_dirs: list[Path] = []

    monkeypatch.setattr(
        "seat_defect_inspection.yolo.training._build_preprocessed_dataset_root",
        lambda project_root, config: expected_target_root,
    )
    monkeypatch.setattr(
        "seat_defect_inspection.yolo.training._preprocess_image_dir",
        lambda image_dir, engine: processed_dirs.append(image_dir),
    )

    target_root, target_config_path = _prepare_preprocessed_dataset(
        dataset_root=prepared_root,
        resolved_data_config_path=resolved_path,
        config=config,
    )

    assert target_root == expected_target_root
    assert (target_root / "images" / "train" / "train_0.png").exists()
    assert (target_root / "images" / "val" / "val_0.png").exists()
    assert (target_root / "labels" / "train" / "train_0.txt").exists()
    assert (target_root / "labels" / "val" / "val_0.txt").exists()
    assert not (target_root / "README.md").exists()
    assert not (target_root / "images" / "unused").exists()
    assert processed_dirs == [
        target_root / "images" / "train",
        target_root / "images" / "val",
    ]

    resolved = yaml.safe_load(target_config_path.read_text(encoding="utf-8"))
    assert resolved["path"] == str(target_root)
    assert resolved["train"] == "images/train"
    assert resolved["val"] == "images/val"


def test_prepare_preprocessed_dataset_skips_missing_optional_test_split(
    tmp_path: Path,
    monkeypatch,
) -> None:
    polygon_label = "0 0.10 0.10 0.80 0.10 0.80 0.85 0.12 0.88\n"
    dataset_yaml = _write_dataset(
        tmp_path,
        train_label=polygon_label,
        val_label=polygon_label,
        declare_test=True,
        create_test=False,
    )

    prepared_root, resolved_path = _prepare_training_dataset(dataset_yaml, tmp_path / "runs")
    config = _build_training_config(tmp_path)
    expected_target_root = tmp_path / "runs" / "_preprocessed_missing_test"

    monkeypatch.setattr(
        "seat_defect_inspection.yolo.training._build_preprocessed_dataset_root",
        lambda project_root, config: expected_target_root,
    )
    monkeypatch.setattr(
        "seat_defect_inspection.yolo.training._preprocess_image_dir",
        lambda image_dir, engine: None,
    )

    _, target_config_path = _prepare_preprocessed_dataset(
        dataset_root=prepared_root,
        resolved_data_config_path=resolved_path,
        config=config,
    )

    resolved = yaml.safe_load(target_config_path.read_text(encoding="utf-8"))
    assert "test" not in resolved
    assert not (expected_target_root / "images" / "test").exists()


def test_prepare_preprocessed_dataset_allows_reused_split_directories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    polygon_label = "0 0.10 0.10 0.80 0.10 0.80 0.85 0.12 0.88\n"
    dataset_yaml = _write_dataset(
        tmp_path,
        train_label=polygon_label,
        val_label=polygon_label,
    )
    dataset_yaml.write_text(
        "path: dataset\n"
        "train: images/train\n"
        "val: images/train\n"
        "names:\n"
        "  0: seat\n",
        encoding="utf-8",
    )

    prepared_root, resolved_path = _prepare_training_dataset(dataset_yaml, tmp_path / "runs")
    config = _build_training_config(tmp_path)
    expected_target_root = tmp_path / "runs" / "_preprocessed_reused_split"
    processed_dirs: list[Path] = []

    monkeypatch.setattr(
        "seat_defect_inspection.yolo.training._build_preprocessed_dataset_root",
        lambda project_root, config: expected_target_root,
    )
    monkeypatch.setattr(
        "seat_defect_inspection.yolo.training._preprocess_image_dir",
        lambda image_dir, engine: processed_dirs.append(image_dir),
    )

    _, target_config_path = _prepare_preprocessed_dataset(
        dataset_root=prepared_root,
        resolved_data_config_path=resolved_path,
        config=config,
    )

    resolved = yaml.safe_load(target_config_path.read_text(encoding="utf-8"))
    assert resolved["train"] == "images/train"
    assert resolved["val"] == "images/train"
    assert (expected_target_root / "images" / "train" / "train_0.png").exists()
    assert (expected_target_root / "labels" / "train" / "train_0.txt").exists()
    assert not (expected_target_root / "images" / "val").exists()
    assert processed_dirs == [expected_target_root / "images" / "train"]


def test_prepare_preprocessed_dataset_cleans_up_target_root_on_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    polygon_label = "0 0.10 0.10 0.80 0.10 0.80 0.85 0.12 0.88\n"
    dataset_yaml = _write_dataset(
        tmp_path,
        train_label=polygon_label,
        val_label=polygon_label,
    )

    prepared_root, resolved_path = _prepare_training_dataset(dataset_yaml, tmp_path / "runs")
    config = _build_training_config(tmp_path)
    expected_target_root = tmp_path / "runs" / "_preprocessed_failure"

    monkeypatch.setattr(
        "seat_defect_inspection.yolo.training._build_preprocessed_dataset_root",
        lambda project_root, config: expected_target_root,
    )

    def _raise_on_preprocess(image_dir: Path, engine) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "seat_defect_inspection.yolo.training._preprocess_image_dir",
        _raise_on_preprocess,
    )

    try:
        _prepare_preprocessed_dataset(
            dataset_root=prepared_root,
            resolved_data_config_path=resolved_path,
            config=config,
        )
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected preprocess failure")

    assert not expected_target_root.exists()


def test_import_ultralytics_yolo_ensures_local_config_dir(monkeypatch) -> None:
    class DummyYOLO:
        pass

    fake_module = ModuleType("ultralytics")
    fake_module.YOLO = DummyYOLO
    ensure_calls: list[str] = []

    monkeypatch.setattr(
        "seat_defect_core.yolo.detection._ensure_local_yolo_config_dir",
        lambda: ensure_calls.append("called"),
    )
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)

    yolo_cls = _import_ultralytics_yolo()

    assert yolo_cls is DummyYOLO
    assert ensure_calls == ["called"]


def test_load_yolo_model_falls_back_on_requests_style_connection_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeRequestsConnectionError(Exception):
        pass

    FakeRequestsConnectionError.__module__ = "requests.exceptions"

    class DummyModel:
        def __init__(self, path: str, task: str) -> None:
            self.path = path
            self.task = task

    calls: list[tuple[str, str]] = []

    def fake_yolo(path: str, *, task: str) -> DummyModel:
        calls.append((path, task))
        if path == YOLO_SEGMENT_MODEL:
            raise FakeRequestsConnectionError("Failed to establish a new connection")
        return DummyModel(path, task)

    monkeypatch.chdir(tmp_path)
    config = YoloTrainingConfig(model_path=YOLO_SEGMENT_MODEL, pretrained=True)

    model, resolved_model_path, effective_pretrained = _load_yolo_model(config, fake_yolo)

    assert isinstance(model, DummyModel)
    assert model.path == YOLO_SEGMENT_MODEL_YAML
    assert calls == [
        (YOLO_SEGMENT_MODEL, "segment"),
        (YOLO_SEGMENT_MODEL_YAML, "segment"),
    ]
    assert resolved_model_path == YOLO_SEGMENT_MODEL_YAML
    assert effective_pretrained is False
