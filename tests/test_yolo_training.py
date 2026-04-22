from __future__ import annotations

from pathlib import Path

import yaml

from seat_defect_inspection.yolo_training import _prepare_training_dataset


def _write_dataset(
    tmp_path: Path,
    *,
    train_label: str,
    val_label: str,
    task: str | None = None,
) -> Path:
    dataset_root = tmp_path / "dataset"
    for split_name, label_content in (("train", train_label), ("val", val_label)):
        image_dir = dataset_root / "images" / split_name
        label_dir = dataset_root / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / f"{split_name}_0.png").write_bytes(b"")
        (label_dir / f"{split_name}_0.txt").write_text(label_content, encoding="utf-8")

    task_line = f"task: {task}\n" if task else ""
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text(
        f"{task_line}"
        "path: dataset\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: seat\n",
        encoding="utf-8",
    )
    return dataset_yaml


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
        _prepare_training_dataset(dataset_yaml, tmp_path / "runs")
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
        _prepare_training_dataset(dataset_yaml, tmp_path / "runs")
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
        _prepare_training_dataset(dataset_yaml, tmp_path / "runs")
    except ValueError as exc:
        assert "非有限" in str(exc)
        return
    raise AssertionError("expected ValueError for non-finite coordinates")
