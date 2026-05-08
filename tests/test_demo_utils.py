from __future__ import annotations

from pathlib import Path

from demo_utils import ensure_raw_input_path


def test_ensure_raw_input_path_accepts_raw_dataset_image() -> None:
    path = ensure_raw_input_path(Path("datasets/seat_defect/images/val/1.png"))
    assert path == Path("datasets/seat_defect/images/val/1.png")


def test_ensure_raw_input_path_rejects_yolo_debug_visualization() -> None:
    try:
        ensure_raw_input_path(Path("runs/segment/outputs/yolo_debug/debug_overlay/output.jpg"))
    except ValueError as exc:
        assert "YOLO 调试可视化结果" in str(exc)
        return
    raise AssertionError("expected ValueError for yolo_debug visualization input")
