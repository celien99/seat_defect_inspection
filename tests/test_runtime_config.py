from __future__ import annotations

import json
from pathlib import Path

from seat_defect_inspection.runtime_config import load_config, load_yolo_training_config


def test_load_config_normalizes_nested_paths_and_default_seat_model(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seat_defect_inspection": {
                    "seat_models": [
                        {
                            "seat_model_id": "seat_model_a",
                            "yolo_training": {
                                "data_config_path": "dataset.yaml",
                                "project": "runs/yolo",
                            },
                            "cameras": [
                                {
                                    "camera_id": "cam_0",
                                    "source": "images/frame.png",
                                    "patchcore_model_path": "models/patchcore.npz",
                                    "train_good_dir": "train/good",
                                    "detection": {
                                        "model_path": "models/yolo.pt",
                                        "fallback_box": {
                                            "x1": 1,
                                            "y1": 2,
                                            "x2": 30,
                                            "y2": 40,
                                        },
                                    },
                                    "roi": {
                                        "alignment": {
                                            "enabled": True,
                                            "template_image_path": "templates/alignment.png",
                                        }
                                    },
                                    "patchcore": {
                                        "backbone_weights_path": "models/backbone.pth"
                                    },
                                }
                            ],
                        }
                    ],
                    "yolo_training": {
                        "data_config_path": "dataset.yaml",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.default_seat_model_id == "seat_model_a"
    assert config.output_json_path == str(
        (tmp_path / "outputs/seat_defect_inspection/results.json").resolve()
    )
    assert config.debug_dir == str(
        (tmp_path / "outputs/seat_defect_inspection/debug").resolve()
    )
    assert config.capture_dir == str(
        (tmp_path / "outputs/seat_defect_inspection/capture").resolve()
    )
    assert config.debug_artifact_mode == "standard"

    camera = config.seat_models[0].cameras[0]
    assert camera.source == str((tmp_path / "images/frame.png").resolve())
    assert camera.patchcore_model_path == str((tmp_path / "models/patchcore.npz").resolve())
    assert camera.train_good_dir == str((tmp_path / "train/good").resolve())
    assert camera.detection.model_path == str((tmp_path / "models/yolo.pt").resolve())
    assert camera.detection.fallback_box is not None
    assert camera.detection.fallback_box.x1 == 1
    assert camera.detection.fallback_box.y2 == 40
    assert camera.roi.alignment.template_image_path == str(
        (tmp_path / "templates/alignment.png").resolve()
    )
    assert camera.patchcore.backbone_weights_path == str(
        (tmp_path / "models/backbone.pth").resolve()
    )
    assert config.seat_models[0].yolo_training is not None
    assert config.seat_models[0].yolo_training.seat_model_id == "seat_model_a"
    assert config.seat_models[0].yolo_training.data_config_path == str(
        (tmp_path / "dataset.yaml").resolve()
    )
    assert config.seat_models[0].yolo_training.project == str(
        (tmp_path / "runs/yolo").resolve()
    )


def test_load_yolo_training_falls_back_to_top_level_for_selected_seat_model(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    dataset_path = tmp_path / "dataset.yaml"
    dataset_path.write_text("path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "seat_defect_inspection": {
                    "default_seat_model_id": "seat_model_b",
                    "seat_models": [
                        {
                            "seat_model_id": "seat_model_a",
                            "cameras": [
                                {
                                    "camera_id": "cam_0",
                                    "source": "0",
                                    "patchcore_model_path": "model_a.npz",
                                }
                            ],
                        },
                        {
                            "seat_model_id": "seat_model_b",
                            "cameras": [
                                {
                                    "camera_id": "cam_1",
                                    "source": "1",
                                    "patchcore_model_path": "model_b.npz",
                                }
                            ],
                        },
                    ],
                    "yolo_training": {
                        "model_path": "yolo11m-seg.pt",
                        "data_config_path": "dataset.yaml",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_yolo_training_config(str(config_path), seat_model_id="seat_model_a")

    assert config.seat_model_id == "seat_model_a"
    assert config.model_path == "yolo11m-seg.pt"
    assert config.data_config_path == str(dataset_path.resolve())
    assert config.project == str((tmp_path / "outputs/yolo_training").resolve())


def test_load_yolo_training_default_data_path_is_resolved(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seat_defect_inspection": {
                    "cameras": [
                        {
                            "camera_id": "cam_0",
                            "source": "0",
                            "patchcore_model_path": "model.npz",
                        }
                    ],
                    "yolo_training": {},
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_yolo_training_config(str(config_path))

    assert config.model_path == "yolo11m-seg.pt"
    assert config.data_config_path == str(
        (tmp_path / "configs/seat_defect_yolo.dataset.example.yaml").resolve()
    )


def test_load_config_rejects_unknown_debug_artifact_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seat_defect_inspection": {
                    "debug_artifact_mode": "verbose",
                    "cameras": [
                        {
                            "camera_id": "cam_0",
                            "source": "0",
                            "patchcore_model_path": "model.npz",
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    try:
        load_config(str(config_path))
    except ValueError as exc:
        message = str(exc)
        assert "debug_artifact_mode" in message
        return
    raise AssertionError("expected ValueError for unknown debug_artifact_mode")


def test_load_config_rejects_full_patchcore_without_backbone_weights(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seat_defect_inspection": {
                    "cameras": [
                        {
                            "camera_id": "cam_0",
                            "source": "0",
                            "patchcore_model_path": "model.npz",
                            "patchcore": {
                                "backend": "full",
                                "backbone_pretrained": False,
                            },
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    try:
        load_config(str(config_path))
    except ValueError as exc:
        message = str(exc)
        assert "patchcore.backend=full" in message
        assert "cam_0" in message
        return
    raise AssertionError("expected ValueError for full patchcore without backbone weights")
