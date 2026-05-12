from __future__ import annotations

import json
from pathlib import Path

import pytest

from seat_defect_core.runtime_config import load_config as load_core_config
from seat_defect_inspection.runtime_config import load_config, load_yolo_training_config


def test_load_config_parses_ini_single_camera(tmp_path: Path) -> None:
    config_path = tmp_path / "config.ini"
    config_path.write_text(
        """
[seat_defect_inspection]
output_json_path = outputs/results.json
debug_dir = outputs/debug
debug_artifacts_enabled = false
capture_dir = outputs/capture
part_id = seat_labview

[fusion]
reject_on_any_reject = true
ng_strategy = any
defect_overrides_reject = true

[camera.cam_0]
source = 0
patchcore_model_path = models/cam_0_patchcore.npz
train_good_dir = train/cam_0/good
enabled = true
color_insensitive_mode = true

[camera.cam_0.detection]
model_path = models/yolo.pt
target_class = seat
confidence = 0.4
imgsz = 960
fill_segmentation_holes = false
segmentation_hole_fill_max_area_ratio = 0.03

[camera.cam_0.roi.alignment]
output_width = 320
output_height = 320

[camera.cam_0.patchcore]
backend = full
backbone_weights_path = models/backbone.pth
feature_layers = layer2, layer3

[camera.cam_0.region.upper]
box = 0.0, 0.0, 1.0, 0.5
patchcore_model_path = models/cam_0_upper_patchcore.npz
enabled = true
""",
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.output_json_path == str((tmp_path / "outputs/results.json").resolve())
    assert config.debug_artifacts_enabled is False
    assert config.capture_dir == str((tmp_path / "outputs/capture").resolve())
    assert config.part_id == "seat_labview"
    camera = config.cameras[0]
    assert camera.camera_id == "cam_0"
    assert camera.source == "0"
    assert camera.patchcore_model_path == str((tmp_path / "models/cam_0_patchcore.npz").resolve())
    assert camera.train_good_dir == str((tmp_path / "train/cam_0/good").resolve())
    assert camera.detection.model_path == str((tmp_path / "models/yolo.pt").resolve())
    assert camera.detection.confidence == 0.4
    assert camera.detection.imgsz == 960
    assert camera.detection.fill_segmentation_holes is False
    assert camera.detection.segmentation_hole_fill_max_area_ratio == 0.03
    assert camera.roi.alignment.output_width == 320
    assert camera.patchcore.feature_layers == ["layer2", "layer3"]
    assert camera.regions[0].region_id == "upper"
    assert camera.regions[0].box == [0.0, 0.0, 1.0, 0.5]


def test_core_load_config_parses_ini_without_engineering_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "core.ini"
    config_path.write_text(
        """
[seat_defect_inspection]
output_json_path = outputs/results.json
debug_artifacts_enabled = false

[camera.cam_0]
source = 0
patchcore_model_path = models/cam_0_patchcore.npz

[camera.cam_0.patchcore]
backbone_pretrained = true
""",
        encoding="utf-8",
    )

    config = load_core_config(str(config_path))

    assert config.output_json_path == str((tmp_path / "outputs/results.json").resolve())
    assert config.debug_artifacts_enabled is False
    assert config.cameras[0].camera_id == "cam_0"


def test_core_load_config_parses_debug_artifact_names(tmp_path: Path) -> None:
    config_path = tmp_path / "core.ini"
    config_path.write_text(
        """
[seat_defect_inspection]
debug_artifact_names = overlay

[camera.cam_0]
source = 0
patchcore_model_path = models/cam_0_patchcore.npz

[camera.cam_0.patchcore]
backbone_pretrained = true
""",
        encoding="utf-8",
    )

    config = load_core_config(str(config_path))

    assert config.debug_artifact_names == ["overlay"]


def test_load_config_parses_ini_seat_models_and_yolo_training(tmp_path: Path) -> None:
    config_path = tmp_path / "config.ini"
    config_path.write_text(
        """
[seat_defect_inspection]
default_seat_model_id = seat_b

[yolo_training]
model_path = yolo11m-seg.pt
data_config_path = datasets/top_level.yaml

[seat_model.seat_a]
display_name = Seat A

[seat_model.seat_a.camera.cam_front]
source = images/a_front.png
patchcore_model_path = models/a_front_patchcore.npz

[seat_model.seat_a.camera.cam_front.patchcore]
backbone_pretrained = true

[seat_model.seat_b]
display_name = Seat B

[seat_model.seat_b.yolo_training]
model_path = models/seat_b_yolo.pt
data_config_path = datasets/seat_b.yaml
project = runs/seat_b

[seat_model.seat_b.camera.cam_front]
source = images/b_front.png
patchcore_model_path = models/b_front_patchcore.npz

[seat_model.seat_b.camera.cam_front.patchcore]
backbone_pretrained = true
""",
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.default_seat_model_id == "seat_b"
    assert [item.seat_model_id for item in config.seat_models] == ["seat_a", "seat_b"]
    assert config.seat_models[1].display_name == "Seat B"
    assert config.seat_models[1].cameras[0].source == str((tmp_path / "images/b_front.png").resolve())

    training = load_yolo_training_config(str(config_path), seat_model_id="seat_b")

    assert training.seat_model_id == "seat_b"
    assert training.model_path == str((tmp_path / "models/seat_b_yolo.pt").resolve())
    assert training.data_config_path == str((tmp_path / "datasets/seat_b.yaml").resolve())
    assert training.project == str((tmp_path / "runs/seat_b").resolve())


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
                                        "imgsz": 960,
                                    },
                                    "roi": {
                                        "alignment": {
                                            "output_width": 320,
                                            "output_height": 320,
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
    camera = config.seat_models[0].cameras[0]
    assert camera.source == str((tmp_path / "images/frame.png").resolve())
    assert camera.patchcore_model_path == str((tmp_path / "models/patchcore.npz").resolve())
    assert camera.train_good_dir == str((tmp_path / "train/good").resolve())
    assert camera.detection.model_path == str((tmp_path / "models/yolo.pt").resolve())
    assert camera.detection.imgsz == 960
    assert camera.roi.alignment.output_width == 320
    assert camera.roi.alignment.output_height == 320
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


def test_load_config_rejects_legacy_debug_artifact_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seat_defect_inspection": {
                    "save_debug_artifacts": False,
                    "debug_artifact_mode": "verbose",
                    "cameras": [
                        {
                            "camera_id": "cam_0",
                            "source": "0",
                            "patchcore_model_path": "model.npz",
                            "patchcore": {
                                "backbone_pretrained": True
                            },
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="未知字段"):
        load_config(str(config_path))


def test_load_config_parses_debug_artifacts_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seat_defect_inspection": {
                    "debug_artifacts_enabled": False,
                    "cameras": [
                        {
                            "camera_id": "cam_0",
                            "source": "0",
                            "patchcore_model_path": "model.npz",
                            "patchcore": {
                                "backbone_pretrained": True
                            },
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.debug_artifacts_enabled is False


def test_load_config_parses_string_false_bool_without_truthiness_trap(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seat_defect_inspection": {
                    "debug_artifacts_enabled": "false",
                    "cameras": [
                        {
                            "camera_id": "cam_0",
                            "source": "0",
                            "patchcore_model_path": "model.npz",
                            "patchcore": {
                                "backbone_pretrained": True
                            },
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.debug_artifacts_enabled is False


def test_load_config_rejects_non_bool_config_values(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seat_defect_inspection": {
                    "debug_artifacts_enabled": 1,
                    "cameras": [
                        {
                            "camera_id": "cam_0",
                            "source": "0",
                            "patchcore_model_path": "model.npz",
                            "patchcore": {
                                "backbone_pretrained": True
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
    except TypeError as exc:
        assert "布尔配置必须是 true/false" in str(exc)
        return
    raise AssertionError("expected TypeError for non-bool config value")


def test_load_config_ignores_legacy_ignore_classes_field(tmp_path: Path) -> None:
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
                            "detection": {
                                "target_class": "seat",
                                "ignore_classes": ["wire", "tooling"],
                            },
                            "patchcore": {
                                "backbone_weights_path": "models/backbone.pth"
                            },
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    camera = config.cameras[0]
    assert camera.detection.target_class == "seat"
    assert not hasattr(camera.detection, "ignore_classes")


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


def test_load_config_rejects_handcrafted_patchcore_backend(tmp_path: Path) -> None:
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
                                "backend": "handcrafted",
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
        assert "patchcore.backend" in message
        assert "handcrafted" in message
        assert "可选值: full" in message
        return
    raise AssertionError("expected ValueError for handcrafted patchcore backend")


def test_load_config_rejects_duplicate_camera_ids(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seat_defect_inspection": {
                    "cameras": [
                        {
                            "camera_id": "cam_0",
                            "source": "0",
                            "patchcore_model_path": "model_a.npz",
                        },
                        {
                            "camera_id": "cam_0",
                            "source": "1",
                            "patchcore_model_path": "model_b.npz",
                        },
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
        assert "重复 camera_id" in message
        assert "cam_0" in message
        return
    raise AssertionError("expected ValueError for duplicate camera_id")


def test_load_config_rejects_unknown_nested_field(tmp_path: Path) -> None:
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
                            "unexpected_field": True,
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
        assert "CameraConfig" in message
        assert "unexpected_field" in message
        return
    raise AssertionError("expected ValueError for unknown nested config field")
