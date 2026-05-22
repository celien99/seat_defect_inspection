"""Tests for benchmark data loading and ground truth parsing."""

import json
from pathlib import Path

import cv2
import numpy as np

from seat_defect_inspection.benchmark.config import BenchmarkConfig
from seat_defect_inspection.benchmark.data import (
    discover_benchmark_samples,
)


def _build_benchmark_dataset(root: Path, rounds=None) -> Path:
    """Create a minimal synthetic benchmark dataset structure."""
    if rounds is None:
        rounds = {"good": 3, "defect": 2}
    for round_name, count in rounds.items():
        round_dir = root / round_name
        for cid in ("cam_0", "cam_1"):
            cam_dir = round_dir / cid
            cam_dir.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(cam_dir / f"sample_{i:04d}.png"), img)
    return root


def _write_ground_truth(round_dir: Path, entries: list) -> Path:
    path = round_dir / "ground_truth.json"
    path.write_text(json.dumps({"samples": entries}, ensure_ascii=False))
    return path


class TestDiscoverSamples:
    def test_basic_discovery(self, tmp_path):
        root = _build_benchmark_dataset(tmp_path, {"good": 3})
        config = BenchmarkConfig(data_dir=str(root), rounds=("good",))
        result = discover_benchmark_samples(config)
        assert "good" in result
        samples, composition = result["good"]
        assert len(samples) == 3
        assert composition.sample_count == 3
        assert composition.camera_count == 2
        assert composition.camera_ids == ["cam_0", "cam_1"]
        # Implicit label for "good" round
        assert composition.has_ground_truth is True
        assert composition.ground_truth_source == "implicit"
        for s in samples:
            assert s.ground_truth_label == "OK"
            assert "cam_0" in s.image_paths
            assert "cam_1" in s.image_paths

    def test_implicit_defect_labels(self, tmp_path):
        root = _build_benchmark_dataset(tmp_path, {"defect": 2})
        config = BenchmarkConfig(data_dir=str(root), rounds=("defect",))
        result = discover_benchmark_samples(config)
        samples, composition = result["defect"]
        assert composition.ground_truth_source == "implicit"
        for s in samples:
            assert s.ground_truth_label == "NG"

    def test_mixed_no_implicit(self, tmp_path):
        root = _build_benchmark_dataset(tmp_path, {"mixed": 2})
        config = BenchmarkConfig(data_dir=str(root), rounds=("mixed",))
        result = discover_benchmark_samples(config)
        samples, composition = result["mixed"]
        # Mixed round has no implicit label without manifest
        assert composition.ground_truth_source == "none"
        for s in samples:
            assert s.ground_truth_label is None

    def test_manifest_based_labels(self, tmp_path):
        root = _build_benchmark_dataset(tmp_path, {"mixed": 3})
        _write_ground_truth(root / "mixed", [
            {"index": 0, "label": "OK"},
            {"index": 1, "label": "NG", "defect_type": "scratch", "severity": "severe"},
            {"index": 2, "label": "OK"},
        ])
        config = BenchmarkConfig(data_dir=str(root), rounds=("mixed",))
        result = discover_benchmark_samples(config)
        samples, composition = result["mixed"]
        assert composition.ground_truth_source == "manifest"
        assert samples[0].ground_truth_label == "OK"
        assert samples[1].ground_truth_label == "NG"
        assert samples[1].ground_truth_defect_type == "scratch"
        assert samples[1].ground_truth_severity == "severe"
        assert samples[2].ground_truth_label == "OK"

    def test_per_camera_gt(self, tmp_path):
        root = _build_benchmark_dataset(tmp_path, {"mixed": 2})
        _write_ground_truth(root / "mixed", [
            {"index": 0, "label": "NG", "camera_results": {
                "cam_0": {"label": "NG"}, "cam_1": {"label": "OK"}
            }},
            {"index": 1, "label": "OK", "camera_results": {
                "cam_0": {"label": "OK"}, "cam_1": {"label": "OK"}
            }},
        ])
        config = BenchmarkConfig(data_dir=str(root), rounds=("mixed",))
        result = discover_benchmark_samples(config)
        samples, _ = result["mixed"]
        assert samples[0].camera_ground_truth == {"cam_0": "NG", "cam_1": "OK"}
        assert samples[1].camera_ground_truth == {"cam_0": "OK", "cam_1": "OK"}

    def test_missing_directory(self):
        config = BenchmarkConfig(data_dir="/nonexistent/path", rounds=("good",))
        try:
            discover_benchmark_samples(config)
            assert False, "Should have raised"
        except FileNotFoundError:
            pass

    def test_camera_filtering(self, tmp_path):
        root = _build_benchmark_dataset(tmp_path, {"good": 2})
        config = BenchmarkConfig(data_dir=str(root), rounds=("good",), camera_ids=["cam_0"])
        result = discover_benchmark_samples(config)
        samples, composition = result["good"]
        assert composition.camera_ids == ["cam_0"]
        # Camera filtering happens in data resolution, not image collection
        # _resolve_camera_ids returns only requested cameras

    def test_empty_round_skipped(self, tmp_path):
        root = _build_benchmark_dataset(tmp_path, {"good": 2})
        config = BenchmarkConfig(data_dir=str(root), rounds=("good", "mixed"))
        result = discover_benchmark_samples(config)
        # mixed doesn't exist, should be skipped
        assert "good" in result
        assert "mixed" not in result


class TestUnequalCameraCounts:
    def test_raises_on_unequal(self, tmp_path):
        round_dir = tmp_path / "good"
        (round_dir / "cam_0").mkdir(parents=True)
        (round_dir / "cam_1").mkdir(parents=True)
        for i in range(3):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(round_dir / "cam_0" / f"s_{i}.png"), img)
        for i in range(2):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(round_dir / "cam_1" / f"s_{i}.png"), img)
        config = BenchmarkConfig(data_dir=str(tmp_path), rounds=("good",))
        try:
            discover_benchmark_samples(config)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "equal" in str(e)
