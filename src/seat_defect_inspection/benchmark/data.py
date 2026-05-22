"""Dataset loading and ground-truth handling for benchmark evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import BenchmarkConfig
from .schemas import BenchmarkSample

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
GROUND_TRUTH_FILENAME = "ground_truth.json"


# ---------- entry ----------


def discover_benchmark_samples(
    config: BenchmarkConfig,
) -> Dict[str, List[BenchmarkSample]]:
    """Discover all benchmark rounds and load their samples.

    Returns
    -------
    Dict mapping round name → list of samples.
    """
    data_root = Path(config.data_dir)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Benchmark data directory not found: {data_root}")

    result: Dict[str, List[BenchmarkSample]] = {}
    for round_name in config.rounds:
        round_dir = data_root / round_name
        if not round_dir.is_dir():
            continue
        camera_ids = _resolve_camera_ids(round_dir, config.camera_ids)
        ground_truth = _load_ground_truth(round_dir)
        implicit_label = _infer_implicit_label(round_name, ground_truth)
        samples = _build_samples(round_dir, round_name, camera_ids, ground_truth, implicit_label)
        result[round_name] = samples
    return result


# ---------- camera discovery ----------


def _resolve_camera_ids(
    round_dir: Path,
    requested_ids: Optional[List[str]],
) -> List[str]:
    camera_dirs = sorted(
        d.name for d in round_dir.iterdir()
        if d.is_dir() and d.name != GROUND_TRUTH_FILENAME
    )
    if not camera_dirs:
        raise FileNotFoundError(f"No camera directories found in {round_dir}")
    if requested_ids is not None:
        unknown = set(requested_ids) - set(camera_dirs)
        if unknown:
            raise ValueError(
                f"Unknown camera(s) in {round_dir}: {', '.join(sorted(unknown))}. "
                f"Available: {', '.join(camera_dirs)}"
            )
        return [cid for cid in requested_ids if cid in camera_dirs]
    return camera_dirs


# ---------- ground truth ----------


def _load_ground_truth(round_dir: Path) -> Optional[List[Dict[str, Any]]]:
    gt_path = round_dir / GROUND_TRUTH_FILENAME
    if not gt_path.is_file():
        return None
    with open(gt_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    samples = manifest.get("samples", [])
    for entry in samples:
        if "index" not in entry:
            raise ValueError(f"Missing 'index' in ground truth entry: {entry}")
        if "label" not in entry:
            raise ValueError(f"Missing 'label' in ground truth entry: {entry}")
    return samples


def _infer_implicit_label(
    round_name: str,
    ground_truth: Optional[List[Dict[str, Any]]],
) -> Optional[str]:
    if ground_truth is not None:
        return None
    if round_name == "good":
        return "OK"
    if round_name == "defect":
        return "NG"
    return None


def _resolve_gt_for_sample(
    ground_truth: Optional[List[Dict[str, Any]]],
    implicit_label: Optional[str],
    sample_index: int,
) -> Tuple[Optional[str], Optional[str], Optional[str], Dict[str, str]]:
    """Return (label, defect_type, severity, camera_ground_truth)."""
    if ground_truth is not None:
        for entry in ground_truth:
            if entry["index"] == sample_index:
                cam_gt: Dict[str, str] = {}
                cam_results = entry.get("camera_results", {})
                if isinstance(cam_results, dict):
                    for cid, info in cam_results.items():
                        if isinstance(info, dict) and "label" in info:
                            cam_gt[cid] = info["label"]
                return (
                    entry["label"],
                    entry.get("defect_type"),
                    entry.get("severity"),
                    cam_gt,
                )
        return (None, None, None, {})
    if implicit_label is not None:
        return (implicit_label, None, None, {})
    return (None, None, None, {})


# ---------- sample builder ----------


def _build_samples(
    round_dir: Path,
    round_name: str,
    camera_ids: List[str],
    ground_truth: Optional[List[Dict[str, Any]]],
    implicit_label: Optional[str],
) -> List[BenchmarkSample]:
    camera_images = _collect_camera_images(round_dir, camera_ids)
    sample_count = len(next(iter(camera_images.values())))
    samples: List[BenchmarkSample] = []
    for idx in range(sample_count):
        label, defect_type, severity, cam_gt = _resolve_gt_for_sample(
            ground_truth, implicit_label, idx
        )
        samples.append(BenchmarkSample(
            index=idx,
            part_id=f"{round_name}_{idx:04d}",
            image_paths={cid: str(camera_images[cid][idx]) for cid in camera_ids},
            ground_truth_label=label,
            ground_truth_defect_type=defect_type,
            ground_truth_severity=severity,
            camera_ground_truth=cam_gt,
        ))
    return samples


def _collect_camera_images(
    round_dir: Path,
    camera_ids: List[str],
) -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = {}
    counts: List[int] = []
    for cid in camera_ids:
        cam_dir = round_dir / cid
        if not cam_dir.is_dir():
            raise FileNotFoundError(f"Camera directory not found: {cam_dir}")
        images = sorted(
            p for p in cam_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise FileNotFoundError(f"No images found in {cam_dir}")
        result[cid] = images
        counts.append(len(images))
    if len(set(counts)) != 1:
        detail = ", ".join(f"{cid}={len(result[cid])}" for cid in camera_ids)
        raise ValueError(f"Camera image counts must be equal, got: {detail}")
    return result
