"""Benchmark inspection pipeline with good/defect/mixed datasets."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from seat_defect_core.service.inspection import inspect_frames
from seat_defect_core.types import InspectionFrame

if TYPE_CHECKING:
    from ..config import CameraConfig
    from .core import InspectionService

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

BENCHMARK_DATA_DIR = Path(__file__).resolve().parents[3] / "benchmark_data"
DEFAULT_BENCHMARK_ARTIFACTS_DIR = (
    Path(__file__).resolve().parents[3]
    / "outputs"
    / "seat_defect_inspection"
    / "benchmark_artifacts"
)
ROUNDS = ("good", "defect", "mixed")
GROUND_TRUTH_FILENAME = "ground_truth.json"

# ---------- public entry ----------


def run_benchmark(
    service: "InspectionService",
    rounds: Tuple[str, ...] = ROUNDS,
    camera_ids: Optional[List[str]] = None,
    export_curves_dir: Optional[Path] = None,
    output_json_path: Optional[Path] = None,
    artifacts_dir: Optional[Path] = DEFAULT_BENCHMARK_ARTIFACTS_DIR,
) -> Dict[str, dict]:
    """Run benchmark inspection on selected rounds and report metrics.

    Parameters
    ----------
    rounds:
        Which rounds to run, e.g. ``("good",)`` or ``("good", "defect")``.
        Defaults to all three.
    camera_ids:
        Which cameras to benchmark. Defaults to all enabled cameras.
    export_curves_dir:
        If set, export per-round ROC/PR CSVs into this directory.
    output_json_path:
        If set, save full benchmark results as JSON to this path.
    artifacts_dir:
        If set, save each camera's production overlay image in this directory.
    """
    if not BENCHMARK_DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Benchmark data directory not found: {BENCHMARK_DATA_DIR}\n"
            "Expected structure:\n"
            "  benchmark_data/\n"
            "  ├── good/        (all OK samples)\n"
            "  │   ├── cam_0/\n"
            "  │   └── cam_1/\n"
            "  ├── defect/      (all NG samples)\n"
            "  │   ├── cam_0/\n"
            "  │   └── cam_1/\n"
            "  └── mixed/       (mixed OK/NG samples)\n"
            "      ├── cam_0/\n"
            "      └── cam_1/"
        )

    context = service.resolve_context(None)
    if not context.cameras:
        raise ValueError("No cameras configured")

    all_camera_ids = [c.camera_id for c in context.cameras]

    if camera_ids is not None:
        for cid in camera_ids:
            if cid not in all_camera_ids:
                raise ValueError(
                    f"Unknown camera '{cid}'. Available: {', '.join(all_camera_ids)}"
                )
    else:
        camera_ids = list(all_camera_ids)

    original_enabled = {c.camera_id: c.enabled for c in context.cameras}
    original_sources = {c.camera_id: c.source for c in context.cameras}

    _filter_enabled_cameras(context.cameras, camera_ids)

    try:
        results: Dict[str, dict] = {}
        for round_name in rounds:
            round_dir = BENCHMARK_DATA_DIR / round_name
            if not round_dir.is_dir():
                print(
                    f"[benchmark] Skipping '{round_name}' — directory not found: {round_dir}"
                )
                continue

            print(f"\n{'=' * 60}")
            print(f"  Benchmark round: {round_name}")
            print(f"{'=' * 60}")

            round_result = _run_single_round(
                service,
                context.cameras,
                round_dir,
                camera_ids,
                export_curves_dir=export_curves_dir,
                artifacts_dir=artifacts_dir,
            )
            results[round_name] = round_result

        _print_summary(results, camera_ids, rounds)
        if output_json_path is not None:
            _export_results_json(results, output_json_path)
        return results
    finally:
        _restore_camera_state(context.cameras, original_enabled, original_sources)


# ---------- camera state helpers ----------


def _filter_enabled_cameras(cameras: List["CameraConfig"], selected: List[str]) -> None:
    for camera in cameras:
        if camera.camera_id not in selected:
            camera.enabled = False


def _restore_camera_state(
    cameras: List["CameraConfig"],
    original_enabled: Dict[str, bool],
    original_sources: Dict[str, str],
) -> None:
    for camera in cameras:
        camera.enabled = original_enabled[camera.camera_id]
        camera.source = original_sources[camera.camera_id]


def _collect_camera_images(
    round_dir: Path,
    camera_ids: List[str],
) -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = {}
    counts: List[int] = []

    for cid in camera_ids:
        cam_dir = round_dir / cid
        if not cam_dir.is_dir():
            raise FileNotFoundError(
                f"Camera directory not found: {cam_dir}\n"
                f"Expected subdirectory '{cid}' under {round_dir}"
            )
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


# ---------- ground truth helpers ----------


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


def _get_gt_for_sample(
    ground_truth: Optional[List[Dict[str, Any]]],
    implicit_label: Optional[str],
    sample_index: int,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if ground_truth is not None:
        for entry in ground_truth:
            if entry["index"] == sample_index:
                return (
                    entry["label"],
                    entry.get("defect_type"),
                    entry.get("severity"),
                )
        return (None, None, None)
    if implicit_label is not None:
        return (implicit_label, None, None)
    return (None, None, None)


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


# ---------- frame builder ----------


def _build_frames(
    camera_images: Dict[str, List[Path]],
    camera_ids: List[str],
    sample_index: int,
) -> List[InspectionFrame]:
    frames: List[InspectionFrame] = []
    for cid in camera_ids:
        path = camera_images[cid][sample_index]
        image = cv2.imread(str(path))
        if image is None:
            frames.append(
                InspectionFrame(
                    camera_id=cid,
                    image=None,
                    source=str(path),
                    source_kind="file_read_error",
                    frame_id=path.stem,
                    error_reason=f"Failed to read image: {path}",
                )
            )
        else:
            frames.append(
                InspectionFrame(
                    camera_id=cid,
                    image=image,
                    source=str(path),
                    source_kind="external_image",
                    frame_id=path.stem,
                )
            )
    return frames


# ---------- metric computation ----------


def _compute_confusion(records: List[Dict[str, Any]]) -> Dict[str, int]:
    tp = tn = fp = fn = 0
    for r in records:
        gt = r.get("ground_truth_label")
        if gt is None:
            continue
        predicted_ng = r["status"] != "OK"
        if gt == "NG" and predicted_ng:
            tp += 1
        elif gt == "NG" and not predicted_ng:
            fn += 1
        elif gt == "OK" and not predicted_ng:
            tn += 1
        elif gt == "OK" and predicted_ng:
            fp += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _compute_binary_metrics(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def _wilson_ci(success: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    if trials == 0:
        return (0.0, 0.0)
    p = success / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    centre = (p + z2 / (2 * trials)) / denominator
    margin = (
        z * math.sqrt((p * (1 - p) + z2 / (4 * trials)) / trials) / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _compute_cis(
    tp: int, tn: int, fp: int, fn: int
) -> Dict[str, Tuple[float, float]]:
    return {
        "precision": _wilson_ci(tp, tp + fp),
        "recall": _wilson_ci(tp, tp + fn),
        "accuracy": _wilson_ci(tp + tn, tp + tn + fp + fn),
        "fpr": _wilson_ci(fp, fp + tn),
    }


def _compute_per_camera(
    records: List[Dict[str, Any]], camera_ids: List[str]
) -> Dict[str, Dict[str, Any]]:
    per_cam: Dict[str, Dict[str, int]] = {
        cid: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for cid in camera_ids
    }

    for r in records:
        gt = r.get("ground_truth_label")
        if gt is None:
            continue
        for cam in r.get("camera_results", []):
            cid = cam["camera_id"]
            if cid not in per_cam:
                continue
            cam_ng = cam["status"] != "OK"
            if gt == "NG" and cam_ng:
                per_cam[cid]["tp"] += 1
            elif gt == "NG" and not cam_ng:
                per_cam[cid]["fn"] += 1
            elif gt == "OK" and not cam_ng:
                per_cam[cid]["tn"] += 1
            elif gt == "OK" and cam_ng:
                per_cam[cid]["fp"] += 1

    result: Dict[str, Dict[str, Any]] = {}
    for cid, cm in per_cam.items():
        metrics = _compute_binary_metrics(cm["tp"], cm["tn"], cm["fp"], cm["fn"])
        result[cid] = {**cm, **metrics}
    return result


def _compute_score_distribution(
    records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    ok_scores: List[float] = []
    ng_scores: List[float] = []

    for r in records:
        gt = r.get("ground_truth_label")
        if gt is None:
            continue
        for cam in r.get("camera_results", []):
            score = cam.get("anomaly_score")
            if score is None:
                continue
            if gt == "OK":
                ok_scores.append(score)
            elif gt == "NG":
                ng_scores.append(score)

    return {
        "ok_scores": _score_stats(ok_scores),
        "ng_scores": _score_stats(ng_scores),
    }


def _score_stats(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {"count": 0}
    arr = np.array(scores, dtype=np.float64)
    return {
        "count": len(scores),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(scores) >= 2 else 0.0,
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def _compute_defect_type_recall(
    records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    type_stats: Dict[str, Dict[str, int]] = {}
    for r in records:
        defect_type = r.get("ground_truth_defect_type")
        if not defect_type:
            continue
        if defect_type not in type_stats:
            type_stats[defect_type] = {"total": 0, "detected": 0}
        type_stats[defect_type]["total"] += 1
        if r["status"] == "NG":
            type_stats[defect_type]["detected"] += 1

    result: Dict[str, Dict[str, Any]] = {}
    for dtype, stats in type_stats.items():
        total = stats["total"]
        detected = stats["detected"]
        result[dtype] = {
            "total": total,
            "detected": detected,
            "recall": detected / total if total > 0 else 0.0,
        }
    return result


# ---------- export ----------


def _export_roc_pr_csv(
    records: List[Dict[str, Any]],
    output_dir: Path,
    round_name: str,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{round_name}_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_index",
            "part_id",
            "ground_truth_label",
            "camera_id",
            "anomaly_score",
            "is_anomaly",
            "threshold",
        ])
        for r in records:
            gt = r.get("ground_truth_label")
            for cam in r.get("camera_results", []):
                writer.writerow([
                    r["index"],
                    r["part_id"],
                    gt,
                    cam["camera_id"],
                    cam.get("anomaly_score", ""),
                    cam.get("is_anomaly", ""),
                    cam.get("anomaly_threshold", ""),
                ])
    return str(csv_path)


def _export_results_json(results: Dict[str, dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _default_serializer(obj):
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_default_serializer, ensure_ascii=False)
    print(f"\n[benchmark] Full results saved to: {output_path}")


def _export_benchmark_overlay(
    *,
    artifacts_dir: Path,
    round_name: str,
    part_id: str,
    cam_result: Any,
) -> Dict[str, str]:
    overlay_image = getattr(cam_result, "overlay_image", None)
    if overlay_image is None:
        return {}

    artifact_paths: Dict[str, str] = {}
    output_path = artifacts_dir / f"{round_name}_{part_id}_{cam_result.camera_id}_overlay.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay_image):
        raise OSError(f"Failed to write benchmark overlay: {output_path}")
    artifact_paths["overlay"] = str(output_path)
    return artifact_paths


# ---------- single round ----------


def _run_single_round(
    service: "InspectionService",
    cameras,
    round_dir: Path,
    camera_ids: List[str],
    export_curves_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = DEFAULT_BENCHMARK_ARTIFACTS_DIR,
) -> dict:
    camera_images = _collect_camera_images(round_dir, camera_ids)
    sample_count = len(next(iter(camera_images.values())))

    ground_truth = _load_ground_truth(round_dir)
    implicit_label = _infer_implicit_label(round_dir.name, ground_truth)
    gt_source = (
        "manifest"
        if ground_truth is not None
        else ("implicit" if implicit_label is not None else "none")
    )

    status_counter: Counter = Counter()
    records: List[Dict[str, Any]] = []

    for idx in range(sample_count):
        frames = _build_frames(camera_images, camera_ids, idx)

        part_id = f"{round_dir.name}_{idx:04d}"
        result = inspect_frames(service, frames, part_id=part_id)

        status_counter[result.status] += 1

        gt_label, gt_defect_type, gt_severity = _get_gt_for_sample(
            ground_truth, implicit_label, idx
        )

        cam_details: List[Dict[str, Any]] = []
        for cam_result in result.camera_results:
            cam_info: Dict[str, Any] = {
                "camera_id": cam_result.camera_id,
                "status": cam_result.status,
                "anomaly_score": None,
                "anomaly_threshold": None,
                "is_anomaly": None,
            }
            if cam_result.texture_result is not None:
                cam_info["anomaly_score"] = cam_result.texture_result.score
                cam_info["anomaly_threshold"] = cam_result.texture_result.threshold
                cam_info["is_anomaly"] = cam_result.texture_result.is_anomaly
            elif cam_result.region_results:
                region_scores = [
                    r.texture_result.score
                    for r in cam_result.region_results
                    if r.texture_result is not None
                ]
                if region_scores:
                    cam_info["anomaly_score"] = max(region_scores)
            if artifacts_dir is not None:
                artifact_paths = _export_benchmark_overlay(
                    artifacts_dir=artifacts_dir,
                    round_name=round_dir.name,
                    part_id=part_id,
                    cam_result=cam_result,
                )
                if artifact_paths:
                    cam_info["artifact_paths"] = artifact_paths
            cam_details.append(cam_info)

        marker = "✓" if result.status == "OK" else "✗"
        print(
            f"  [{idx + 1:04d}/{sample_count}] {marker} {result.status}"
            f"  part_id={part_id}"
        )

        records.append({
            "index": idx,
            "part_id": part_id,
            "status": result.status,
            "decision_reason": result.decision_reason,
            "ground_truth_label": gt_label,
            "ground_truth_defect_type": gt_defect_type,
            "ground_truth_severity": gt_severity,
            "camera_results": cam_details,
            "source_map": {cid: str(camera_images[cid][idx]) for cid in camera_ids},
        })

    stats: Dict[str, Any] = {
        "total": sample_count,
        "ok": status_counter.get("OK", 0),
        "ng": status_counter.get("NG", 0),
        "reject": status_counter.get("REJECT", 0),
        "records": records,
        "ground_truth_source": gt_source,
    }

    ok_rate = stats["ok"] / stats["total"] * 100 if stats["total"] else 0
    ng_rate = stats["ng"] / stats["total"] * 100 if stats["total"] else 0
    reject_rate = stats["reject"] / stats["total"] * 100 if stats["total"] else 0

    print(
        f"\n  Results: OK={stats['ok']} ({ok_rate:.1f}%), "
        f"NG={stats['ng']} ({ng_rate:.1f}%), "
        f"REJECT={stats['reject']} ({reject_rate:.1f}%)"
    )

    if gt_source != "none":
        cm = _compute_confusion(records)
        stats.update(cm)
        metrics = _compute_binary_metrics(cm["tp"], cm["tn"], cm["fp"], cm["fn"])
        stats.update(metrics)
        stats["confidence_intervals"] = _compute_cis(
            cm["tp"], cm["tn"], cm["fp"], cm["fn"]
        )
        stats["per_camera"] = _compute_per_camera(records, camera_ids)
        stats["anomaly_scores"] = _compute_score_distribution(records)
        defect_recall = _compute_defect_type_recall(records)
        if defect_recall:
            stats["defect_type_recall"] = defect_recall

    if export_curves_dir is not None and gt_source != "none":
        stats["roc_pr_curve_path"] = _export_roc_pr_csv(
            records, export_curves_dir, round_dir.name
        )

    _print_round_details(records)

    return stats


# ---------- printing ----------


def _print_round_details(records: List[Dict[str, Any]]) -> None:
    ng_records = [r for r in records if r["status"] in ("NG", "REJECT")]
    if not ng_records:
        return

    print(f"\n  NG/REJECT samples ({len(ng_records)}):")
    for r in ng_records:
        print(
            f"    [{r['index']:04d}] {r['status']}  part_id={r['part_id']}"
            f"  reason={r['decision_reason']}"
        )


def _print_summary(
    results: dict, camera_ids: List[str], rounds: Tuple[str, ...]
) -> None:
    print(f"\n{'=' * 60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Cameras: {', '.join(camera_ids)}")

    all_labeled: List[Dict[str, Any]] = []

    for round_name in rounds:
        r = results.get(round_name)
        if r is None:
            continue

        total = r["total"]
        raw_ok = r["ok"]
        raw_ng = r["ng"]
        raw_reject = r["reject"]
        gt_source = r.get("ground_truth_source", "none")

        if round_name == "good":
            label = "Good (all OK)"
        elif round_name == "defect":
            label = "Defect (all NG)"
        else:
            label = "Mixed"

        print(f"\n  [{label}]")
        print(f"    Samples: {total}")
        print(f"    OK: {raw_ok}  |  NG: {raw_ng}  |  REJECT: {raw_reject}")

        if gt_source != "none":
            _print_metrics_block(r, gt_source)
            all_labeled.extend(r["records"])
        else:
            _print_legacy_rates(round_name, r)

    if all_labeled:
        _print_combined_metrics(all_labeled, camera_ids)

    print(f"\n{'=' * 60}\n")


def _print_metrics_block(r: dict, gt_source: str) -> None:
    label_note = " (implicit)" if gt_source == "implicit" else ""
    print(f"\n    Ground truth{label_note}:")

    tp = r.get("tp", 0)
    tn = r.get("tn", 0)
    fp = r.get("fp", 0)
    fn = r.get("fn", 0)

    print(f"    TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    precision = r.get("precision", 0.0) * 100
    recall = r.get("recall", 0.0) * 100
    f1 = r.get("f1", 0.0) * 100
    accuracy = r.get("accuracy", 0.0) * 100

    cis = r.get("confidence_intervals", {})
    prec_ci = cis.get("precision", (0.0, 0.0))
    rec_ci = cis.get("recall", (0.0, 0.0))
    acc_ci = cis.get("accuracy", (0.0, 0.0))
    fpr_ci = cis.get("fpr", (0.0, 0.0))

    fpr_val = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0.0

    print(
        f"    Precision: {precision:.1f}%"
        f"  [95% CI: {prec_ci[0] * 100:.1f}–{prec_ci[1] * 100:.1f}%]"
    )
    print(
        f"    Recall:    {recall:.1f}%"
        f"  [95% CI: {rec_ci[0] * 100:.1f}–{rec_ci[1] * 100:.1f}%]"
    )
    print(f"    F1 Score:  {f1:.1f}%")
    print(
        f"    Accuracy:  {accuracy:.1f}%"
        f"  [95% CI: {acc_ci[0] * 100:.1f}–{acc_ci[1] * 100:.1f}%]"
    )
    print(
        f"    FPR:       {fpr_val:.1f}%"
        f"  [95% CI: {fpr_ci[0] * 100:.1f}–{fpr_ci[1] * 100:.1f}%]"
    )

    per_camera = r.get("per_camera", {})
    if per_camera:
        print(f"\n    Per-camera metrics:")
        print(
            f"    {'Camera':<12} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}"
            f" {'Prec':>7} {'Rec':>7} {'F1':>7}"
        )
        for cid in sorted(per_camera.keys()):
            pc = per_camera[cid]
            print(
                f"    {cid:<12} {pc['tp']:>5} {pc['tn']:>5} {pc['fp']:>5}"
                f" {pc['fn']:>5} {pc['precision'] * 100:>6.1f}%"
                f" {pc['recall'] * 100:>6.1f}% {pc['f1'] * 100:>6.1f}%"
            )

    scores = r.get("anomaly_scores", {})
    _print_score_block("OK", scores.get("ok_scores", {}))
    _print_score_block("NG", scores.get("ng_scores", {}))

    defect_recall = r.get("defect_type_recall", {})
    if defect_recall:
        print(f"\n    Defect-type recall:")
        for dtype in sorted(defect_recall.keys()):
            dr = defect_recall[dtype]
            print(
                f"      {dtype:<15} {dr['detected']}/{dr['total']}"
                f"  recall={dr['recall'] * 100:.1f}%"
            )

    curve_path = r.get("roc_pr_curve_path")
    if curve_path:
        print(f"\n    ROC/PR data exported to: {curve_path}")


def _print_score_block(label: str, stats: dict) -> None:
    count = stats.get("count", 0)
    if count == 0:
        return
    print(
        f"\n    [{label}] anomaly scores (n={count}):"
        f"  min={stats['min']:.4f}  max={stats['max']:.4f}"
        f"  mean={stats['mean']:.4f}  median={stats['median']:.4f}"
        f"  std={stats['std']:.4f}  p5={stats['p5']:.4f}  p95={stats['p95']:.4f}"
    )


def _print_legacy_rates(round_name: str, r: dict) -> None:
    total = r["total"]
    if total == 0:
        return
    if round_name == "good":
        fp_rate = (r["ng"] + r["reject"]) / total * 100
        print(f"    False positive rate: {fp_rate:.1f}%")
    elif round_name == "defect":
        miss_rate = r["ok"] / total * 100
        detection_rate = r["ng"] / total * 100
        print(
            f"    Miss rate: {miss_rate:.1f}%  |  Detection rate: {detection_rate:.1f}%"
        )
    else:
        print(
            f"    OK rate: {r['ok'] / total * 100:.1f}%"
            f"  |  NG rate: {r['ng'] / total * 100:.1f}%"
        )


def _print_combined_metrics(
    all_labeled: List[Dict[str, Any]], camera_ids: List[str]
) -> None:
    cm = _compute_confusion(all_labeled)
    metrics = _compute_binary_metrics(cm["tp"], cm["tn"], cm["fp"], cm["fn"])
    cis = _compute_cis(cm["tp"], cm["tn"], cm["fp"], cm["fn"])

    precision = metrics["precision"] * 100
    recall = metrics["recall"] * 100
    f1 = metrics["f1"] * 100
    accuracy = metrics["accuracy"] * 100
    fpr_val = (
        cm["fp"] / (cm["fp"] + cm["tn"]) * 100
        if (cm["fp"] + cm["tn"]) > 0
        else 0.0
    )

    prec_ci = cis["precision"]
    rec_ci = cis["recall"]
    acc_ci = cis["accuracy"]
    fpr_ci = cis["fpr"]

    print(f"\n  [Combined Metrics (all labeled rounds)]")
    print(f"    TP={cm['tp']}  TN={cm['tn']}  FP={cm['fp']}  FN={cm['fn']}")
    print(
        f"    Precision (精准率): {precision:.1f}%"
        f"  [95% CI: {prec_ci[0] * 100:.1f}–{prec_ci[1] * 100:.1f}%]"
    )
    print(
        f"    Recall    (召回率): {recall:.1f}%"
        f"  [95% CI: {rec_ci[0] * 100:.1f}–{rec_ci[1] * 100:.1f}%]"
    )
    print(f"    F1 Score  (F1 值):  {f1:.1f}%")
    print(
        f"    Accuracy  (准确率): {accuracy:.1f}%"
        f"  [95% CI: {acc_ci[0] * 100:.1f}–{acc_ci[1] * 100:.1f}%]"
    )
    print(
        f"    FPR       (误报率): {fpr_val:.1f}%"
        f"  [95% CI: {fpr_ci[0] * 100:.1f}–{fpr_ci[1] * 100:.1f}%]"
    )

    per_camera = _compute_per_camera(all_labeled, camera_ids)
    if per_camera:
        print(f"\n    Per-camera (all rounds):")
        print(
            f"    {'Camera':<12} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}"
            f" {'Prec':>7} {'Rec':>7} {'F1':>7}"
        )
        for cid in sorted(per_camera.keys()):
            pc = per_camera[cid]
            print(
                f"    {cid:<12} {pc['tp']:>5} {pc['tn']:>5} {pc['fp']:>5}"
                f" {pc['fn']:>5} {pc['precision'] * 100:>6.1f}%"
                f" {pc['recall'] * 100:>6.1f}% {pc['f1'] * 100:>6.1f}%"
            )

    scores = _compute_score_distribution(all_labeled)
    _print_score_block("OK", scores.get("ok_scores", {}))
    _print_score_block("NG", scores.get("ng_scores", {}))

    defect_recall = _compute_defect_type_recall(all_labeled)
    if defect_recall:
        print(f"\n    Defect-type recall (all rounds):")
        for dtype in sorted(defect_recall.keys()):
            dr = defect_recall[dtype]
            print(
                f"      {dtype:<15} {dr['detected']}/{dr['total']}"
                f"  recall={dr['recall'] * 100:.1f}%"
            )
