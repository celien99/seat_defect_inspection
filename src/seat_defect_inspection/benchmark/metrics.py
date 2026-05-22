"""Metric computation for benchmark evaluation."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import BenchmarkConfig
from .schemas import (
    BenchmarkRecord,
    BinaryMetrics,
    ConfusionMatrix,
    CurveResult,
    DefectTypeMetrics,
    PerCameraMetrics,
    RoundResult,
    ScoreDistribution,
    ThresholdSweepPoint,
    TimingStats,
)


# ---------- confusion matrix ----------


def compute_confusion_matrix(records: List[BenchmarkRecord]) -> ConfusionMatrix:
    cm = ConfusionMatrix()
    for r in records:
        gt = r.sample.ground_truth_label
        if gt is None:
            continue
        predicted_ng = r.predicted_status != "OK"
        if gt == "NG" and predicted_ng:
            cm.tp += 1
        elif gt == "NG" and not predicted_ng:
            cm.fn += 1
        elif gt == "OK" and not predicted_ng:
            cm.tn += 1
        elif gt == "OK" and predicted_ng:
            cm.fp += 1
    return cm


def compute_confusion_per_camera(
    records: List[BenchmarkRecord],
) -> Dict[str, ConfusionMatrix]:
    per_cam: Dict[str, Dict[str, int]] = {}
    for r in records:
        gt = r.sample.ground_truth_label
        if gt is None:
            continue
        for cam in r.camera_records:
            cid = cam.camera_id
            if cid not in per_cam:
                per_cam[cid] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
            cam_ng = cam.predicted_status != "OK"
            if gt == "NG" and cam_ng:
                per_cam[cid]["tp"] += 1
            elif gt == "NG" and not cam_ng:
                per_cam[cid]["fn"] += 1
            elif gt == "OK" and not cam_ng:
                per_cam[cid]["tn"] += 1
            elif gt == "OK" and cam_ng:
                per_cam[cid]["fp"] += 1
    return {
        cid: ConfusionMatrix(**counts) for cid, counts in per_cam.items()
    }


# ---------- binary metrics ----------


def compute_binary_metrics(cm: ConfusionMatrix) -> BinaryMetrics:
    precision = cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) > 0 else 0.0
    recall = cm.tp / (cm.tp + cm.fn) if (cm.tp + cm.fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (cm.tp + cm.tn) / cm.total if cm.total > 0 else 0.0
    return BinaryMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        miss_rate=cm.miss_rate,
        false_alarm_rate=cm.false_alarm_rate,
        confidence_intervals=compute_wilson_cis(cm),
    )


def compute_wilson_cis(cm: ConfusionMatrix, z: float = 1.96) -> Dict[str, Tuple[float, float]]:
    return {
        "precision": _wilson_ci(cm.tp, cm.tp + cm.fp, z),
        "recall": _wilson_ci(cm.tp, cm.tp + cm.fn, z),
        "accuracy": _wilson_ci(cm.tp + cm.tn, cm.total, z),
        "fpr": _wilson_ci(cm.fp, cm.fp + cm.tn, z),
        "miss_rate": _wilson_ci(cm.fn, cm.tp + cm.fn, z),
    }


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


# ---------- per-camera metrics ----------


def compute_per_camera_metrics(
    records: List[BenchmarkRecord],
) -> List[PerCameraMetrics]:
    per_cam = compute_confusion_per_camera(records)
    result: List[PerCameraMetrics] = []
    for cid in sorted(per_cam.keys()):
        cm = per_cam[cid]
        precision = cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) > 0 else 0.0
        recall = cm.tp / (cm.tp + cm.fn) if (cm.tp + cm.fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (cm.tp + cm.tn) / cm.total if cm.total > 0 else 0.0
        result.append(PerCameraMetrics(
            camera_id=cid,
            confusion=cm,
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
        ))
    return result


def compute_per_camera_confusion_from_records(
    records: List[BenchmarkRecord],
    camera_ids: List[str],
) -> Dict[str, ConfusionMatrix]:
    per_cam: Dict[str, Dict[str, int]] = {cid: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for cid in camera_ids}
    for r in records:
        gt = r.sample.ground_truth_label
        if gt is None:
            continue
        for cam in r.camera_records:
            cid = cam.camera_id
            if cid not in per_cam:
                continue
            cam_ng = cam.predicted_status != "OK"
            if gt == "NG" and cam_ng:
                per_cam[cid]["tp"] += 1
            elif gt == "NG" and not cam_ng:
                per_cam[cid]["fn"] += 1
            elif gt == "OK" and not cam_ng:
                per_cam[cid]["tn"] += 1
            elif gt == "OK" and cam_ng:
                per_cam[cid]["fp"] += 1
    return {cid: ConfusionMatrix(**counts) for cid, counts in per_cam.items()}


# ---------- defect-type metrics ----------


def compute_defect_type_metrics(
    records: List[BenchmarkRecord],
    confusion: ConfusionMatrix,
) -> List[DefectTypeMetrics]:
    type_stats: Dict[str, Dict[str, int]] = {}
    for r in records:
        defect_type = r.sample.ground_truth_defect_type
        if not defect_type:
            continue
        if defect_type not in type_stats:
            type_stats[defect_type] = {"total": 0, "detected": 0, "fp": 0}
        type_stats[defect_type]["total"] += 1
        if r.predicted_status == "NG":
            type_stats[defect_type]["detected"] += 1

    result: List[DefectTypeMetrics] = []
    for dtype in sorted(type_stats.keys()):
        stats = type_stats[dtype]
        total = stats["total"]
        detected = stats["detected"]
        recall = detected / total if total > 0 else 0.0
        total_ng_pred = confusion.tp + confusion.fp
        precision = detected / total_ng_pred if total_ng_pred > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        result.append(DefectTypeMetrics(
            defect_type=dtype,
            total=total,
            detected=detected,
            recall=recall,
            precision=precision,
            f1=f1,
        ))
    return result


# ---------- score distribution ----------


def compute_score_distributions(
    records: List[BenchmarkRecord],
) -> List[ScoreDistribution]:
    ok_scores: List[float] = []
    ng_scores: List[float] = []
    for r in records:
        gt = r.sample.ground_truth_label
        if gt is None:
            continue
        for cam in r.camera_records:
            if cam.anomaly_score is None:
                continue
            if gt == "OK":
                ok_scores.append(cam.anomaly_score)
            elif gt == "NG":
                ng_scores.append(cam.anomaly_score)
    return [
        _build_score_distribution("OK", ok_scores),
        _build_score_distribution("NG", ng_scores),
    ]


def _build_score_distribution(label: str, scores: List[float]) -> ScoreDistribution:
    if not scores:
        return ScoreDistribution(label=label, count=0)
    arr = np.array(scores, dtype=np.float64)
    return ScoreDistribution(
        label=label,
        count=len(scores),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr, ddof=1)) if len(scores) >= 2 else 0.0,
        p5=float(np.percentile(arr, 5)),
        p95=float(np.percentile(arr, 95)),
        all_scores=list(scores),
    )


# ---------- timing ----------


def compute_timing_stats(records: List[BenchmarkRecord]) -> TimingStats:
    timings = [r.inference_timing_ms for r in records]
    if not timings:
        return TimingStats()
    arr = np.array(timings, dtype=np.float64)
    return TimingStats(
        mean_ms=float(np.mean(arr)),
        std_ms=float(np.std(arr, ddof=1)) if len(timings) >= 2 else 0.0,
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        all_timings_ms=list(timings),
    )


# ---------- threshold sweep (ROC / PR) ----------


def compute_threshold_sweep(
    records: List[BenchmarkRecord],
    config: BenchmarkConfig,
) -> Tuple[CurveResult, CurveResult]:
    pairs: List[Tuple[int, float]] = []
    for r in records:
        gt = r.sample.ground_truth_label
        if gt is None:
            continue
        true_label = 1 if gt == "NG" else 0
        for cam in r.camera_records:
            if cam.anomaly_score is not None:
                pairs.append((true_label, cam.anomaly_score))

    if not pairs:
        return CurveResult(), CurveResult()

    scores = sorted(set(p[1] for p in pairs))
    if len(scores) <= 2:
        return CurveResult(), CurveResult()

    # Use unique scores as threshold points for precise step-function capture
    thresholds: List[float] = [float(s) for s in scores]
    if len(thresholds) > config.sweep_steps:
        # Subsample if too many unique scores
        indices = np.linspace(0, len(thresholds) - 1, config.sweep_steps, dtype=int)
        thresholds = [thresholds[i] for i in indices]

    # Include boundary thresholds to capture (FPR=1,TPR=1) and (FPR=0,TPR=0) endpoints
    eps = (thresholds[-1] - thresholds[0]) * 0.01 if thresholds[-1] > thresholds[0] else 0.01
    thresholds = [thresholds[0] - eps] + thresholds + [thresholds[-1] + eps]

    roc_points: List[ThresholdSweepPoint] = []
    pr_points: List[ThresholdSweepPoint] = []
    for thr in thresholds:
        tp = fp = tn = fn = 0
        for true_label, score in pairs:
            pred = 1 if score >= thr else 0
            if true_label == 1 and pred == 1:
                tp += 1
            elif true_label == 1 and pred == 0:
                fn += 1
            elif true_label == 0 and pred == 0:
                tn += 1
            elif true_label == 0 and pred == 1:
                fp += 1
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        f1 = (
            2 * precision * tpr / (precision + tpr)
            if (precision + tpr) > 0
            else 0.0
        )
        roc_points.append(ThresholdSweepPoint(
            threshold=float(thr), tpr=tpr, fpr=fpr_val, precision=precision, f1=f1,
        ))
        pr_points.append(ThresholdSweepPoint(
            threshold=float(thr), tpr=tpr, fpr=fpr_val, precision=precision, f1=f1,
        ))

    roc_auc = _trapezoidal_auc([(p.fpr, p.tpr) for p in roc_points])
    pr_auc = _trapezoidal_auc([(p.tpr, p.precision) for p in pr_points])
    return (
        CurveResult(points=roc_points, auc=roc_auc),
        CurveResult(points=pr_points, auc=pr_auc),
    )


def _trapezoidal_auc(points: List[Tuple[float, float]]) -> float:
    """Compute AUC via trapezoidal rule on the step function defined by points.

    Sorts by x (FPR for ROC, Recall for PR), takes max y per unique x,
    ensures (0,0) and (1, max_y) endpoints, then integrates.
    """
    if not points:
        return 0.0

    # Group by x, take max y for each x (step function upper envelope)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    unique_xs = sorted(set(xs))

    # Ensure (0,0) start and (1, last_y) end
    x_y_map = {x: 0.0 for x in unique_xs}
    for x, y in points:
        if y > x_y_map.get(x, 0.0):
            x_y_map[x] = y

    if 0.0 not in x_y_map:
        x_y_map[0.0] = 0.0

    sorted_xs = sorted(x_y_map.keys())
    if sorted_xs[-1] < 1.0 and x_y_map[sorted_xs[-1]] > 0:
        x_y_map[1.0] = x_y_map[sorted_xs[-1]]

    sorted_xs = sorted(x_y_map.keys())

    auc = 0.0
    for i in range(1, len(sorted_xs)):
        x0, x1 = sorted_xs[i - 1], sorted_xs[i]
        y0, y1 = x_y_map[x0], x_y_map[x1]
        auc += (x1 - x0) * (y0 + y1) / 2.0
    return auc


# ---------- failure cases ----------


def identify_failure_cases(
    records: List[BenchmarkRecord],
) -> List[BenchmarkRecord]:
    cases: List[BenchmarkRecord] = []
    for r in records:
        gt = r.sample.ground_truth_label
        if gt is None:
            continue
        if gt == "NG" and r.predicted_status == "OK":
            cases.append(r)
        elif gt == "OK" and r.predicted_status == "NG":
            cases.append(r)
    return cases


# ---------- all-in-one ----------


def compute_all_metrics(
    records: List[BenchmarkRecord],
    config: BenchmarkConfig,
    camera_ids: List[str],
) -> RoundResult:
    """Compute all metrics for a single round."""
    cm = compute_confusion_matrix(records)
    metrics = compute_binary_metrics(cm) if cm.total > 0 else None
    per_camera = compute_per_camera_metrics(records)
    defect_breakdown = compute_defect_type_metrics(records, cm)
    score_dists = compute_score_distributions(records)
    timing = compute_timing_stats(records)
    failure_cases = identify_failure_cases(records)

    roc_curve = None
    pr_curve = None
    if config.enable_threshold_sweep and cm.total > 0:
        roc_curve, pr_curve = compute_threshold_sweep(records, config)

    return RoundResult(
        confusion=cm if cm.total > 0 else None,
        binary_metrics=metrics,
        per_camera=per_camera,
        defect_type_breakdown=defect_breakdown,
        score_distributions=score_dists,
        timing=timing,
        roc_curve=roc_curve,
        pr_curve=pr_curve,
        records=records,
        failure_cases=failure_cases,
    )


def compute_combined_metrics(
    all_records: List[BenchmarkRecord],
    camera_ids: List[str],
) -> Tuple[Optional[BinaryMetrics], List[PerCameraMetrics], List[DefectTypeMetrics]]:
    """Compute combined metrics across all rounds."""
    if not all_records:
        return None, [], []
    cm = compute_confusion_matrix(all_records)
    if cm.total == 0:
        return None, [], []
    binary = compute_binary_metrics(cm)
    per_camera = compute_per_camera_metrics(all_records)
    defect_bd = compute_defect_type_metrics(all_records, cm)
    return binary, per_camera, defect_bd
