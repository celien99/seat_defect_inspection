"""Metric computation for benchmark evaluation."""

from __future__ import annotations

from typing import Dict, List

from .schemas import (
    BenchmarkRecord,
    BinaryMetrics,
    ConfusionMatrix,
    PerCameraMetrics,
)


def compute_confusion_matrix(records: List[BenchmarkRecord]) -> ConfusionMatrix:
    """Compute confusion matrix from fused (multi-camera) predictions.

    REJECT is treated as not-OK for safety: a REJECT means the system
    could not verify the part is good, so it counts as a positive call.
    """
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


def compute_binary_metrics(cm: ConfusionMatrix) -> BinaryMetrics:
    precision = cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) > 0 else 0.0
    recall = cm.tp / (cm.tp + cm.fn) if (cm.tp + cm.fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (cm.tp + cm.tn) / cm.total if cm.total > 0 else 0.0
    miss_rate = cm.fn / (cm.tp + cm.fn) if (cm.tp + cm.fn) > 0 else 0.0
    false_alarm_rate = cm.fp / (cm.fp + cm.tn) if (cm.fp + cm.tn) > 0 else 0.0
    return BinaryMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        miss_rate=miss_rate,
        false_alarm_rate=false_alarm_rate,
    )


def compute_per_camera_metrics(
    records: List[BenchmarkRecord],
    camera_ids: List[str],
) -> List[PerCameraMetrics]:
    """Compute per-camera confusion matrix and binary metrics.

    Uses ``sample.camera_ground_truth`` when available, otherwise falls
    back to the sample's overall ``ground_truth_label``.
    """
    per_cam: Dict[str, Dict[str, int]] = {
        cid: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for cid in camera_ids
    }

    for r in records:
        for cam in r.camera_records:
            cid = cam.camera_id
            if cid not in per_cam:
                continue

            # Prefer per-camera ground truth, fall back to overall label
            gt = r.sample.camera_ground_truth.get(cid)
            if gt is None:
                gt = r.sample.ground_truth_label
            if gt is None:
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

    result: List[PerCameraMetrics] = []
    for cid in sorted(per_cam.keys()):
        cm = ConfusionMatrix(**per_cam[cid])
        bm = compute_binary_metrics(cm) if cm.total > 0 else None
        if bm:
            result.append(PerCameraMetrics(
                camera_id=cid,
                confusion=cm,
                precision=bm.precision,
                recall=bm.recall,
                f1=bm.f1,
                accuracy=bm.accuracy,
                miss_rate=bm.miss_rate,
                false_alarm_rate=bm.false_alarm_rate,
            ))
        else:
            result.append(PerCameraMetrics(camera_id=cid, confusion=cm))
    return result


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
