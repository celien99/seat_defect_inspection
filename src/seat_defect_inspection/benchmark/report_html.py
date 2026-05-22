"""HTML report generation for benchmark evaluation."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Template

from .config import BenchmarkConfig
from .metrics import (
    compute_binary_metrics,
    compute_combined_metrics,
    compute_confusion_matrix,
    compute_score_distributions,
    compute_timing_stats,
)
from .plots import (
    plot_confusion_matrix,
    plot_defect_type_metrics,
    plot_per_camera_metrics,
    plot_pr_curve,
    plot_roc_curve,
    plot_score_distribution,
)
from .schemas import (
    BenchmarkRecord,
    BenchmarkSummary,
    DatasetComposition,
    RoundResult,
)

_TEMPLATE_PATH = str(
    Path(__file__).resolve().parent / "templates" / "benchmark_report.html.j2"
)


def generate_html_report(
    summary: BenchmarkSummary,
    config: BenchmarkConfig,
    output_path: str,
) -> str:
    template = Template(open(_TEMPLATE_PATH, "r", encoding="utf-8").read())

    all_records = [r for rd in summary.rounds for r in rd.records]
    all_labeled = [r for r in all_records if r.sample.ground_truth_label is not None]

    # Meta
    meta = {
        "report_title": "Seat Defect Inspection — Benchmark Report",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "camera_ids": [],
        "config_path": config.config_path or "N/A",
        "round_names": [r.round_name for r in summary.rounds],
        "fusion_strategy": "N/A",
        "threshold_sweep_enabled": config.enable_threshold_sweep,
    }

    # Round data
    round_data: Dict[str, Any] = {}
    round_metrics: Dict[str, Any] = {}
    cm_charts: Dict[str, str] = {}
    for rd in summary.rounds:
        round_data[rd.round_name] = rd
        if rd.binary_metrics:
            round_metrics[rd.round_name] = rd.binary_metrics
        if rd.confusion and rd.confusion.total > 0:
            cm_charts[rd.round_name] = plot_confusion_matrix(
                rd.confusion,
                title=f"Confusion Matrix — {rd.round_name}",
            )

    # KPI cards from combined metrics
    combined_metrics = summary.combined_metrics
    kpi_cards = []
    if combined_metrics:
        kpi_cards = [
            {"value": f"{combined_metrics.precision * 100:.1f}%", "label": "Precision (精准率)", "css_class": ""},
            {"value": f"{combined_metrics.recall * 100:.1f}%", "label": "Recall (召回率)", "css_class": ""},
            {"value": f"{combined_metrics.f1 * 100:.1f}%", "label": "F1 Score", "css_class": ""},
            {"value": f"{combined_metrics.accuracy * 100:.1f}%", "label": "Accuracy (准确率)", "css_class": ""},
            {"value": f"{combined_metrics.miss_rate * 100:.1f}%", "label": "Miss Rate (漏检率)", "css_class": "ng"},
            {"value": f"{combined_metrics.false_alarm_rate * 100:.1f}%", "label": "False Alarm (错检率)", "css_class": "ng"},
        ]
    meta["camera_ids"] = list(
        {c.camera_id for rd in summary.rounds for c in rd.per_camera}
    )

    # Per-camera
    per_camera_metrics: Dict[str, Any] = {}
    per_camera_confusions: Dict[str, Any] = {}
    for rd in summary.rounds:
        for pc in rd.per_camera:
            per_camera_metrics[pc.camera_id] = pc
            per_camera_confusions[pc.camera_id] = pc.confusion
    per_camera_chart = ""
    if per_camera_metrics:
        all_cam = list(per_camera_metrics.values())
        per_camera_chart = plot_per_camera_metrics(all_cam, title="Per-Camera Metrics")

    # Defect types
    defect_type_data: List[Any] = []
    seen = set()
    for rd in summary.rounds:
        for dt in rd.defect_type_breakdown:
            if dt.defect_type not in seen:
                seen.add(dt.defect_type)
                defect_type_data.append(dt)
    defect_type_chart = ""
    if defect_type_data:
        defect_type_chart = plot_defect_type_metrics(defect_type_data)

    # ROC / PR
    roc_chart = ""
    pr_chart = ""
    roc_auc = 0.0
    pr_auc = 0.0
    for rd in summary.rounds:
        if rd.roc_curve and rd.roc_curve.points:
            roc_chart = plot_roc_curve(rd.roc_curve, title=f"ROC Curve — {rd.round_name}")
            roc_auc = rd.roc_curve.auc
        if rd.pr_curve and rd.pr_curve.points:
            pr_chart = plot_pr_curve(rd.pr_curve, title=f"PR Curve — {rd.round_name}")
            pr_auc = rd.pr_curve.auc

    # Score distribution
    score_dists = compute_score_distributions(all_labeled)
    score_dist_chart = ""
    if score_dists:
        score_dist_chart = plot_score_distribution(score_dists)

    # Timing
    timing_data = compute_timing_stats(all_records) if all_records else None

    # Failure cases
    failure_cases = []
    for rd in summary.rounds:
        failure_cases.extend(rd.failure_cases)

    # Executive summary text
    total_samples = sum(c.sample_count for c in summary.dataset_overview)
    total_ng = sum(c.ng_count for c in summary.dataset_overview)
    exec_text = f"This benchmark evaluated the inspection pipeline on {total_samples} samples across {len(summary.rounds)} rounds."
    if combined_metrics:
        exec_text += (
            f" The combined miss rate (漏检率) is {combined_metrics.miss_rate * 100:.1f}% "
            f"and false alarm rate (错检率) is {combined_metrics.false_alarm_rate * 100:.1f}%."
        )
        exec_text += (
            f" Overall precision reaches {combined_metrics.precision * 100:.1f}% "
            f"with recall at {combined_metrics.recall * 100:.1f}%."
        )

    # Test method
    test_method = (
        "Standard inspection pipeline (YOLO → ROI → Quality → PatchCore → Fusion) "
        "applied to curated benchmark datasets with ground-truth annotations. "
        "Each sample is processed independently and evaluated against its known label. "
        "Metrics include confusion matrix, per-camera breakdown, per-defect-type analysis, "
        "score distribution, timing analysis, and threshold sensitivity curves."
    )

    # Recommendations
    recommendations: List[Dict[str, str]] = []
    if combined_metrics:
        if combined_metrics.miss_rate > 0.05:
            recommendations.append({
                "title": "High Miss Rate Alert",
                "body": f"Miss rate is {combined_metrics.miss_rate * 100:.1f}%, exceeding the 5% threshold. Consider reviewing PatchCore decision thresholds (decision_score_margin, strong_patch_count) or improving YOLO detection for missed samples.",
            })
        if combined_metrics.false_alarm_rate > 0.1:
            recommendations.append({
                "title": "High False Alarm Rate",
                "body": f"False alarm rate is {combined_metrics.false_alarm_rate * 100:.1f}%, exceeding the 10% threshold. Consider increasing the decision threshold or adjusting the fusion strategy.",
            })
        if combined_metrics.miss_rate <= 0.05 and combined_metrics.false_alarm_rate <= 0.1:
            recommendations.append({
                "title": "Good Pipeline Performance",
                "body": "Both miss rate and false alarm rate are within acceptable ranges. The inspection pipeline is performing well on the current benchmark dataset.",
            })

    # Compositions
    compositions = [rd.composition for rd in summary.rounds]

    context = {
        "meta": meta,
        "kpi_cards": kpi_cards,
        "executive_summary_text": exec_text,
        "test_method": test_method,
        "compositions": compositions,
        "round_data": round_data,
        "round_metrics": round_metrics,
        "cm_charts": cm_charts,
        "combined_metrics": combined_metrics,
        "per_camera_metrics": per_camera_metrics,
        "per_camera_confusions": per_camera_confusions,
        "per_camera_chart": per_camera_chart,
        "defect_type_data": defect_type_data,
        "defect_type_chart": defect_type_chart,
        "roc_chart": roc_chart,
        "pr_chart": pr_chart,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "score_dist_chart": score_dist_chart,
        "score_dists": score_dists,
        "timing_data": timing_data,
        "failure_cases": failure_cases,
        "recommendations": recommendations,
    }

    html = template.render(**context)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"\n[benchmark] HTML report saved to: {output_path}")
    return str(output_path)


def export_results_json(summary: BenchmarkSummary, output_path: str) -> str:
    """Export benchmark results as JSON (backward-compatible format)."""
    from datetime import datetime

    def _serialize(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _serialize(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, (Path,)):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    payload = {
        "rounds": [
            {
                "round_name": rd.round_name,
                "composition": _serialize(rd.composition),
                "confusion": _serialize(rd.confusion) if rd.confusion else None,
                "binary_metrics": _serialize(rd.binary_metrics) if rd.binary_metrics else None,
                "per_camera": [_serialize(pc) for pc in rd.per_camera],
            }
            for rd in summary.rounds
        ],
        "combined_metrics": _serialize(summary.combined_metrics) if summary.combined_metrics else None,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str, ensure_ascii=False)
    print(f"[benchmark] JSON results saved to: {output_path}")
    return str(output_path)


def _export_scores_csv(summary: BenchmarkSummary, output_dir: str) -> str:
    """Export per-sample anomaly scores as CSVs for external ROC/PR plotting."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for rd in summary.rounds:
        csv_path = output_dir / f"{rd.round_name}_scores.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sample_index", "part_id", "ground_truth_label",
                "camera_id", "anomaly_score", "is_anomaly", "threshold",
            ])
            for r in rd.records:
                for cam in r.camera_records:
                    writer.writerow([
                        r.sample.index,
                        r.sample.part_id,
                        r.sample.ground_truth_label,
                        cam.camera_id,
                        cam.anomaly_score if cam.anomaly_score is not None else "",
                        cam.is_anomaly if cam.is_anomaly is not None else "",
                        cam.anomaly_threshold if cam.anomaly_threshold is not None else "",
                    ])
        print(f"[benchmark] Score CSV exported to: {csv_path}")
    return str(output_dir)
