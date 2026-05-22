"""Markdown report generation for benchmark evaluation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

from .config import BenchmarkConfig
from .schemas import BenchmarkSummary


def generate_md_report(
    summary: BenchmarkSummary,
    config: BenchmarkConfig,
    output_path: str,
) -> str:
    lines: List[str] = []
    a = lines.append

    a("# Seat Defect Inspection — Benchmark Report")
    a("")
    a(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"**Config**: `{config.config_path or 'N/A'}`")
    a(f"**Cameras**: {', '.join(summary.camera_ids)}")
    a(f"**Rounds**: {', '.join(r.round_name for r in summary.rounds)}")
    a("")

    # Combined metrics
    if summary.combined_metrics:
        bm = summary.combined_metrics
        a("## Combined Metrics")
        a("")
        a("| Metric | Value |")
        a("|--------|-------|")
        a(f"| Precision (精准率) | {bm.precision * 100:.1f}% |")
        a(f"| Recall (召回率) | {bm.recall * 100:.1f}% |")
        a(f"| F1 Score | {bm.f1 * 100:.1f}% |")
        a(f"| Accuracy (准确率) | {bm.accuracy * 100:.1f}% |")
        a(f"| Miss Rate (漏检率) | {bm.miss_rate * 100:.1f}% |")
        a(f"| False Alarm (错检率) | {bm.false_alarm_rate * 100:.1f}% |")
        a("")

    # Per-round results
    for rd in summary.rounds:
        a(f"## Round: {rd.round_name}")
        a("")
        a("| | Count |")
        a("|--------|-------|")
        a(f"| Total Samples | {rd.sample_count} |")
        a(f"| OK (predicted) | {rd.ok_count} |")
        a(f"| NG (predicted) | {rd.ng_count} |")
        a(f"| REJECT | {rd.reject_count} |")
        a("")

        if rd.confusion and rd.confusion.total > 0:
            cm = rd.confusion
            a("### Confusion Matrix (fused)")
            a("")
            a("| | Predicted OK | Predicted NG/REJECT |")
            a("|--------|--------------|---------------------|")
            a(f"| **Actual OK** | TN={cm.tn} | FP={cm.fp} |")
            a(f"| **Actual NG** | FN={cm.fn} | TP={cm.tp} |")
            a("")

        if rd.binary_metrics:
            bm = rd.binary_metrics
            a("### Metrics (fused)")
            a("")
            a("| Metric | Value |")
            a("|--------|-------|")
            a(f"| Precision | {bm.precision * 100:.1f}% |")
            a(f"| Recall | {bm.recall * 100:.1f}% |")
            a(f"| F1 | {bm.f1 * 100:.1f}% |")
            a(f"| Accuracy | {bm.accuracy * 100:.1f}% |")
            a(f"| Miss Rate (漏检率) | {bm.miss_rate * 100:.1f}% |")
            a(f"| False Alarm (错检率) | {bm.false_alarm_rate * 100:.1f}% |")
            a("")

        # Per-camera metrics
        if rd.per_camera:
            a("### Per-Camera Metrics")
            a("")
            a("| Camera | TP | TN | FP | FN | Precision | Recall | F1 | Accuracy | Miss Rate | False Alarm |")
            a("|--------|----|----|----|----|-----------|--------|----|----------|-----------|-------------|")
            for pc in rd.per_camera:
                cm = pc.confusion
                a(
                    f"| {pc.camera_id} | {cm.tp} | {cm.tn} | {cm.fp} | {cm.fn} |"
                    f" {pc.precision * 100:.1f}% | {pc.recall * 100:.1f}% |"
                    f" {pc.f1 * 100:.1f}% | {pc.accuracy * 100:.1f}% |"
                    f" {pc.miss_rate * 100:.1f}% | {pc.false_alarm_rate * 100:.1f}% |"
                )
            a("")

        # Failure cases
        if rd.failure_cases:
            a("### Failure Cases")
            a("")
            a("| Part ID | GT | Predicted | Defect Type | Decision Reason |")
            a("|---------|----|-----------|-------------|-----------------|")
            for fc in rd.failure_cases:
                gt = fc.sample.ground_truth_label or "?"
                dt = fc.sample.ground_truth_defect_type or "N/A"
                reason = fc.decision_reason[:80]
                a(f"| {fc.sample.part_id} | {gt} | {fc.predicted_status} | {dt} | {reason} |")
            a("")

        # Overlay image paths
        overlay_paths = [
            c.overlay_path
            for r in rd.records
            for c in r.camera_records
            if c.overlay_path
        ]
        if overlay_paths:
            a("### Result Images")
            a("")
            for p in overlay_paths:
                a(f"- `{p}`")
            a("")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")
    print(f"\n[benchmark] Markdown report saved to: {output_path}")
    return str(output_path)
