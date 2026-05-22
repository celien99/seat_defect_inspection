"""Benchmark evaluation for the seat defect inspection pipeline.

Provides dataset loading, inference execution, metric computation,
and markdown report generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import BenchmarkConfig
from .schemas import BenchmarkSummary, RoundResult

if TYPE_CHECKING:
    from ..service.core import InspectionService


def run_benchmark(
    service: "InspectionService",
    config: BenchmarkConfig,
) -> BenchmarkSummary:
    """Run full benchmark evaluation and generate reports."""
    from .data import discover_benchmark_samples
    from .metrics import (
        compute_binary_metrics,
        compute_confusion_matrix,
        compute_per_camera_metrics,
        identify_failure_cases,
    )
    from .pipeline import run_round

    rounds_data = discover_benchmark_samples(config)
    if not rounds_data:
        raise RuntimeError(
            "No benchmark rounds found. Check that --data-dir contains "
            "subdirectories named 'good', 'defect', or 'mixed' with per-camera image folders."
        )

    context = service.resolve_context(config.seat_model_id)
    camera_ids = config.camera_ids or [c.camera_id for c in context.cameras]
    original_enabled = {c.camera_id: c.enabled for c in context.cameras}
    original_sources = {c.camera_id: c.source for c in context.cameras}

    for c in context.cameras:
        if c.camera_id not in camera_ids:
            c.enabled = False

    # Save and enable debug artifacts for overlay image generation
    original_debug_enabled = getattr(service.config, "debug_artifacts_enabled", True)
    original_debug_names = list(getattr(service.config, "debug_artifact_names", ["overlay"]))
    _ensure_debug_artifacts(service.config)

    summary_rounds = []
    all_labeled_records = []
    try:
        for round_name, samples in rounds_data.items():
            print(f"\n{'=' * 60}")
            print(f"  Benchmark round: {round_name}")
            print(f"{'=' * 60}")

            records = run_round(service, config, round_name, samples)
            cm = compute_confusion_matrix(records)
            bm = compute_binary_metrics(cm) if cm.total > 0 else None
            per_cam = compute_per_camera_metrics(records, camera_ids)
            failures = identify_failure_cases(records)
            ok_count = sum(1 for r in records if r.predicted_status == "OK")
            ng_count = sum(1 for r in records if r.predicted_status == "NG")
            reject_count = sum(1 for r in records if r.predicted_status == "REJECT")

            round_result = RoundResult(
                round_name=round_name,
                sample_count=len(samples),
                ok_count=ok_count,
                ng_count=ng_count,
                reject_count=reject_count,
                confusion=cm if cm.total > 0 else None,
                binary_metrics=bm,
                per_camera=per_cam,
                records=records,
                failure_cases=failures,
            )
            summary_rounds.append(round_result)

            _print_round_summary(round_result)
            all_labeled_records.extend([
                r for r in records if r.sample.ground_truth_label is not None
            ])

        combined_metrics = None
        if all_labeled_records:
            combined_cm = compute_confusion_matrix(all_labeled_records)
            if combined_cm.total > 0:
                combined_metrics = compute_binary_metrics(combined_cm)

        summary = BenchmarkSummary(
            rounds=summary_rounds,
            camera_ids=list(camera_ids),
            combined_metrics=combined_metrics,
        )

        _print_final_summary(summary)

        if config.report_output:
            from .report_md import generate_md_report
            generate_md_report(summary, config, config.report_output)

        return summary
    finally:
        for c in context.cameras:
            c.enabled = original_enabled.get(c.camera_id, c.enabled)
            c.source = original_sources.get(c.camera_id, c.source)
        service.config.debug_artifacts_enabled = original_debug_enabled
        service.config.debug_artifact_names = original_debug_names


def _ensure_debug_artifacts(inspection_config) -> None:
    """Enable overlay generation if not already enabled."""
    if not getattr(inspection_config, "debug_artifacts_enabled", True):
        inspection_config.debug_artifacts_enabled = True
    names = getattr(inspection_config, "debug_artifact_names", None)
    if names is not None and "overlay" not in names:
        names.append("overlay")


def _print_round_summary(round_result: RoundResult) -> None:
    print(f"\n  Results: Total={round_result.sample_count}")
    print(f"  OK={round_result.ok_count}  NG={round_result.ng_count}  REJECT={round_result.reject_count}")
    if round_result.binary_metrics:
        bm = round_result.binary_metrics
        cm = round_result.confusion
        print(f"  TP={cm.tp}  TN={cm.tn}  FP={cm.fp}  FN={cm.fn}")
        print(
            f"  Precision: {bm.precision * 100:.1f}%  "
            f"Recall: {bm.recall * 100:.1f}%  "
            f"F1: {bm.f1 * 100:.1f}%  "
            f"Accuracy: {bm.accuracy * 100:.1f}%"
        )
        print(
            f"  Miss Rate (漏检率): {bm.miss_rate * 100:.1f}%  "
            f"False Alarm (错检率): {bm.false_alarm_rate * 100:.1f}%"
        )
    if round_result.per_camera:
        print(f"\n  Per-camera metrics:")
        print(f"  {'Camera':<12} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}")
        for pc in round_result.per_camera:
            cm = pc.confusion
            print(
                f"  {pc.camera_id:<12} {cm.tp:>5} {cm.tn:>5} {cm.fp:>5} {cm.fn:>5}"
                f" {pc.precision * 100:>6.1f}% {pc.recall * 100:>6.1f}% {pc.f1 * 100:>6.1f}%"
            )


def _print_final_summary(summary: BenchmarkSummary) -> None:
    print(f"\n{'=' * 60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Cameras: {', '.join(summary.camera_ids)}")
    print(f"  Rounds: {', '.join(r.round_name for r in summary.rounds)}")

    if summary.combined_metrics:
        bm = summary.combined_metrics
        print(f"  Precision (精准率): {bm.precision * 100:.1f}%")
        print(f"  Recall    (召回率): {bm.recall * 100:.1f}%")
        print(f"  F1 Score  (F1 值):  {bm.f1 * 100:.1f}%")
        print(f"  Accuracy  (准确率): {bm.accuracy * 100:.1f}%")
        print(f"  Miss Rate (漏检率): {bm.miss_rate * 100:.1f}%")
        print(f"  False Alarm (错检率): {bm.false_alarm_rate * 100:.1f}%")
    print(f"{'=' * 60}\n")
