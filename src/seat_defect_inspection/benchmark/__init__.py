"""Benchmark evaluation platform for the seat defect inspection pipeline.

Provides dataset loading, inference execution, metric computation,
chart generation, and report output (HTML / PPTX / JSON).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import BenchmarkConfig
from .schemas import BenchmarkSummary

if TYPE_CHECKING:
    from ..service.core import InspectionService


def run_benchmark(
    service: "InspectionService",
    config: BenchmarkConfig,
) -> BenchmarkSummary:
    """Run full benchmark evaluation and generate reports.

    Parameters
    ----------
    service:
        Initialized InspectionService with loaded config.
    config:
        Benchmark configuration (data dir, rounds, report settings).

    Returns
    -------
    BenchmarkSummary with all round results and combined metrics.
    """
    from .data import discover_benchmark_samples
    from .metrics import compute_all_metrics, compute_combined_metrics
    from .pipeline import run_round

    rounds_data = discover_benchmark_samples(config)
    if not rounds_data:
        raise RuntimeError(
            "No benchmark rounds found. Check that --data-dir contains "
            "subdirectories named 'good', 'defect', or 'mixed' with per-camera image folders."
        )

    from ..service.core import InspectionService
    # Resolve camera state (enable only benchmarked cameras)
    context = service.resolve_context(config.seat_model_id)
    camera_ids = config.camera_ids or [c.camera_id for c in context.cameras]
    original_enabled = {c.camera_id: c.enabled for c in context.cameras}
    original_sources = {c.camera_id: c.source for c in context.cameras}

    # Filter cameras to only benchmarked ones
    for c in context.cameras:
        if c.camera_id not in camera_ids:
            c.enabled = False

    summary_rounds = []
    all_labeled_records = []
    try:
        for round_name, (samples, composition) in rounds_data.items():
            print(f"\n{'=' * 60}")
            print(f"  Benchmark round: {round_name}")
            print(f"{'=' * 60}")

            records = run_round(service, config, round_name, samples)
            round_result = compute_all_metrics(records, config, camera_ids)
            round_result.round_name = round_name
            round_result.composition = composition
            summary_rounds.append(round_result)

            _print_round_summary(round_result)
            all_labeled_records.extend([
                r for r in records if r.sample.ground_truth_label is not None
            ])

        combined_metrics, combined_per_camera, combined_defect = compute_combined_metrics(
            all_labeled_records, camera_ids
        )

        summary = BenchmarkSummary(
            rounds=summary_rounds,
            combined_metrics=combined_metrics,
            dataset_overview=[r.composition for r in summary_rounds],
        )

        _print_final_summary(summary, camera_ids)

        # Generate reports
        if config.output_json:
            from .report_html import export_results_json
            export_results_json(summary, config.output_json)

        if config.curve_csv_dir:
            from .report_html import _export_scores_csv
            _export_scores_csv(summary, config.curve_csv_dir)

        if config.report_format == "html" and config.report_output:
            from .report_html import generate_html_report
            generate_html_report(summary, config, config.report_output)

        if config.report_format == "pptx" and config.report_output:
            from .report_pptx import generate_pptx_report
            generate_pptx_report(summary, config, config.report_output)

        return summary
    finally:
        # Restore original camera state
        for c in context.cameras:
            c.enabled = original_enabled.get(c.camera_id, c.enabled)
            c.source = original_sources.get(c.camera_id, c.source)


def _print_round_summary(round_result) -> None:
    comp = round_result.composition
    print(f"\n  Results: Total={comp.sample_count}")
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
        if round_result.roc_curve and round_result.roc_curve.auc > 0:
            print(f"  ROC AUC: {round_result.roc_curve.auc:.4f}  PR AUC: {round_result.pr_curve.auc:.4f}")
        if round_result.timing:
            t = round_result.timing
            print(f"  Timing: mean={t.mean_ms:.0f}ms  p50={t.p50_ms:.0f}ms  p95={t.p95_ms:.0f}ms")


def _print_final_summary(summary, camera_ids) -> None:
    print(f"\n{'=' * 60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Cameras: {', '.join(camera_ids)}")
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
