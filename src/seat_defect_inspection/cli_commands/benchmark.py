"""Benchmark command — evaluate inspection pipeline on standard datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..runtime_config import load_config
from .common import DEFAULT_CONFIG_PATH


def register_benchmark_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "benchmark",
        help="Evaluate inspection pipeline on good/defect/mixed datasets",
    )
    parser.set_defaults(run=run_benchmark_command)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Inspection config JSON/INI path",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Benchmark dataset root directory (default: ./benchmark_data)",
    )
    parser.add_argument(
        "--round",
        choices=["good", "defect", "mixed", "all"],
        default="all",
        help="Which round to run (default: all)",
    )
    parser.add_argument(
        "--cameras",
        help="Comma-separated camera IDs to benchmark (default: all enabled cameras)",
    )
    parser.add_argument(
        "--seat-model-id",
        help="Seat model ID for multi-model configs",
    )
    parser.add_argument(
        "--export-curves",
        type=Path,
        default=None,
        help="Output directory for ROC/PR score CSV exports",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save full benchmark results as JSON to this path",
    )
    parser.add_argument(
        "--report-format",
        choices=["html", "pptx", "json"],
        default="html",
        help="Report output format (default: html)",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="Report output file path (auto-generated if omitted)",
    )
    parser.add_argument(
        "--threshold-sweep",
        action="store_true",
        default=False,
        help="Enable threshold sweep for ROC/PR curve computation",
    )
    parser.add_argument(
        "--sweep-steps",
        type=int,
        default=50,
        help="Number of threshold sweep steps (default: 50)",
    )


def run_benchmark_command(args: argparse.Namespace) -> None:
    from ..benchmark import run_benchmark
    from ..benchmark.config import BenchmarkConfig
    from ..service.core import InspectionService

    config = load_config(args.config)
    service = InspectionService(config)
    rounds = ("good", "defect", "mixed") if args.round == "all" else (args.round,)
    camera_ids = [c.strip() for c in args.cameras.split(",")] if args.cameras else None

    data_dir = args.data_dir
    if data_dir is None:
        from pathlib import Path
        default = Path.cwd() / "benchmark_data"
        data_dir = str(default)

    report_output = args.report_output
    if report_output is None and args.report_format != "json":
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.report_format == "html":
            report_output = f"outputs/benchmark/benchmark_report_{ts}.html"
        elif args.report_format == "pptx":
            report_output = f"outputs/benchmark/benchmark_report_{ts}.pptx"

    benchmark_cfg = BenchmarkConfig(
        data_dir=data_dir,
        rounds=rounds,
        camera_ids=camera_ids,
        seat_model_id=args.seat_model_id,
        enable_threshold_sweep=args.threshold_sweep,
        sweep_steps=args.sweep_steps,
        report_format=args.report_format,
        report_output=str(report_output) if report_output else None,
        curve_csv_dir=str(args.export_curves) if args.export_curves else None,
        output_json=str(args.output) if args.output else None,
        config_path=args.config,
    )
    run_benchmark(service, benchmark_cfg)
