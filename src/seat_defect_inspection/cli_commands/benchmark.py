"""Benchmark command — evaluate inspection pipeline on standard datasets."""

from __future__ import annotations

import argparse


def register_benchmark_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "benchmark",
        help="Evaluate inspection pipeline on good/defect/mixed datasets",
    )
    parser.set_defaults(run=run_benchmark_command)
    parser.add_argument(
        "--config",
        default=None,
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
        "--artifacts-dir",
        default=None,
        help="Output directory for overlay result images",
    )
    parser.add_argument(
        "--report-output",
        default=None,
        help="Markdown report output path (auto-generated if omitted)",
    )


def run_benchmark_command(args: argparse.Namespace) -> None:
    from ..benchmark import run_benchmark
    from ..benchmark.config import BenchmarkConfig
    from ..service.core import InspectionService
    from ..runtime_config import load_config
    from .common import DEFAULT_CONFIG_PATH

    config_path = args.config or DEFAULT_CONFIG_PATH
    config = load_config(config_path)
    service = InspectionService(config)
    rounds = ("good", "defect", "mixed") if args.round == "all" else (args.round,)
    camera_ids = [c.strip() for c in args.cameras.split(",")] if args.cameras else None

    data_dir = args.data_dir
    if data_dir is None:
        from pathlib import Path
        data_dir = str(Path.cwd() / "benchmark_data")

    report_output = args.report_output
    if report_output is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_output = f"outputs/benchmark/benchmark_report_{ts}.md"

    artifacts_dir = args.artifacts_dir
    if artifacts_dir is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifacts_dir = f"outputs/benchmark/artifacts_{ts}"

    benchmark_cfg = BenchmarkConfig(
        data_dir=data_dir,
        rounds=rounds,
        camera_ids=camera_ids,
        seat_model_id=args.seat_model_id,
        config_path=config_path,
        artifacts_dir=artifacts_dir,
        report_output=report_output,
    )
    run_benchmark(service, benchmark_cfg)
