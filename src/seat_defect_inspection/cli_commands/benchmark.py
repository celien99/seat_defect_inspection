"""Benchmark command — evaluate inspection pipeline on standard datasets."""

from __future__ import annotations

import argparse

from ..runtime_config import load_config
from .common import DEFAULT_CONFIG_PATH


def register_benchmark_command(subparsers) -> None:
    """Register the benchmark command."""
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
        "--round",
        choices=["good", "defect", "mixed", "all"],
        default="all",
        help="Which round to run (default: all)",
    )
    parser.add_argument(
        "--cameras",
        help="Comma-separated camera IDs to benchmark (default: all enabled cameras)",
    )


def run_benchmark_command(args: argparse.Namespace) -> None:
    """Run benchmark inspection."""
    from ..service.benchmark import run_benchmark
    from ..service.core import InspectionService

    config = load_config(args.config)
    service = InspectionService(config)
    rounds = ("good", "defect", "mixed") if args.round == "all" else (args.round,)
    camera_ids = [c.strip() for c in args.cameras.split(",")] if args.cameras else None
    run_benchmark(service, rounds=rounds, camera_ids=camera_ids)
