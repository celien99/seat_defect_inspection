"""Benchmark configuration model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class BenchmarkConfig:
    data_dir: str = "benchmark_data"
    rounds: Tuple[str, ...] = ("good", "defect", "mixed")
    camera_ids: Optional[List[str]] = None
    seat_model_id: Optional[str] = None
    enable_threshold_sweep: bool = False
    sweep_steps: int = 50
    report_format: str = "html"
    report_output: Optional[str] = None
    curve_csv_dir: Optional[str] = None
    output_json: Optional[str] = None
    config_path: Optional[str] = None
