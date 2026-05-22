"""Benchmark configuration model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class BenchmarkConfig:
    data_dir: str = "benchmark_data"
    rounds: Tuple[str, ...] = ("good", "defect", "mixed")
    camera_ids: Optional[List[str]] = None
    seat_model_id: Optional[str] = None
    config_path: Optional[str] = None
    artifacts_dir: Optional[str] = None
    report_output: Optional[str] = None
