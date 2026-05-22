"""Structured data models for benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------- sample / record ----------


@dataclass
class BenchmarkSample:
    index: int
    part_id: str
    image_paths: Dict[str, str]
    ground_truth_label: Optional[str] = None
    ground_truth_defect_type: Optional[str] = None
    ground_truth_severity: Optional[str] = None
    camera_ground_truth: Dict[str, str] = field(default_factory=dict)


@dataclass
class CameraBenchmarkRecord:
    camera_id: str
    predicted_status: str  # OK / NG / REJECT
    anomaly_score: Optional[float] = None
    anomaly_threshold: Optional[float] = None
    overlay_path: Optional[str] = None


@dataclass
class BenchmarkRecord:
    sample: BenchmarkSample
    predicted_status: str  # OK / NG / REJECT
    decision_reason: str
    camera_records: List[CameraBenchmarkRecord] = field(default_factory=list)
    inference_timing_ms: float = 0.0


# ---------- metrics ----------


@dataclass
class ConfusionMatrix:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn


@dataclass
class BinaryMetrics:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    miss_rate: float = 0.0
    false_alarm_rate: float = 0.0


@dataclass
class PerCameraMetrics:
    camera_id: str
    confusion: ConfusionMatrix = field(default_factory=ConfusionMatrix)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    miss_rate: float = 0.0
    false_alarm_rate: float = 0.0


# ---------- round / summary ----------


@dataclass
class RoundResult:
    round_name: str = ""
    sample_count: int = 0
    ok_count: int = 0
    ng_count: int = 0
    reject_count: int = 0
    confusion: Optional[ConfusionMatrix] = None
    binary_metrics: Optional[BinaryMetrics] = None
    per_camera: List[PerCameraMetrics] = field(default_factory=list)
    records: List[BenchmarkRecord] = field(default_factory=list)
    failure_cases: List[BenchmarkRecord] = field(default_factory=list)


@dataclass
class BenchmarkSummary:
    rounds: List[RoundResult] = field(default_factory=list)
    camera_ids: List[str] = field(default_factory=list)
    combined_metrics: Optional[BinaryMetrics] = None
