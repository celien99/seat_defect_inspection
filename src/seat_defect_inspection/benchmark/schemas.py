"""Structured data models for benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------- dataset ----------


@dataclass
class DatasetComposition:
    round_name: str
    sample_count: int
    camera_count: int
    camera_ids: List[str]
    ng_count: int = 0
    ok_count: int = 0
    has_ground_truth: bool = False
    ground_truth_source: str = "none"  # "manifest" | "implicit" | "none"


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
    decision_threshold: Optional[float] = None
    peak_patch_score: Optional[float] = None
    strong_patch_count: Optional[int] = None
    decision_mode: Optional[str] = None
    is_anomaly: Optional[bool] = None
    valid_patch_ratio: Optional[float] = None
    timing_ms: Dict[str, float] = field(default_factory=dict)


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

    @property
    def miss_rate(self) -> float:
        denom = self.tp + self.fn
        return self.fn / denom if denom > 0 else 0.0

    @property
    def false_alarm_rate(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0.0


@dataclass
class BinaryMetrics:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    miss_rate: float = 0.0
    false_alarm_rate: float = 0.0
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class ThresholdSweepPoint:
    threshold: float
    tpr: float
    fpr: float
    precision: float
    f1: float


@dataclass
class CurveResult:
    points: List[ThresholdSweepPoint] = field(default_factory=list)
    auc: float = 0.0


@dataclass
class PerCameraMetrics:
    camera_id: str
    confusion: ConfusionMatrix = field(default_factory=ConfusionMatrix)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0


@dataclass
class DefectTypeMetrics:
    defect_type: str
    total: int = 0
    detected: int = 0
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0


@dataclass
class ScoreDistribution:
    label: str  # "OK" | "NG"
    count: int = 0
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    p5: float = 0.0
    p95: float = 0.0
    all_scores: List[float] = field(default_factory=list)


@dataclass
class TimingStats:
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    all_timings_ms: List[float] = field(default_factory=list)


# ---------- round / summary ----------


@dataclass
class RoundResult:
    round_name: str
    composition: DatasetComposition = field(default_factory=lambda: DatasetComposition(round_name=""))
    confusion: Optional[ConfusionMatrix] = None
    binary_metrics: Optional[BinaryMetrics] = None
    per_camera: List[PerCameraMetrics] = field(default_factory=list)
    defect_type_breakdown: List[DefectTypeMetrics] = field(default_factory=list)
    score_distributions: List[ScoreDistribution] = field(default_factory=list)
    timing: Optional[TimingStats] = None
    roc_curve: Optional[CurveResult] = None
    pr_curve: Optional[CurveResult] = None
    records: List[BenchmarkRecord] = field(default_factory=list)
    failure_cases: List[BenchmarkRecord] = field(default_factory=list)


@dataclass
class BenchmarkSummary:
    rounds: List[RoundResult] = field(default_factory=list)
    combined_metrics: Optional[BinaryMetrics] = None
    dataset_overview: List[DatasetComposition] = field(default_factory=list)
