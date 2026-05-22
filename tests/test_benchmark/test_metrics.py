"""Tests for benchmark metric computation."""

from seat_defect_inspection.benchmark.metrics import (
    _trapezoidal_auc,
    _wilson_ci,
    compute_all_metrics,
    compute_binary_metrics,
    compute_confusion_matrix,
    compute_defect_type_metrics,
    compute_score_distributions,
    compute_threshold_sweep,
    compute_timing_stats,
)
from seat_defect_inspection.benchmark.config import BenchmarkConfig
from seat_defect_inspection.benchmark.schemas import (
    BenchmarkRecord,
    BenchmarkSample,
    CameraBenchmarkRecord,
    ConfusionMatrix,
)


def _make_record(
    index: int,
    predicted_status: str,
    gt_label: str,
    anomaly_scores: dict = None,
    defect_type: str = None,
    timing_ms: float = 100.0,
) -> BenchmarkRecord:
    cam_records = []
    if anomaly_scores:
        for cid, score in anomaly_scores.items():
            cam_records.append(CameraBenchmarkRecord(
                camera_id=cid,
                predicted_status="NG" if score > 1.0 else "OK",
                anomaly_score=score,
            ))
    else:
        cam_records = [
            CameraBenchmarkRecord(
                camera_id="cam_0",
                predicted_status=predicted_status,
                anomaly_score=2.0 if predicted_status == "NG" else 0.5,
            )
        ]
    return BenchmarkRecord(
        sample=BenchmarkSample(
            index=index,
            part_id=f"test_{index:04d}",
            image_paths={"cam_0": f"test_{index}.png"},
            ground_truth_label=gt_label,
            ground_truth_defect_type=defect_type,
        ),
        predicted_status=predicted_status,
        decision_reason="test_reason",
        camera_records=cam_records,
        inference_timing_ms=timing_ms,
    )


class TestConfusionMatrix:
    def test_perfect_predictions(self):
        records = [
            _make_record(0, "OK", "OK"),
            _make_record(1, "OK", "OK"),
            _make_record(2, "NG", "NG"),
            _make_record(3, "NG", "NG"),
        ]
        cm = compute_confusion_matrix(records)
        assert cm.tp == 2
        assert cm.tn == 2
        assert cm.fp == 0
        assert cm.fn == 0
        assert cm.miss_rate == 0.0
        assert cm.false_alarm_rate == 0.0

    def test_all_missed(self):
        records = [
            _make_record(0, "OK", "NG"),
            _make_record(1, "OK", "NG"),
        ]
        cm = compute_confusion_matrix(records)
        assert cm.tp == 0
        assert cm.fn == 2
        assert cm.miss_rate == 1.0

    def test_all_false_alarm(self):
        records = [
            _make_record(0, "NG", "OK"),
            _make_record(1, "NG", "OK"),
        ]
        cm = compute_confusion_matrix(records)
        assert cm.fp == 2
        assert cm.false_alarm_rate == 1.0

    def test_empty_records(self):
        cm = compute_confusion_matrix([])
        assert cm.total == 0


class TestBinaryMetrics:
    def test_perfect_metrics(self):
        records = [
            _make_record(0, "OK", "OK"),
            _make_record(1, "NG", "NG"),
        ]
        cm = compute_confusion_matrix(records)
        bm = compute_binary_metrics(cm)
        assert bm.precision == 1.0
        assert bm.recall == 1.0
        assert bm.f1 == 1.0
        assert bm.accuracy == 1.0
        assert bm.miss_rate == 0.0
        assert bm.false_alarm_rate == 0.0

    def test_imbalanced(self):
        records = [
            _make_record(0, "OK", "OK"),
            _make_record(1, "OK", "OK"),
            _make_record(2, "OK", "OK"),
            _make_record(3, "NG", "NG"),  # tp
            _make_record(4, "OK", "NG"),  # fn
        ]
        cm = compute_confusion_matrix(records)
        bm = compute_binary_metrics(cm)
        assert bm.recall == 0.5  # 1/2
        assert bm.miss_rate == 0.5  # 1/2
        assert bm.false_alarm_rate == 0.0  # 0/3


class TestWilsonCI:
    def test_basic(self):
        lo, hi = _wilson_ci(95, 100)
        assert 0.88 < lo < 0.99
        assert 0.95 < hi < 1.0

    def test_zero_trials(self):
        lo, hi = _wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 0.0

    def test_perfect_score(self):
        lo, hi = _wilson_ci(50, 50)
        assert lo > 0.9


class TestDefectTypeMetrics:
    def test_per_type_recall(self):
        records = [
            _make_record(0, "NG", "NG", defect_type="scratch"),
            _make_record(1, "NG", "NG", defect_type="scratch"),
            _make_record(2, "OK", "NG", defect_type="scratch"),  # missed
            _make_record(3, "NG", "NG", defect_type="dent"),
            _make_record(4, "NG", "NG", defect_type="dent"),
        ]
        cm = compute_confusion_matrix(records)
        results = compute_defect_type_metrics(records, cm)
        scratch = [r for r in results if r.defect_type == "scratch"][0]
        assert scratch.total == 3
        assert scratch.detected == 2
        assert abs(scratch.recall - 2/3) < 0.01
        dent = [r for r in results if r.defect_type == "dent"][0]
        assert dent.total == 2
        assert dent.detected == 2
        assert dent.recall == 1.0


class TestScoreDistribution:
    def test_distribution_stats(self):
        records = [
            _make_record(0, "OK", "OK", anomaly_scores={"cam_0": 0.1}),
            _make_record(1, "OK", "OK", anomaly_scores={"cam_0": 0.3}),
            _make_record(2, "NG", "NG", anomaly_scores={"cam_0": 2.0}),
            _make_record(3, "NG", "NG", anomaly_scores={"cam_0": 4.0}),
        ]
        dists = compute_score_distributions(records)
        ok_dist = [d for d in dists if d.label == "OK"][0]
        ng_dist = [d for d in dists if d.label == "NG"][0]
        assert ok_dist.count == 2
        assert ng_dist.count == 2
        assert ok_dist.mean < ng_dist.mean


class TestThresholdSweep:
    def test_sweep_two_classes(self):
        records = [
            _make_record(0, "OK", "OK", anomaly_scores={"cam_0": 0.1}),
            _make_record(1, "OK", "OK", anomaly_scores={"cam_0": 0.3}),
            _make_record(2, "OK", "OK", anomaly_scores={"cam_0": 0.2}),
            _make_record(3, "NG", "NG", anomaly_scores={"cam_0": 2.0}),
            _make_record(4, "NG", "NG", anomaly_scores={"cam_0": 3.0}),
            _make_record(5, "NG", "NG", anomaly_scores={"cam_0": 4.0}),
        ]
        config = BenchmarkConfig(enable_threshold_sweep=True, sweep_steps=10)
        roc, pr = compute_threshold_sweep(records, config)
        # With well-separated scores, AUC should be high
        assert roc.auc > 0.9
        assert pr.auc > 0.9

    def test_sweep_no_gt(self):
        records = [
            _make_record(0, "OK", None),
        ]
        config = BenchmarkConfig(enable_threshold_sweep=True, sweep_steps=10)
        roc, pr = compute_threshold_sweep(records, config)
        assert roc.auc == 0.0
        assert pr.auc == 0.0


class TestTimingStats:
    def test_timing(self):
        records = [
            _make_record(i, "OK", "OK", timing_ms=100.0 + i * 10)
            for i in range(5)
        ]
        ts = compute_timing_stats(records)
        assert len(ts.all_timings_ms) == 5
        assert ts.mean_ms > 100
        assert ts.max_ms > ts.min_ms


class TestTrapezoidalAUC:
    def test_perfect(self):
        points = [(0.0, 0.0), (0.5, 1.0), (1.0, 1.0)]
        assert abs(_trapezoidal_auc(points) - 0.75) < 0.01

    def test_perfect_curve(self):
        points = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        assert abs(_trapezoidal_auc(points) - 1.0) < 0.01
