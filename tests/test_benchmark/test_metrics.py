"""Tests for benchmark metric computation."""

from seat_defect_inspection.benchmark.metrics import (
    compute_binary_metrics,
    compute_confusion_matrix,
    compute_per_camera_metrics,
    identify_failure_cases,
)
from seat_defect_inspection.benchmark.schemas import (
    BenchmarkRecord,
    BenchmarkSample,
    CameraBenchmarkRecord,
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

    def test_all_missed(self):
        records = [
            _make_record(0, "OK", "NG"),
            _make_record(1, "OK", "NG"),
        ]
        cm = compute_confusion_matrix(records)
        assert cm.tp == 0
        assert cm.fn == 2

    def test_all_false_alarm(self):
        records = [
            _make_record(0, "NG", "OK"),
            _make_record(1, "NG", "OK"),
        ]
        cm = compute_confusion_matrix(records)
        assert cm.fp == 2

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


class TestFailureCases:
    def test_identify_miss_and_false_alarm(self):
        records = [
            _make_record(0, "OK", "NG"),   # miss
            _make_record(1, "NG", "OK"),   # false alarm
            _make_record(2, "OK", "OK"),   # correct
            _make_record(3, "NG", "NG"),   # correct
        ]
        failures = identify_failure_cases(records)
        assert len(failures) == 2
        indices = {f.sample.index for f in failures}
        assert indices == {0, 1}

    def test_no_failures(self):
        records = [
            _make_record(0, "OK", "OK"),
            _make_record(1, "NG", "NG"),
        ]
        failures = identify_failure_cases(records)
        assert len(failures) == 0


def _make_multi_camera_record(
    index: int,
    predicted_statuses: dict,
    gt_label: str,
    camera_gt: dict = None,
) -> BenchmarkRecord:
    """Build a record with multiple cameras, each having its own predicted status."""
    cam_records = [
        CameraBenchmarkRecord(
            camera_id=cid,
            predicted_status=status,
            anomaly_score=2.0 if status != "OK" else 0.5,
        )
        for cid, status in predicted_statuses.items()
    ]
    image_paths = {cid: f"test_{index}_{cid}.png" for cid in predicted_statuses}
    return BenchmarkRecord(
        sample=BenchmarkSample(
            index=index,
            part_id=f"test_{index:04d}",
            image_paths=image_paths,
            ground_truth_label=gt_label,
            camera_ground_truth=camera_gt or {},
        ),
        predicted_status="NG" if any(s != "OK" for s in predicted_statuses.values()) else "OK",
        decision_reason="test_reason",
        camera_records=cam_records,
    )


class TestPerCameraMetrics:
    def test_perfect_per_camera(self):
        """Both cameras predict correctly against overall GT."""
        records = [
            _make_multi_camera_record(0, {"cam_0": "OK", "cam_1": "OK"}, "OK"),
            _make_multi_camera_record(1, {"cam_0": "NG", "cam_1": "NG"}, "NG"),
        ]
        result = compute_per_camera_metrics(records, ["cam_0", "cam_1"])
        for pc in result:
            assert pc.confusion.tp == 1
            assert pc.confusion.tn == 1
            assert pc.confusion.fp == 0
            assert pc.confusion.fn == 0
            assert pc.precision == 1.0
            assert pc.recall == 1.0
            assert pc.f1 == 1.0

    def test_uses_camera_ground_truth(self):
        """cam_0 GT=NG but overall GT=OK. cam_0 predicted NG correctly per its own GT."""
        records = [
            _make_multi_camera_record(
                0,
                {"cam_0": "NG", "cam_1": "OK"},
                gt_label="OK",
                camera_gt={"cam_0": "NG", "cam_1": "OK"},
            ),
        ]
        result = compute_per_camera_metrics(records, ["cam_0", "cam_1"])
        cam0 = [pc for pc in result if pc.camera_id == "cam_0"][0]
        cam1 = [pc for pc in result if pc.camera_id == "cam_1"][0]
        # cam_0: GT=NG, pred=NG → TP
        assert cam0.confusion.tp == 1
        assert cam0.confusion.fn == 0
        # cam_1: GT=OK, pred=OK → TN
        assert cam1.confusion.tn == 1
        assert cam1.confusion.fp == 0

    def test_fallback_to_overall_gt(self):
        """When camera_ground_truth is empty, fall back to overall GT."""
        records = [
            _make_multi_camera_record(
                0,
                {"cam_0": "OK", "cam_1": "NG"},
                gt_label="NG",
                camera_gt={},
            ),
        ]
        result = compute_per_camera_metrics(records, ["cam_0", "cam_1"])
        cam0 = [pc for pc in result if pc.camera_id == "cam_0"][0]
        cam1 = [pc for pc in result if pc.camera_id == "cam_1"][0]
        # cam_0: GT=NG (fallback), pred=OK → FN
        assert cam0.confusion.fn == 1
        # cam_1: GT=NG (fallback), pred=NG → TP
        assert cam1.confusion.tp == 1

    def test_reject_treated_as_not_ok(self):
        """REJECT predicted → counted as not-OK (positive call)."""
        records = [
            _make_multi_camera_record(
                0,
                {"cam_0": "REJECT", "cam_1": "OK"},
                gt_label="OK",
            ),
        ]
        result = compute_per_camera_metrics(records, ["cam_0", "cam_1"])
        cam0 = [pc for pc in result if pc.camera_id == "cam_0"][0]
        # GT=OK, pred=REJECT → FP (REJECT != OK)
        assert cam0.confusion.fp == 1

    def test_empty_records(self):
        result = compute_per_camera_metrics([], ["cam_0"])
        assert len(result) == 1
        assert result[0].confusion.total == 0
