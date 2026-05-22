"""Tests for benchmark chart generation."""

from seat_defect_inspection.benchmark.plots import (
    fig_to_base64,
    plot_confusion_matrix,
    plot_defect_type_metrics,
    plot_per_camera_metrics,
    plot_pr_curve,
    plot_roc_curve,
    plot_score_distribution,
)
from seat_defect_inspection.benchmark.schemas import (
    ConfusionMatrix,
    CurveResult,
    DefectTypeMetrics,
    PerCameraMetrics,
    ScoreDistribution,
    ThresholdSweepPoint,
)


class TestConfusionMatrixPlot:
    def test_renders_base64(self):
        cm = ConfusionMatrix(tp=80, tn=15, fp=3, fn=2)
        result = plot_confusion_matrix(cm, title="Test CM")
        assert result
        assert len(result) > 100
        # Should be valid base64
        import base64
        base64.b64decode(result)


class TestROCPlot:
    def test_renders_with_data(self):
        points = [
            ThresholdSweepPoint(threshold=0.0, tpr=1.0, fpr=1.0, precision=0.5, f1=0.0),
            ThresholdSweepPoint(threshold=0.5, tpr=0.9, fpr=0.1, precision=0.9, f1=0.0),
            ThresholdSweepPoint(threshold=1.0, tpr=0.0, fpr=0.0, precision=0.0, f1=0.0),
        ]
        curve = CurveResult(points=points, auc=0.95)
        result = plot_roc_curve(curve)
        assert result
        assert len(result) > 100

    def test_renders_empty(self):
        curve = CurveResult()
        result = plot_roc_curve(curve)
        assert result


class TestPRPlot:
    def test_renders_with_data(self):
        points = [
            ThresholdSweepPoint(threshold=0.0, tpr=1.0, fpr=1.0, precision=0.5, f1=0.0),
            ThresholdSweepPoint(threshold=0.5, tpr=0.5, fpr=0.1, precision=0.8, f1=0.0),
            ThresholdSweepPoint(threshold=1.0, tpr=0.0, fpr=0.0, precision=0.0, f1=0.0),
        ]
        curve = CurveResult(points=points, auc=0.8)
        result = plot_pr_curve(curve)
        assert result


class TestScoreDistributionPlot:
    def test_renders(self):
        dists = [
            ScoreDistribution(label="OK", count=100, min=0.01, max=0.5, mean=0.2, median=0.18, std=0.1, p5=0.05, p95=0.45, all_scores=[0.1, 0.15, 0.2, 0.25, 0.3]),
            ScoreDistribution(label="NG", count=10, min=1.0, max=5.0, mean=2.5, median=2.3, std=1.2, p5=1.1, p95=4.8, all_scores=[1.0, 1.5, 2.0, 3.0, 5.0]),
        ]
        result = plot_score_distribution(dists)
        assert result

    def test_renders_empty(self):
        dists = [ScoreDistribution(label="OK", count=0)]
        result = plot_score_distribution(dists)
        assert result


class TestPerCameraPlot:
    def test_renders(self):
        metrics = [
            PerCameraMetrics(camera_id="cam_0", confusion=ConfusionMatrix(tp=40, tn=8, fp=1, fn=1), precision=0.98, recall=0.98, f1=0.98, accuracy=0.96),
            PerCameraMetrics(camera_id="cam_1", confusion=ConfusionMatrix(tp=38, tn=10, fp=1, fn=1), precision=0.97, recall=0.97, f1=0.97, accuracy=0.96),
        ]
        result = plot_per_camera_metrics(metrics)
        assert result

    def test_empty_returns_empty(self):
        result = plot_per_camera_metrics([])
        assert result == ""


class TestDefectTypePlot:
    def test_renders(self):
        metrics = [
            DefectTypeMetrics(defect_type="scratch", total=15, detected=13, recall=0.87, precision=0.90, f1=0.88),
            DefectTypeMetrics(defect_type="dent", total=8, detected=7, recall=0.88, precision=0.92, f1=0.90),
        ]
        result = plot_defect_type_metrics(metrics)
        assert result

    def test_empty_returns_empty(self):
        result = plot_defect_type_metrics([])
        assert result == ""


class TestFigToBase64:
    def test_converts(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        result = fig_to_base64(fig)
        assert result
        assert len(result) > 100
