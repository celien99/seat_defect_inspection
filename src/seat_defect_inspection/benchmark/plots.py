"""Chart generation for benchmark reports using matplotlib."""

from __future__ import annotations

import io
import base64
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .schemas import (
    ConfusionMatrix,
    CurveResult,
    DefectTypeMetrics,
    PerCameraMetrics,
    RoundResult,
    ScoreDistribution,
    TimingStats,
)


FIGSIZE_SMALL = (8, 5)
FIGSIZE_MEDIUM = (10, 6)
DEFAULT_DPI = 150
BAR_COLORS = ["#2ecc71", "#e74c3c"]


def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------- confusion matrix heatmap ----------


def plot_confusion_matrix(
    cm: ConfusionMatrix,
    title: str = "Confusion Matrix",
) -> str:
    matrix = np.array([[cm.tn, cm.fp], [cm.fn, cm.tp]], dtype=np.float64)
    total = cm.total
    normalized = matrix / total if total > 0 else matrix
    labels = np.array([
        [f"TN={cm.tn}\n({normalized[0, 0]:.1%})", f"FP={cm.fp}\n({normalized[0, 1]:.1%})"],
        [f"FN={cm.fn}\n({normalized[1, 0]:.1%})", f"TP={cm.tp}\n({normalized[1, 1]:.1%})"],
    ])
    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted OK", "Predicted NG"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual OK", "Actual NG"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i, j], ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if normalized[i, j] > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Count")
    ax.set_title(title, fontsize=13, fontweight="bold")
    return fig_to_base64(fig)


# ---------- ROC curve ----------


def plot_roc_curve(curve: CurveResult, title: str = "ROC Curve") -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)
    if curve.points:
        fprs = [p.fpr for p in curve.points]
        tprs = [p.tpr for p in curve.points]
        ax.plot(fprs, tprs, "b-", linewidth=2, label=f"ROC (AUC = {curve.auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    return fig_to_base64(fig)


# ---------- PR curve ----------


def plot_pr_curve(curve: CurveResult, baseline: float = 0.5, title: str = "Precision-Recall Curve") -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)
    if curve.points:
        tprs = [p.tpr for p in curve.points]
        precs = [p.precision for p in curve.points]
        ax.plot(tprs, precs, "r-", linewidth=2, label=f"PR (AUC = {curve.auc:.4f})")
    ax.axhline(y=baseline, color="k", linestyle="--", linewidth=1, label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    return fig_to_base64(fig)


# ---------- score distribution histogram ----------


def plot_score_distribution(distributions: List[ScoreDistribution], title: str = "Anomaly Score Distribution") -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)
    colors = {"OK": "#3498db", "NG": "#e74c3c"}
    for dist in distributions:
        if dist.count == 0:
            continue
        color = colors.get(dist.label, "#95a5a6")
        ax.hist(
            dist.all_scores, bins=30, alpha=0.5, label=f"{dist.label} (n={dist.count})",
            color=color, edgecolor="white",
        )
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    return fig_to_base64(fig)


# ---------- per-camera bar chart ----------


def plot_per_camera_metrics(
    per_camera: List[PerCameraMetrics],
    title: str = "Per-Camera Metrics",
) -> str:
    if not per_camera:
        return ""
    camera_ids = [p.camera_id for p in per_camera]
    x = np.arange(len(camera_ids))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(camera_ids) * 1.2), 6))
    precision_vals = [p.precision * 100 for p in per_camera]
    recall_vals = [p.recall * 100 for p in per_camera]
    f1_vals = [p.f1 * 100 for p in per_camera]
    bars1 = ax.bar(x - width, precision_vals, width, label="Precision (%)", color="#3498db")
    bars2 = ax.bar(x, recall_vals, width, label="Recall (%)", color="#2ecc71")
    bars3 = ax.bar(x + width, f1_vals, width, label="F1 (%)", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(camera_ids, rotation=45, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 105)
    return fig_to_base64(fig)


# ---------- defect-type bar chart ----------


def plot_defect_type_metrics(
    defect_types: List[DefectTypeMetrics],
    title: str = "Per-Defect-Type Recall",
) -> str:
    if not defect_types:
        return ""
    labels = [d.defect_type for d in defect_types]
    recall_vals = [d.recall * 100 for d in defect_types]
    precision_vals = [d.precision * 100 for d in defect_types]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 6))
    bars1 = ax.bar(x - width / 2, recall_vals, width, label="Recall (%)", color="#2ecc71")
    bars2 = ax.bar(x + width / 2, precision_vals, width, label="Precision (%)", color="#3498db")
    # Annotate totals above bars
    for i, d in enumerate(defect_types):
        ax.text(i - width / 2, recall_vals[i] + 1, f"{d.detected}/{d.total}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 105)
    return fig_to_base64(fig)
