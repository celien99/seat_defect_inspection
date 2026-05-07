"""PatchCore runtime scoring and anomaly decision helpers."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import PatchCoreConfig


def min_distance_to_bank(
    embeddings: np.ndarray,
    memory_bank: np.ndarray,
    chunk_size: int = 128,
) -> np.ndarray:
    """分块计算每个 embedding 到 memory bank 的最近距离。"""
    scores = []
    for start in range(0, len(embeddings), chunk_size):
        chunk = embeddings[start : start + chunk_size]
        distances = np.linalg.norm(chunk[:, None, :] - memory_bank[None, :, :], axis=2)
        scores.append(distances.min(axis=1))
    return np.concatenate(scores).astype(np.float32)


def normalize_map(
    heatmap: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """把热力图线性归一化到 [0, 1]，可选只在有效区域内归一化。"""
    normalized = np.zeros_like(heatmap, dtype=np.float32)
    if mask is None:
        active_mask = np.ones(heatmap.shape, dtype=bool)
    else:
        active_mask = np.asarray(mask).astype(bool)
        if active_mask.shape != heatmap.shape:
            raise ValueError("normalize_map mask shape must match heatmap")
        if not active_mask.any():
            return normalized

    active_values = heatmap[active_mask].astype(np.float32)
    minimum = float(active_values.min())
    maximum = float(active_values.max())
    if maximum - minimum < 1e-6:
        return normalized

    normalized[active_mask] = (active_values - minimum) / (maximum - minimum)
    return normalized.astype(np.float32)


def _threshold_margin(value: float) -> float:
    """阈值倍率默认不允许低于 1，避免把模型阈值反向调松。"""
    return max(float(value), 1.0)


def _decide_patchcore_anomaly(
    *,
    score: float,
    threshold: float,
    evidence: dict[str, float | int],
    config: PatchCoreConfig,
) -> tuple[bool, str]:
    """组合常规规则和小缺陷快路径，给出最终异常判定。"""
    decision_threshold = float(threshold) * _threshold_margin(config.decision_score_margin)
    critical_score_threshold = float(threshold) * _threshold_margin(config.critical_score_margin)
    critical_peak_threshold = float(threshold) * _threshold_margin(config.critical_peak_score_margin)
    peak_min_patch_count = max(2, int(config.critical_min_component_patch_count))
    component_min_patch_count = max(2, int(config.critical_min_component_patch_count))
    # peak_rule 仍然要比 normal_rule 更宽松，否则小面积真实缺陷会被漏掉；
    # 但也不能宽松到把边缘/附件热点都放行。
    peak_strong_patch_ratio_threshold = min(float(config.min_strong_patch_ratio) * 0.8, 0.0048)
    peak_component_ratio_threshold = min(float(config.min_strong_component_ratio) * 0.6, 0.0024)

    normal_trigger = (
        float(score) > decision_threshold
        and int(evidence["strong_patch_count"]) >= int(config.min_strong_patch_count)
        and int(evidence["largest_component_patch_count"]) >= int(config.min_strong_component_count)
        and float(evidence["strong_patch_ratio"]) >= float(config.min_strong_patch_ratio)
        and float(evidence["largest_component_patch_ratio"]) >= float(config.min_strong_component_ratio)
    )
    critical_trigger = (
        float(score) > critical_score_threshold
        and float(evidence["peak_patch_score"]) > critical_peak_threshold
        and int(evidence["largest_component_patch_count"]) >= component_min_patch_count
    )
    # 小面积缺陷容易被 99 分位 image score 稀释，这里补一条“局部峰值”直通规则。
    peak_trigger = (
        float(evidence["peak_patch_score"]) > critical_peak_threshold
        and int(evidence["strong_patch_count"]) >= peak_min_patch_count
        and int(evidence["largest_component_patch_count"]) >= component_min_patch_count
        and float(evidence["strong_patch_ratio"]) >= peak_strong_patch_ratio_threshold
        and float(evidence["largest_component_patch_ratio"]) >= peak_component_ratio_threshold
    )

    if normal_trigger and critical_trigger:
        return True, "normal_and_critical"
    if critical_trigger:
        return True, "critical_rule"
    if normal_trigger:
        return True, "normal_rule"
    if peak_trigger:
        return True, "peak_rule"
    return False, "none"


def _analyze_patch_evidence(
    patch_map: np.ndarray,
    *,
    score: float,
    threshold: float,
    valid_patch_count: int,
    config: PatchCoreConfig,
) -> dict[str, float | int]:
    """从 patch map 中提取最终判定依赖的统计证据。"""
    peak_patch_score = float(patch_map.max()) if patch_map.size > 0 else 0.0
    if patch_map.size == 0 or valid_patch_count <= 0:
        return {
            "peak_patch_score": peak_patch_score,
            "strong_patch_count": 0,
            "largest_component_patch_count": 0,
            "strong_patch_ratio": 0.0,
            "largest_component_patch_ratio": 0.0,
        }

    # 强 patch 门槛按比例收紧，而不是强行抬到完整 threshold。
    strong_patch_ratio = float(np.clip(config.strong_patch_score_ratio, 0.0, 1.0))
    strong_patch_floor = max(float(threshold), float(score)) * strong_patch_ratio
    strong_patch_mask = (patch_map >= strong_patch_floor).astype(np.uint8)
    strong_patch_count = int(strong_patch_mask.sum())
    if strong_patch_count == 0:
        return {
            "peak_patch_score": peak_patch_score,
            "strong_patch_count": 0,
            "largest_component_patch_count": 0,
            "strong_patch_ratio": 0.0,
            "largest_component_patch_ratio": 0.0,
        }

    _, _, stats, _ = cv2.connectedComponentsWithStats(strong_patch_mask, connectivity=8)
    largest_component_patch_count = int(stats[1:, cv2.CC_STAT_AREA].max()) if len(stats) > 1 else 0
    return {
        "peak_patch_score": peak_patch_score,
        "strong_patch_count": strong_patch_count,
        "largest_component_patch_count": largest_component_patch_count,
        "strong_patch_ratio": float(strong_patch_count) / float(max(1, valid_patch_count)),
        "largest_component_patch_ratio": float(largest_component_patch_count) / float(max(1, valid_patch_count)),
    }
