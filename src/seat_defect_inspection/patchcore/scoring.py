"""PatchCore 打分、阈值校准与异常判定。"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import PatchCoreConfig


def _determine_memory_bank_size(
    embeddings: np.ndarray,
    config: PatchCoreConfig,
) -> int:
    """根据配置和样本量确定 memory bank 目标容量。"""
    ratio = float(np.clip(config.coreset_sampling_ratio, 0.0, 1.0))
    if ratio > 0.0:
        ratio_target = max(1, int(round(len(embeddings) * ratio)))
    else:
        ratio_target = max(64, min(config.max_memory, len(embeddings) // 4 or 1))
    return min(len(embeddings), max(1, min(config.max_memory, ratio_target)))


def coreset_subsample_indices(embeddings: np.ndarray, max_points: int) -> np.ndarray:
    """贪心 coreset 采样，返回被保留的 embedding 行索引。"""
    if len(embeddings) <= max_points:
        return np.arange(len(embeddings), dtype=np.int32)

    rng = np.random.default_rng(42)
    first_index = int(rng.integers(0, len(embeddings)))
    chosen_indices = [first_index]
    min_distances = np.linalg.norm(embeddings - embeddings[first_index], axis=1)

    while len(chosen_indices) < max_points:
        next_index = int(np.argmax(min_distances))
        chosen_indices.append(next_index)
        next_distances = np.linalg.norm(embeddings - embeddings[next_index], axis=1)
        min_distances = np.minimum(min_distances, next_distances)

    return np.asarray(chosen_indices, dtype=np.int32)


def _exclude_embedding_slice(embeddings: np.ndarray, start: int, end: int) -> np.ndarray:
    """从拼接后的 embedding 中排除当前样本切片。"""
    if start <= 0:
        return embeddings[end:]
    if end >= len(embeddings):
        return embeddings[:start]
    return np.concatenate((embeddings[:start], embeddings[end:]), axis=0)


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


def _score_embeddings_leave_one_out(embeddings: np.ndarray) -> tuple[float, np.ndarray]:
    """缺少外部校准 bank 时，退化成样本内 leave-one-out 打分。"""
    if len(embeddings) <= 1:
        patch_scores = np.zeros((len(embeddings),), dtype=np.float32)
        return 0.0, patch_scores

    distances = np.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    patch_scores = distances.min(axis=1).astype(np.float32)
    image_score = float(np.percentile(patch_scores, 99))
    return image_score, patch_scores


def normalize_map(heatmap: np.ndarray) -> np.ndarray:
    """把热力图线性归一化到 [0, 1]。"""
    minimum = float(heatmap.min())
    maximum = float(heatmap.max())
    if maximum - minimum < 1e-6:
        return np.zeros_like(heatmap, dtype=np.float32)
    return ((heatmap - minimum) / (maximum - minimum)).astype(np.float32)


def _positive_margin(value: float) -> float:
    """避免阈值倍率被配置成 0 或负数。"""
    return max(float(value), 1e-6)


def _decide_patchcore_anomaly(
    *,
    score: float,
    threshold: float,
    evidence: dict[str, float | int],
    config: PatchCoreConfig,
) -> tuple[bool, str]:
    """组合常规规则和小缺陷快路径，给出最终异常判定。"""
    decision_threshold = float(threshold) * _positive_margin(config.decision_score_margin)
    critical_score_threshold = float(threshold) * _positive_margin(config.critical_score_margin)
    critical_peak_threshold = float(threshold) * _positive_margin(config.critical_peak_score_margin)
    peak_min_patch_count = max(2, int(config.critical_min_component_patch_count))

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
        and int(evidence["largest_component_patch_count"]) >= int(config.critical_min_component_patch_count)
    )
    # 小面积缺陷容易被 99 分位 image score 稀释，这里补一条“局部峰值”直通规则。
    peak_trigger = (
        float(evidence["peak_patch_score"]) > critical_peak_threshold
        and int(evidence["strong_patch_count"]) >= peak_min_patch_count
        and int(evidence["largest_component_patch_count"]) >= int(config.critical_min_component_patch_count)
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

    # 强 patch 门槛按比例收缩，而不是强行抬到完整 threshold，
    # 否则小缺陷即便有明显热点，也可能永远进不到强证据统计。
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
