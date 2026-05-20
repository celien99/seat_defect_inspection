"""PatchCore runtime scoring and anomaly decision helpers."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional runtime acceleration
    torch = None

from ..config import PatchCoreConfig


def _determine_memory_bank_size(
    embeddings: np.ndarray,
    config: PatchCoreConfig,
) -> int:
    """Choose a target memory-bank size from config and sample count."""
    ratio = float(np.clip(config.coreset_sampling_ratio, 0.0, 1.0))
    if ratio > 0.0:
        ratio_target = max(1, int(round(len(embeddings) * ratio)))
    else:
        ratio_target = max(64, min(config.max_memory, len(embeddings) // 4 or 1))
    return min(len(embeddings), max(1, min(config.max_memory, ratio_target)))


def coreset_subsample_indices(embeddings: np.ndarray, max_points: int) -> np.ndarray:
    """Greedy coreset sampling that keeps representative embedding rows."""
    if len(embeddings) <= max_points:
        return np.arange(len(embeddings), dtype=np.int32)

    if torch is not None and len(embeddings) * embeddings.shape[1] >= 1_000_000:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(device) == "cpu" and torch.backends.mps.is_available():
            device = torch.device("mps")
        try:
            return _coreset_subsample_torch(embeddings, max_points, device)
        except Exception:
            pass

    return _coreset_subsample_numpy(embeddings, max_points)


def _coreset_subsample_numpy(embeddings: np.ndarray, max_points: int) -> np.ndarray:
    """CPU numpy implementation of greedy coreset sampling."""
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


def _coreset_subsample_torch(
    embeddings: np.ndarray,
    max_points: int,
    device,
) -> np.ndarray:
    """GPU-accelerated greedy coreset sampling via torch.cdist."""
    rng = np.random.default_rng(42)
    first_index = int(rng.integers(0, len(embeddings)))

    tensor = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    chosen_indices = [first_index]
    min_distances = torch.norm(tensor - tensor[first_index], dim=1)

    with torch.inference_mode():
        while len(chosen_indices) < max_points:
            next_index = int(torch.argmax(min_distances).item())
            chosen_indices.append(next_index)
            next_distances = torch.norm(tensor - tensor[next_index], dim=1)
            torch.minimum(min_distances, next_distances, out=min_distances)

    return np.asarray(chosen_indices, dtype=np.int32)


def _exclude_embedding_slice(embeddings: np.ndarray, start: int, end: int) -> np.ndarray:
    """Exclude the current sample slice from a concatenated embedding array."""
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
    """Compute per-embedding nearest distance to the memory bank in chunks."""
    scores = []
    for start in range(0, len(embeddings), chunk_size):
        chunk = embeddings[start : start + chunk_size]
        distances = np.linalg.norm(chunk[:, None, :] - memory_bank[None, :, :], axis=2)
        scores.append(distances.min(axis=1))
    return np.concatenate(scores).astype(np.float32)


def min_distance_to_bank_torch(
    embeddings: np.ndarray,
    memory_bank,
    *,
    device,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Compute nearest memory-bank distance on a torch device."""
    if torch is None:
        return min_distance_to_bank(embeddings, np.asarray(memory_bank), chunk_size=128)
    if len(embeddings) == 0:
        return np.zeros((0,), dtype=np.float32)

    bank_tensor = memory_bank
    if not torch.is_tensor(bank_tensor):
        bank_tensor = torch.as_tensor(memory_bank, dtype=torch.float32, device=device)
    else:
        bank_tensor = bank_tensor.to(device=device, dtype=torch.float32)

    embedding_tensor = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    scores = []
    with torch.inference_mode():
        for start in range(0, int(embedding_tensor.shape[0]), chunk_size):
            chunk = embedding_tensor[start : start + chunk_size]
            distances = torch.cdist(chunk, bank_tensor)
            scores.append(distances.min(dim=1).values.detach().cpu())
    return torch.cat(scores).numpy().astype(np.float32)


def _score_embeddings_leave_one_out(embeddings: np.ndarray) -> Tuple[float, np.ndarray]:
    """Fallback leave-one-out scoring when no external bank is available."""
    if len(embeddings) <= 1:
        patch_scores = np.zeros((len(embeddings),), dtype=np.float32)
        return 0.0, patch_scores

    distances = np.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    patch_scores = distances.min(axis=1).astype(np.float32)
    image_score = float(np.percentile(patch_scores, 99))
    return image_score, patch_scores


def normalize_map(
    heatmap: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Normalize a heatmap to [0, 1] within the active region."""
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


def normalize_map_against_threshold(
    heatmap: np.ndarray,
    *,
    threshold: float,
    mask: Optional[np.ndarray] = None,
    floor_ratio: float = 0.5,
) -> np.ndarray:
    """Normalize a heatmap against an absolute decision threshold."""
    normalized = np.zeros_like(heatmap, dtype=np.float32)
    reference = float(threshold)
    if reference <= 1e-6:
        return normalize_map(heatmap, mask=mask)

    if mask is None:
        active_mask = np.ones(heatmap.shape, dtype=bool)
    else:
        active_mask = np.asarray(mask).astype(bool)
        if active_mask.shape != heatmap.shape:
            raise ValueError("normalize_map_against_threshold mask shape must match heatmap")
        if not active_mask.any():
            return normalized

    clipped_floor_ratio = float(np.clip(floor_ratio, 0.0, 0.95))
    floor_value = reference * clipped_floor_ratio
    scale = max(reference - floor_value, 1e-6)
    active_values = heatmap[active_mask].astype(np.float32)
    normalized[active_mask] = np.clip((active_values - floor_value) / scale, 0.0, 1.0)
    return normalized.astype(np.float32)


def _threshold_margin(value: float) -> float:
    """Do not allow threshold multipliers below 1.0."""
    return max(float(value), 1.0)


def _decide_patchcore_anomaly(
    *,
    score: float,
    threshold: float,
    evidence: Dict[str, Union[float, int]],
    config: PatchCoreConfig,
) -> Tuple[bool, str]:
    """Combine normal and small-defect rules into one final anomaly decision."""
    decision_threshold = float(threshold) * _threshold_margin(config.decision_score_margin)
    critical_score_threshold = float(threshold) * _threshold_margin(config.critical_score_margin)
    critical_peak_threshold = float(threshold) * _threshold_margin(config.critical_peak_score_margin)
    component_min_patch_count = max(1, int(config.critical_min_component_patch_count))
    largest_decision_component_patch_count = int(
        evidence.get("largest_decision_component_patch_count", 0),
    )
    largest_component_patch_count = int(
        evidence.get("largest_component_patch_count", 0),
    )

    # Each dimension independently signals anomaly.
    score_trigger = float(score) > decision_threshold
    peak_trigger = (
        float(evidence["peak_patch_score"]) > decision_threshold
        and largest_decision_component_patch_count >= max(1, int(config.min_peak_component_patch_count))
    )
    spatial_trigger = (
        int(evidence["strong_patch_count"]) >= int(config.min_strong_patch_count)
        and int(evidence["largest_component_patch_count"]) >= int(config.min_strong_component_count)
        and float(evidence["strong_patch_ratio"]) >= float(config.min_strong_patch_ratio)
        and float(evidence["largest_component_patch_ratio"]) >= float(config.min_strong_component_ratio)
    )
    critical_trigger = (
        float(score) > critical_score_threshold
        and float(evidence["peak_patch_score"]) > critical_peak_threshold
        and max(largest_component_patch_count, largest_decision_component_patch_count) >= component_min_patch_count
    )

    if critical_trigger:
        return True, "critical_rule"
    if score_trigger:
        return True, "score_rule"
    if peak_trigger:
        return True, "peak_rule"
    if spatial_trigger:
        return True, "spatial_rule"
    return False, "none"


def _analyze_patch_evidence(
    patch_map: np.ndarray,
    *,
    score: float,
    threshold: float,
    valid_patch_count: int,
    config: PatchCoreConfig,
) -> Dict[str, Union[float, int]]:
    """Extract patch statistics consumed by the final anomaly decision."""
    peak_patch_score = float(patch_map.max()) if patch_map.size > 0 else 0.0
    if patch_map.size == 0 or valid_patch_count <= 0:
        return {
            "peak_patch_score": peak_patch_score,
            "strong_patch_count": 0,
            "largest_component_patch_count": 0,
            "strong_patch_ratio": 0.0,
            "largest_component_patch_ratio": 0.0,
            "decision_patch_count": 0,
            "largest_decision_component_patch_count": 0,
            "decision_patch_ratio": 0.0,
            "largest_decision_component_patch_ratio": 0.0,
        }

    strong_patch_ratio = float(np.clip(config.strong_patch_score_ratio, 0.0, 1.0))
    strong_patch_floor = max(float(threshold), float(score)) * strong_patch_ratio
    strong_patch_mask = (patch_map >= strong_patch_floor).astype(np.uint8)
    strong_patch_count, largest_component_patch_count = _measure_patch_components(strong_patch_mask)

    decision_threshold = float(threshold) * _threshold_margin(config.decision_score_margin)
    decision_patch_mask = (patch_map >= decision_threshold).astype(np.uint8)
    decision_patch_count, largest_decision_component_patch_count = _measure_patch_components(
        decision_patch_mask,
    )

    return {
        "peak_patch_score": peak_patch_score,
        "strong_patch_count": strong_patch_count,
        "largest_component_patch_count": largest_component_patch_count,
        "strong_patch_ratio": float(strong_patch_count) / float(max(1, valid_patch_count)),
        "largest_component_patch_ratio": float(largest_component_patch_count) / float(max(1, valid_patch_count)),
        "decision_patch_count": decision_patch_count,
        "largest_decision_component_patch_count": largest_decision_component_patch_count,
        "decision_patch_ratio": float(decision_patch_count) / float(max(1, valid_patch_count)),
        "largest_decision_component_patch_ratio": float(largest_decision_component_patch_count)
        / float(max(1, valid_patch_count)),
    }


def _measure_patch_components(binary_mask: np.ndarray) -> Tuple[int, int]:
    """Return active patch count and largest connected component size."""
    patch_count = int(binary_mask.sum())
    if patch_count == 0:
        return 0, 0

    _, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    largest_component_patch_count = int(stats[1:, cv2.CC_STAT_AREA].max()) if len(stats) > 1 else 0
    return patch_count, largest_component_patch_count
