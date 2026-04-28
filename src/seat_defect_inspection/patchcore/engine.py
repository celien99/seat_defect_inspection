"""PatchCore model orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .color_branch import ColorReferenceProfile
from .features import _TorchPatchFeatureExtractor, extract_patch_embeddings
from .scoring import (
    _analyze_patch_evidence,
    _decide_patchcore_anomaly,
    _determine_memory_bank_size,
    _exclude_embedding_slice,
    _score_embeddings_leave_one_out,
    _threshold_margin,
    coreset_subsample_indices,
    min_distance_to_bank,
    normalize_map,
)
from ..config import PatchCoreConfig
from ..schemas import TextureAnomalyResult

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
_RUNTIME_DECISION_OVERRIDE_FIELDS = (
    "min_valid_patch_ratio",
    "decision_score_margin",
    "strong_patch_score_ratio",
    "min_strong_patch_count",
    "min_strong_component_count",
    "min_strong_patch_ratio",
    "min_strong_component_ratio",
    "critical_score_margin",
    "critical_peak_score_margin",
    "critical_min_component_patch_count",
)


@dataclass(slots=True)
class LoadedModelBundle:
    """Model bundle restored from disk."""

    patchcore: "PatchCoreService"
    color_profile: ColorReferenceProfile | None = None


class PatchCoreService:
    """Train and run PatchCore anomaly detection."""

    def __init__(
        self,
        config: PatchCoreConfig,
        memory_bank: np.ndarray | None = None,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        threshold: float | None = None,
    ) -> None:
        self.config = config
        self.memory_bank = memory_bank
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.threshold = threshold
        self._torch_feature_extractor: _TorchPatchFeatureExtractor | None = None

    def fit(
        self,
        samples: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> dict[str, float | int | str]:
        """Train the model from normal ROI samples."""
        raw_embeddings: list[np.ndarray] = []
        sample_count = 0

        for image, target_mask, ignore_mask in samples:
            embeddings, _ = extract_patch_embeddings(
                image,
                self.config,
                target_mask=target_mask,
                ignore_mask=ignore_mask,
                feature_extractor=self._get_torch_feature_extractor(),
            )
            if len(embeddings) == 0:
                continue
            raw_embeddings.append(embeddings.astype(np.float32))
            sample_count += 1

        if not raw_embeddings:
            raise ValueError("PatchCore has no valid training samples")

        stacked = np.concatenate(raw_embeddings, axis=0).astype(np.float32)
        self.feature_mean = stacked.mean(axis=0).astype(np.float32)
        self.feature_std = (stacked.std(axis=0) + 1e-6).astype(np.float32)
        normalized_samples = [self._normalize(embeddings) for embeddings in raw_embeddings]
        normalized = np.concatenate(normalized_samples, axis=0).astype(np.float32)
        target_bank_size = _determine_memory_bank_size(normalized, self.config)
        selected_indices = coreset_subsample_indices(normalized, target_bank_size)
        self.memory_bank = normalized[selected_indices]

        image_scores: list[float] = []
        sample_start = 0
        for embeddings in normalized_samples:
            sample_end = sample_start + len(embeddings)
            calibration_indices = selected_indices[
                (selected_indices < sample_start) | (selected_indices >= sample_end)
            ]
            if calibration_indices.size > 0:
                calibration_bank = normalized[calibration_indices]
                score, _ = self.score_embeddings(embeddings, memory_bank=calibration_bank)
            else:
                calibration_bank = _exclude_embedding_slice(normalized, sample_start, sample_end)
                if len(calibration_bank) > 0:
                    score, _ = self.score_embeddings(embeddings, memory_bank=calibration_bank)
                else:
                    score, _ = _score_embeddings_leave_one_out(embeddings)
            image_scores.append(score)
            sample_start = sample_end

        score_array = np.asarray(image_scores, dtype=np.float32)
        self.threshold = max(
            float(np.quantile(score_array, self.config.threshold_quantile)),
            float(score_array.mean() + 3.0 * score_array.std()),
            float(score_array.max() * 1.1 + 1e-6),
        )
        return {
            "backend": self.config.backend,
            "train_sample_count": int(sample_count),
            "patch_count": int(stacked.shape[0]),
            "memory_bank_size": int(self.memory_bank.shape[0]),
            "threshold": float(self.threshold),
        }

    def predict(
        self,
        image: np.ndarray,
        target_mask: np.ndarray,
        ignore_mask: np.ndarray,
    ) -> TextureAnomalyResult:
        """Score one ROI and return heatmap plus patch evidence."""
        if self.memory_bank is None or self.feature_mean is None or self.feature_std is None or self.threshold is None:
            raise RuntimeError("PatchCore model is not ready")

        embeddings, batch = extract_patch_embeddings(
            image,
            self.config,
            target_mask=target_mask,
            ignore_mask=ignore_mask,
            feature_extractor=self._get_torch_feature_extractor(),
        )
        heatmap = np.zeros(image.shape[:2], dtype=np.float32)
        valid_patch_ratio = (
            float(batch.valid_patch_count) / float(batch.total_patch_count)
            if batch.total_patch_count > 0
            else 0.0
        )
        if len(embeddings) == 0:
            decision_threshold = float(self.threshold) * _threshold_margin(self.config.decision_score_margin)
            return TextureAnomalyResult(
                score=0.0,
                threshold=float(self.threshold),
                is_anomaly=False,
                heatmap=heatmap,
                valid_patch_ratio=valid_patch_ratio,
                valid_patch_count=int(batch.valid_patch_count),
                total_patch_count=int(batch.total_patch_count),
                decision_threshold=decision_threshold,
                decision_mode="none",
            )

        normalized_embeddings = self._normalize(embeddings)
        score, patch_scores = self.score_embeddings(normalized_embeddings)

        full_patch_scores = np.zeros((batch.total_patch_count,), dtype=np.float32)
        full_patch_scores[batch.valid_indices] = patch_scores
        patch_map = full_patch_scores.reshape(batch.grid_shape)

        valid_patch_map = np.zeros((batch.total_patch_count,), dtype=np.float32)
        valid_patch_map[batch.valid_indices] = 1.0
        valid_patch_map = valid_patch_map.reshape(batch.grid_shape)

        heatmap = cv2.resize(
            patch_map,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        resized_patch_mask = cv2.resize(
            valid_patch_map,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        active_mask = np.logical_and(
            _as_binary_mask(target_mask, image.shape[:2]) > 0,
            resized_patch_mask > 0.5,
        )
        heatmap = normalize_map(heatmap, mask=active_mask)

        decision_threshold = float(self.threshold) * _threshold_margin(self.config.decision_score_margin)
        evidence = _analyze_patch_evidence(
            patch_map,
            score=float(score),
            threshold=float(self.threshold),
            valid_patch_count=int(batch.valid_patch_count),
            config=self.config,
        )
        is_anomaly, decision_mode = _decide_patchcore_anomaly(
            score=float(score),
            threshold=float(self.threshold),
            evidence=evidence,
            config=self.config,
        )

        return TextureAnomalyResult(
            score=float(score),
            threshold=float(self.threshold),
            is_anomaly=is_anomaly,
            heatmap=heatmap,
            valid_patch_ratio=valid_patch_ratio,
            valid_patch_count=int(batch.valid_patch_count),
            total_patch_count=int(batch.total_patch_count),
            decision_threshold=decision_threshold,
            peak_patch_score=float(evidence["peak_patch_score"]),
            strong_patch_count=int(evidence["strong_patch_count"]),
            largest_component_patch_count=int(evidence["largest_component_patch_count"]),
            strong_patch_ratio=float(evidence["strong_patch_ratio"]),
            largest_component_patch_ratio=float(evidence["largest_component_patch_ratio"]),
            decision_mode=decision_mode,
        )

    def score_embeddings(
        self,
        embeddings: np.ndarray,
        memory_bank: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        """Score an embedding batch against the active memory bank."""
        active_bank = self.memory_bank if memory_bank is None else memory_bank
        if active_bank is None or len(active_bank) == 0:
            raise RuntimeError("PatchCore memory bank is empty")
        patch_scores = min_distance_to_bank(embeddings, active_bank)
        image_score = float(np.percentile(patch_scores, 99))
        return image_score, patch_scores

    def save(
        self,
        model_path: str | Path,
        color_profile: ColorReferenceProfile | None = None,
        *,
        pipeline_signature: str | None = None,
        pipeline_context: dict[str, object] | None = None,
    ) -> None:
        """Persist the PatchCore model and optional color profile."""
        if self.memory_bank is None or self.feature_mean is None or self.feature_std is None or self.threshold is None:
            raise RuntimeError("No trained PatchCore model is available to save")

        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "backend": self.config.backend,
            "image_size": self.config.image_size,
            "patch_size": self.config.patch_size,
            "stride": self.config.stride,
            "max_memory": self.config.max_memory,
            "threshold_quantile": self.config.threshold_quantile,
            "texture_input": self.config.texture_input,
            "min_target_coverage": self.config.min_target_coverage,
            "max_ignore_overlap": self.config.max_ignore_overlap,
            "min_valid_patch_ratio": self.config.min_valid_patch_ratio,
            "decision_score_margin": self.config.decision_score_margin,
            "strong_patch_score_ratio": self.config.strong_patch_score_ratio,
            "min_strong_patch_count": self.config.min_strong_patch_count,
            "min_strong_component_count": self.config.min_strong_component_count,
            "min_strong_patch_ratio": self.config.min_strong_patch_ratio,
            "min_strong_component_ratio": self.config.min_strong_component_ratio,
            "critical_score_margin": self.config.critical_score_margin,
            "critical_peak_score_margin": self.config.critical_peak_score_margin,
            "critical_min_component_patch_count": self.config.critical_min_component_patch_count,
            "backbone_name": self.config.backbone_name,
            "feature_layers": list(self.config.feature_layers),
            "backbone_pretrained": self.config.backbone_pretrained,
            "backbone_weights_path": self.config.backbone_weights_path,
            "backbone_device": self.config.backbone_device,
            "feature_pool_kernel_size": self.config.feature_pool_kernel_size,
            "coreset_sampling_ratio": self.config.coreset_sampling_ratio,
            "threshold": float(self.threshold),
        }
        if pipeline_signature is not None:
            meta["pipeline_signature"] = pipeline_signature
        if pipeline_context is not None:
            meta["pipeline_context"] = pipeline_context
        np.savez_compressed(
            path,
            memory_bank=self.memory_bank.astype(np.float32),
            feature_mean=self.feature_mean.astype(np.float32),
            feature_std=self.feature_std.astype(np.float32),
            meta_json=np.array(json.dumps(meta)),
            color_profile_json=np.array(color_profile.to_json() if color_profile is not None else ""),
        )

    @classmethod
    def load_bundle(
        cls,
        model_path: str | Path,
        runtime_config: PatchCoreConfig | None = None,
        expected_pipeline_signature: str | None = None,
    ) -> LoadedModelBundle:
        """Restore a model bundle from disk."""
        saved = np.load(model_path, allow_pickle=False)
        meta = json.loads(saved["meta_json"].item())
        saved_pipeline_signature = meta.get("pipeline_signature")
        if expected_pipeline_signature is not None:
            if not saved_pipeline_signature:
                raise RuntimeError(
                    "PatchCore model is missing the upstream pipeline signature. "
                    "Please retrain the model before running inspection.",
                )
            if str(saved_pipeline_signature) != str(expected_pipeline_signature):
                raise RuntimeError(
                    "PatchCore model no longer matches the current preprocess/ROI pipeline. "
                    "Please retrain the model before running inspection.",
                )
        trained_config = PatchCoreConfig(
            backend=str(meta.get("backend", "handcrafted")),
            image_size=int(meta["image_size"]),
            patch_size=int(meta["patch_size"]),
            stride=int(meta["stride"]),
            max_memory=int(meta["max_memory"]),
            threshold_quantile=float(meta["threshold_quantile"]),
            texture_input=str(meta.get("texture_input", "lab_l")),
            min_target_coverage=float(meta.get("min_target_coverage", 0.8)),
            max_ignore_overlap=float(meta.get("max_ignore_overlap", 0.1)),
            min_valid_patch_ratio=float(meta.get("min_valid_patch_ratio", 0.65)),
            decision_score_margin=float(meta.get("decision_score_margin", 1.08)),
            strong_patch_score_ratio=float(meta.get("strong_patch_score_ratio", 0.9)),
            min_strong_patch_count=int(meta.get("min_strong_patch_count", 3)),
            min_strong_component_count=int(meta.get("min_strong_component_count", 2)),
            min_strong_patch_ratio=float(meta.get("min_strong_patch_ratio", 0.015)),
            min_strong_component_ratio=float(meta.get("min_strong_component_ratio", 0.01)),
            critical_score_margin=float(meta.get("critical_score_margin", 1.35)),
            critical_peak_score_margin=float(meta.get("critical_peak_score_margin", 1.45)),
            critical_min_component_patch_count=int(meta.get("critical_min_component_patch_count", 2)),
            backbone_name=str(meta.get("backbone_name", "wide_resnet50_2")),
            feature_layers=list(meta.get("feature_layers", ["layer2", "layer3"])),
            backbone_pretrained=bool(meta.get("backbone_pretrained", False)),
            backbone_weights_path=meta.get("backbone_weights_path"),
            backbone_device=str(meta.get("backbone_device", "cpu")),
            feature_pool_kernel_size=int(meta.get("feature_pool_kernel_size", 3)),
            coreset_sampling_ratio=float(meta.get("coreset_sampling_ratio", 0.1)),
        )
        config = _apply_runtime_patchcore_overrides(trained_config, runtime_config)
        patchcore = cls(
            config=config,
            memory_bank=saved["memory_bank"].astype(np.float32),
            feature_mean=saved["feature_mean"].astype(np.float32),
            feature_std=saved["feature_std"].astype(np.float32),
            threshold=float(meta["threshold"]),
        )
        color_profile_json = saved["color_profile_json"].item() if "color_profile_json" in saved.files else ""
        color_profile = (
            ColorReferenceProfile.from_json(color_profile_json)
            if color_profile_json
            else None
        )
        return LoadedModelBundle(patchcore=patchcore, color_profile=color_profile)

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply training-time feature normalization."""
        return ((embeddings - self.feature_mean) / self.feature_std).astype(np.float32)

    def _get_torch_feature_extractor(self) -> "_TorchPatchFeatureExtractor | None":
        """Lazy-init the full-backend torch feature extractor."""
        if self.config.backend.strip().lower() != "full":
            return None
        if self._torch_feature_extractor is None:
            self._torch_feature_extractor = _TorchPatchFeatureExtractor(self.config)
        return self._torch_feature_extractor


def _apply_runtime_patchcore_overrides(
    trained_config: PatchCoreConfig,
    runtime_config: PatchCoreConfig | None,
) -> PatchCoreConfig:
    """Apply runtime-only overrides while preserving the trained structure."""
    if runtime_config is None:
        return trained_config
    trained_backend = trained_config.backend.strip().lower()
    runtime_backend = runtime_config.backend.strip().lower()
    if trained_backend != runtime_backend:
        raise RuntimeError(
            "PatchCore model backend does not match the current runtime config. "
            f"model backend={trained_config.backend}, runtime backend={runtime_config.backend}. "
            "Please retrain the PatchCore model.",
        )

    overrides: dict[str, float | int] = {
        # Runtime may tighten patch validity constraints, but should not loosen them
        # beyond the structure used to build the memory bank.
        "min_target_coverage": max(
            float(trained_config.min_target_coverage),
            float(runtime_config.min_target_coverage),
        ),
        "max_ignore_overlap": min(
            float(trained_config.max_ignore_overlap),
            float(runtime_config.max_ignore_overlap),
        ),
    }
    overrides.update(
        {
            field_name: getattr(runtime_config, field_name)
            for field_name in _RUNTIME_DECISION_OVERRIDE_FIELDS
        }
    )
    return replace(trained_config, **overrides)


def list_images(folder: Path) -> list[Path]:
    """Recursively collect image files under a folder."""
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def _as_binary_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Normalize a binary mask or BGRA transparent mask to the requested shape."""
    array = np.asarray(mask)
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = array[:, :, 3]
        else:
            array = np.any(array > 0, axis=2)
    binary = (array > 0).astype(np.uint8)
    if binary.shape != shape:
        binary = cv2.resize(
            binary,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return binary
