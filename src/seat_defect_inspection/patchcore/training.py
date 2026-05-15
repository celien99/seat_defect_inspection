"""PatchCore 训练与模型保存能力。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from seat_defect_core.config import PatchCoreConfig
from seat_defect_core.patchcore.color_branch import ColorReferenceProfile
from seat_defect_core.patchcore.engine import PatchCoreService
from seat_defect_core.patchcore.features import extract_patch_embeddings
from seat_defect_core.patchcore.scoring import (
    _determine_memory_bank_size,
    _exclude_embedding_slice,
    _score_embeddings_leave_one_out,
    coreset_subsample_indices,
)

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


class PatchCoreTrainer(PatchCoreService):
    """训练 PatchCore 模型并保存为运行时可加载的模型包。"""

    def fit(
        self,
        samples: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> dict[str, float | int | str]:
        """从正常 ROI 样本训练 memory bank 和异常阈值。"""
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
            raise ValueError("PatchCore 没有可用的有效训练样本")

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
        upper_quantile = float(
            np.clip(self.config.training_threshold_upper_quantile, 0.9, 1.0)
        )
        self.threshold = max(
            float(np.quantile(score_array, self.config.threshold_quantile)),
            float(score_array.mean() + 3.0 * score_array.std()),
            float(np.quantile(score_array, upper_quantile)),
        )
        return {
            "backend": self.config.backend,
            "train_sample_count": int(sample_count),
            "patch_count": int(stacked.shape[0]),
            "memory_bank_size": int(self.memory_bank.shape[0]),
            "threshold": float(self.threshold),
        }

    def save(
        self,
        model_path: str | Path,
        color_profile: ColorReferenceProfile | None = None,
        *,
        pipeline_signature: str | None = None,
        pipeline_context: dict[str, object] | None = None,
    ) -> None:
        """保存 PatchCore 模型和可选颜色参考分布。"""
        if (
            self.memory_bank is None
            or self.feature_mean is None
            or self.feature_std is None
            or self.threshold is None
        ):
            raise RuntimeError("No trained PatchCore model is available to save")

        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = _build_model_meta(
            self.config,
            threshold=float(self.threshold),
            pipeline_signature=pipeline_signature,
            pipeline_context=pipeline_context,
        )
        np.savez_compressed(
            path,
            memory_bank=self.memory_bank.astype(np.float32),
            feature_mean=self.feature_mean.astype(np.float32),
            feature_std=self.feature_std.astype(np.float32),
            meta_json=np.array(json.dumps(meta)),
            color_profile_json=np.array(
                color_profile.to_json() if color_profile is not None else "",
            ),
        )


def list_images(folder: Path) -> list[Path]:
    """递归收集目录中的图片文件。"""
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def _build_model_meta(
    config: PatchCoreConfig,
    *,
    threshold: float,
    pipeline_signature: str | None,
    pipeline_context: dict[str, object] | None,
) -> dict[str, object]:
    """构造运行时模型包 metadata。"""
    meta: dict[str, object] = {
        "backend": config.backend,
        "image_size": config.image_size,
        "patch_size": config.patch_size,
        "stride": config.stride,
        "max_memory": config.max_memory,
        "threshold_quantile": config.threshold_quantile,
        "training_threshold_upper_quantile": config.training_threshold_upper_quantile,
        "texture_input": config.texture_input,
        "min_target_coverage": config.min_target_coverage,
        "max_ignore_overlap": config.max_ignore_overlap,
        "min_valid_patch_ratio": config.min_valid_patch_ratio,
        "decision_score_margin": config.decision_score_margin,
        "strong_patch_score_ratio": config.strong_patch_score_ratio,
        "min_strong_patch_count": config.min_strong_patch_count,
        "min_strong_component_count": config.min_strong_component_count,
        "min_strong_patch_ratio": config.min_strong_patch_ratio,
        "min_strong_component_ratio": config.min_strong_component_ratio,
        "critical_score_margin": config.critical_score_margin,
        "critical_peak_score_margin": config.critical_peak_score_margin,
        "critical_min_component_patch_count": config.critical_min_component_patch_count,
        "min_peak_component_patch_count": config.min_peak_component_patch_count,
        "backbone_name": config.backbone_name,
        "feature_layers": list(config.feature_layers),
        "backbone_pretrained": config.backbone_pretrained,
        "backbone_weights_path": config.backbone_weights_path,
        "backbone_device": config.backbone_device,
        "feature_pool_kernel_size": config.feature_pool_kernel_size,
        "coreset_sampling_ratio": config.coreset_sampling_ratio,
        "threshold": threshold,
    }
    if pipeline_signature is not None:
        meta["pipeline_signature"] = pipeline_signature
    if pipeline_context is not None:
        meta["pipeline_context"] = pipeline_context
    return meta
