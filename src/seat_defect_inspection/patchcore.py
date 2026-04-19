"""简化版 PatchCore 服务。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .color_branch import ColorReferenceProfile
from .config import PatchCoreConfig
from .schemas import TextureAnomalyResult

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(slots=True)
class _PatchBatch:
    grid_shape: tuple[int, int]
    valid_indices: np.ndarray
    valid_patch_count: int
    total_patch_count: int


@dataclass(slots=True)
class LoadedModelBundle:
    """从磁盘加载的 PatchCore 模型与颜色分支配置。"""

    patchcore: "PatchCoreService"
    color_profile: ColorReferenceProfile | None = None


class PatchCoreService:
    """训练和推理轻量 PatchCore 异常检测器。"""

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

    def fit(
        self,
        samples: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> dict[str, float | int]:
        """使用正常 ROI 样本和掩膜训练模型。"""
        raw_embeddings: list[np.ndarray] = []
        image_scores: list[float] = []
        sample_count = 0

        for image, target_mask, ignore_mask in samples:
            embeddings, batch = extract_patch_embeddings(
                image,
                self.config,
                target_mask=target_mask,
                ignore_mask=ignore_mask,
            )
            if len(embeddings) == 0:
                continue
            raw_embeddings.append(embeddings)
            sample_count += 1

        if not raw_embeddings:
            raise ValueError("PatchCore 没有可用的有效训练样本")

        stacked = np.concatenate(raw_embeddings, axis=0)
        self.feature_mean = stacked.mean(axis=0)
        self.feature_std = stacked.std(axis=0) + 1e-6
        normalized = self._normalize(stacked)
        target_bank_size = min(
            len(normalized),
            max(64, min(self.config.max_memory, len(normalized) // 4 or 1)),
        )
        self.memory_bank = coreset_subsample(normalized, target_bank_size)

        for embeddings in raw_embeddings:
            score, _ = self.score_embeddings(self._normalize(embeddings))
            image_scores.append(score)

        score_array = np.asarray(image_scores, dtype=np.float32)
        self.threshold = max(
            float(np.quantile(score_array, self.config.threshold_quantile)),
            float(score_array.mean() + 3.0 * score_array.std()),
            float(score_array.max() * 1.1 + 1e-6),
        )
        return {
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
        """对单张 ROI 打分并返回热力图与 patch 统计信息。"""
        if self.memory_bank is None or self.feature_mean is None or self.feature_std is None or self.threshold is None:
            raise RuntimeError("PatchCore 模型尚未准备完成")

        embeddings, batch = extract_patch_embeddings(
            image,
            self.config,
            target_mask=target_mask,
            ignore_mask=ignore_mask,
        )
        heatmap = np.zeros(image.shape[:2], dtype=np.float32)
        valid_patch_ratio = (
            float(batch.valid_patch_count) / float(batch.total_patch_count)
            if batch.total_patch_count > 0
            else 0.0
        )
        if len(embeddings) == 0:
            return TextureAnomalyResult(
                score=0.0,
                threshold=float(self.threshold),
                is_anomaly=False,
                heatmap=heatmap,
                valid_patch_ratio=valid_patch_ratio,
                valid_patch_count=int(batch.valid_patch_count),
                total_patch_count=int(batch.total_patch_count),
            )

        normalized_embeddings = self._normalize(embeddings)
        score, patch_scores = self.score_embeddings(normalized_embeddings)

        full_patch_scores = np.zeros((batch.total_patch_count,), dtype=np.float32)
        full_patch_scores[batch.valid_indices] = patch_scores
        patch_map = full_patch_scores.reshape(batch.grid_shape)
        heatmap = cv2.resize(
            patch_map,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        heatmap = normalize_map(heatmap)
        heatmap *= (target_mask > 0).astype(np.float32)

        return TextureAnomalyResult(
            score=float(score),
            threshold=float(self.threshold),
            is_anomaly=bool(score > self.threshold),
            heatmap=heatmap,
            valid_patch_ratio=valid_patch_ratio,
            valid_patch_count=int(batch.valid_patch_count),
            total_patch_count=int(batch.total_patch_count),
        )

    def score_embeddings(self, embeddings: np.ndarray) -> tuple[float, np.ndarray]:
        patch_scores = min_distance_to_bank(embeddings, self.memory_bank)
        image_score = float(np.percentile(patch_scores, 99))
        return image_score, patch_scores

    def save(
        self,
        model_path: str | Path,
        color_profile: ColorReferenceProfile | None = None,
    ) -> None:
        """保存 PatchCore 模型及可选颜色分支配置。"""
        if self.memory_bank is None or self.feature_mean is None or self.feature_std is None or self.threshold is None:
            raise RuntimeError("当前没有可保存的模型，请先完成训练")

        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "image_size": self.config.image_size,
            "patch_size": self.config.patch_size,
            "stride": self.config.stride,
            "max_memory": self.config.max_memory,
            "threshold_quantile": self.config.threshold_quantile,
            "texture_input": self.config.texture_input,
            "min_target_coverage": self.config.min_target_coverage,
            "max_ignore_overlap": self.config.max_ignore_overlap,
            "min_valid_patch_ratio": self.config.min_valid_patch_ratio,
            "threshold": float(self.threshold),
        }
        np.savez_compressed(
            path,
            memory_bank=self.memory_bank,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            meta_json=np.array(json.dumps(meta)),
            color_profile_json=np.array(color_profile.to_json() if color_profile is not None else ""),
        )

    @classmethod
    def load_bundle(cls, model_path: str | Path) -> LoadedModelBundle:
        """从磁盘恢复模型包。"""
        saved = np.load(model_path, allow_pickle=False)
        meta = json.loads(saved["meta_json"].item())
        config = PatchCoreConfig(
            image_size=int(meta["image_size"]),
            patch_size=int(meta["patch_size"]),
            stride=int(meta["stride"]),
            max_memory=int(meta["max_memory"]),
            threshold_quantile=float(meta["threshold_quantile"]),
            texture_input=str(meta.get("texture_input", "lab_l")),
            min_target_coverage=float(meta.get("min_target_coverage", 0.8)),
            max_ignore_overlap=float(meta.get("max_ignore_overlap", 0.1)),
            min_valid_patch_ratio=float(meta.get("min_valid_patch_ratio", 0.65)),
        )
        patchcore = cls(
            config=config,
            memory_bank=saved["memory_bank"],
            feature_mean=saved["feature_mean"],
            feature_std=saved["feature_std"],
            threshold=float(meta["threshold"]),
        )
        color_profile_json = saved["color_profile_json"].item()
        color_profile = (
            ColorReferenceProfile.from_json(color_profile_json)
            if color_profile_json
            else None
        )
        return LoadedModelBundle(patchcore=patchcore, color_profile=color_profile)

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        return (embeddings - self.feature_mean) / self.feature_std


def list_images(folder: Path) -> list[Path]:
    """递归收集目录中的图像文件。"""
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def extract_patch_embeddings(
    image: np.ndarray,
    config: PatchCoreConfig,
    *,
    target_mask: np.ndarray | None = None,
    ignore_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, _PatchBatch]:
    """Extract valid patch features using target/ignore masks."""
    resized_image = cv2.resize(
        image,
        (config.image_size, config.image_size),
        interpolation=cv2.INTER_AREA,
    )
    resized_target = _resize_mask(target_mask, config.image_size)
    resized_ignore = _resize_mask(ignore_mask, config.image_size)

    features: list[np.ndarray] = []
    valid_indices: list[int] = []
    grid_rows = 0
    grid_cols = 0
    patch_index = 0

    texture_mode = config.texture_input.strip().lower()
    prepared = _prepare_feature_inputs(resized_image, texture_mode)

    for top in range(0, config.image_size - config.patch_size + 1, config.stride):
        grid_rows += 1
        current_cols = 0
        for left in range(0, config.image_size - config.patch_size + 1, config.stride):
            current_cols += 1
            target_coverage = float(
                resized_target[top : top + config.patch_size, left : left + config.patch_size].mean(),
            )
            ignore_overlap = float(
                resized_ignore[top : top + config.patch_size, left : left + config.patch_size].mean(),
            )
            if target_coverage >= config.min_target_coverage and ignore_overlap <= config.max_ignore_overlap:
                features.append(
                    _build_patch_feature(
                        prepared,
                        top=top,
                        left=left,
                        patch_size=config.patch_size,
                        texture_mode=texture_mode,
                    )
                )
                valid_indices.append(patch_index)
            patch_index += 1
        grid_cols = current_cols

    if features:
        embeddings = np.stack(features).astype(np.float32)
    else:
        embeddings = np.zeros((0, 1), dtype=np.float32)

    return embeddings, _PatchBatch(
        grid_shape=(grid_rows, grid_cols),
        valid_indices=np.asarray(valid_indices, dtype=np.int32),
        valid_patch_count=len(valid_indices),
        total_patch_count=patch_index,
    )


def _prepare_feature_inputs(image: np.ndarray, texture_mode: str) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))

    if texture_mode == "gray":
        return {
            "primary": gray,
            "grad": grad_mag,
            "lap": laplacian,
        }

    if texture_mode == "lab_l":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32) / 255.0
        grad_x = cv2.Sobel(l_channel, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(l_channel, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        laplacian = np.abs(cv2.Laplacian(l_channel, cv2.CV_32F))
        return {
            "primary": l_channel,
            "grad": grad_mag,
            "lap": laplacian,
        }

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    return {
        "rgb": rgb,
        "lab": lab,
        "gray": gray,
        "grad": grad_mag,
    }


def _build_patch_feature(
    prepared: dict[str, np.ndarray],
    *,
    top: int,
    left: int,
    patch_size: int,
    texture_mode: str,
) -> np.ndarray:
    if texture_mode in {"gray", "lab_l"}:
        primary_patch = prepared["primary"][top : top + patch_size, left : left + patch_size]
        grad_patch = prepared["grad"][top : top + patch_size, left : left + patch_size]
        lap_patch = prepared["lap"][top : top + patch_size, left : left + patch_size]
        thumb = cv2.resize(primary_patch, (8, 8), interpolation=cv2.INTER_AREA).reshape(-1)
        return np.concatenate(
            [
                np.asarray(
                    [
                        primary_patch.mean(),
                        primary_patch.std(),
                        grad_patch.mean(),
                        grad_patch.std(),
                        lap_patch.mean(),
                        lap_patch.std(),
                    ],
                    dtype=np.float32,
                ),
                thumb.astype(np.float32),
            ]
        )

    rgb_patch = prepared["rgb"][top : top + patch_size, left : left + patch_size]
    lab_patch = prepared["lab"][top : top + patch_size, left : left + patch_size]
    gray_patch = prepared["gray"][top : top + patch_size, left : left + patch_size]
    grad_patch = prepared["grad"][top : top + patch_size, left : left + patch_size]
    thumb = cv2.resize(gray_patch, (8, 8), interpolation=cv2.INTER_AREA).reshape(-1)
    return np.concatenate(
        [
            rgb_patch.mean(axis=(0, 1)),
            rgb_patch.std(axis=(0, 1)),
            lab_patch.mean(axis=(0, 1)),
            np.asarray(
                [
                    gray_patch.mean(),
                    gray_patch.std(),
                    grad_patch.mean(),
                    grad_patch.std(),
                ],
                dtype=np.float32,
            ),
            thumb.astype(np.float32),
        ]
    )


def _resize_mask(mask: np.ndarray | None, image_size: int) -> np.ndarray:
    if mask is None:
        return np.ones((image_size, image_size), dtype=np.float32)
    resized = cv2.resize(
        (mask > 0).astype(np.float32),
        (image_size, image_size),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(np.float32)


def coreset_subsample(embeddings: np.ndarray, max_points: int) -> np.ndarray:
    """Greedy coreset selection to keep diverse normal patches."""
    if len(embeddings) <= max_points:
        return embeddings

    rng = np.random.default_rng(42)
    first_index = int(rng.integers(0, len(embeddings)))
    chosen_indices = [first_index]
    min_distances = np.linalg.norm(embeddings - embeddings[first_index], axis=1)

    while len(chosen_indices) < max_points:
        next_index = int(np.argmax(min_distances))
        chosen_indices.append(next_index)
        next_distances = np.linalg.norm(embeddings - embeddings[next_index], axis=1)
        min_distances = np.minimum(min_distances, next_distances)

    return embeddings[np.asarray(chosen_indices, dtype=np.int32)]


def min_distance_to_bank(
    embeddings: np.ndarray,
    memory_bank: np.ndarray,
    chunk_size: int = 128,
) -> np.ndarray:
    """Compute the nearest-neighbor distance to the memory bank."""
    scores = []
    for start in range(0, len(embeddings), chunk_size):
        chunk = embeddings[start : start + chunk_size]
        distances = np.linalg.norm(chunk[:, None, :] - memory_bank[None, :, :], axis=2)
        scores.append(distances.min(axis=1))
    return np.concatenate(scores).astype(np.float32)


def normalize_map(heatmap: np.ndarray) -> np.ndarray:
    """Normalize a heatmap into the [0, 1] range."""
    minimum = float(heatmap.min())
    maximum = float(heatmap.max())
    if maximum - minimum < 1e-6:
        return np.zeros_like(heatmap, dtype=np.float32)
    return ((heatmap - minimum) / (maximum - minimum)).astype(np.float32)
