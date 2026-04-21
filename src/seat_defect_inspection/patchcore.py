"""PatchCore anomaly detection service with full and fallback backends."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as torch_f
    from torchvision.models import (
        ResNet18_Weights,
        ResNet50_Weights,
        Wide_ResNet50_2_Weights,
        resnet18,
        resnet50,
        wide_resnet50_2,
    )
except ImportError:  # pragma: no cover - fallback for minimal runtime
    torch = None
    torch_f = None
    ResNet18_Weights = None
    ResNet50_Weights = None
    Wide_ResNet50_2_Weights = None
    resnet18 = None
    resnet50 = None
    wide_resnet50_2 = None

from .color_branch import ColorReferenceProfile
from .config import PatchCoreConfig
from .schemas import TextureAnomalyResult

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


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
    """训练和推理 PatchCore 异常检测器。"""

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
        """使用正常 ROI 样本和掩膜训练模型。"""
        raw_embeddings: list[np.ndarray] = []
        image_scores: list[float] = []
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
            raw_embeddings.append(embeddings)
            sample_count += 1

        if not raw_embeddings:
            raise ValueError("PatchCore 没有可用的有效训练样本")

        stacked = np.concatenate(raw_embeddings, axis=0).astype(np.float32)
        self.feature_mean = stacked.mean(axis=0).astype(np.float32)
        self.feature_std = (stacked.std(axis=0) + 1e-6).astype(np.float32)
        normalized = self._normalize(stacked)
        target_bank_size = _determine_memory_bank_size(
            normalized,
            self.config,
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
        """对单张 ROI 打分并返回热力图与 patch 统计信息。"""
        if self.memory_bank is None or self.feature_mean is None or self.feature_std is None or self.threshold is None:
            raise RuntimeError("PatchCore 模型尚未准备完成")

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
            decision_threshold = float(self.threshold) * _positive_margin(self.config.decision_score_margin)
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
        heatmap = cv2.resize(
            patch_map,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        heatmap = normalize_map(heatmap)
        heatmap *= (target_mask > 0).astype(np.float32)
        decision_threshold = float(self.threshold) * _positive_margin(self.config.decision_score_margin)
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
        np.savez_compressed(
            path,
            memory_bank=self.memory_bank.astype(np.float32),
            feature_mean=self.feature_mean.astype(np.float32),
            feature_std=self.feature_std.astype(np.float32),
            meta_json=np.array(json.dumps(meta)),
            color_profile_json=np.array(color_profile.to_json() if color_profile is not None else ""),
        )

    @classmethod
    def load_bundle(cls, model_path: str | Path) -> LoadedModelBundle:
        """从磁盘恢复模型包。"""
        saved = np.load(model_path, allow_pickle=False)
        meta = json.loads(saved["meta_json"].item())
        config = PatchCoreConfig(
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
        return ((embeddings - self.feature_mean) / self.feature_std).astype(np.float32)

    def _get_torch_feature_extractor(self) -> "_TorchPatchFeatureExtractor | None":
        if self.config.backend.strip().lower() != "full":
            return None
        if self._torch_feature_extractor is None:
            self._torch_feature_extractor = _TorchPatchFeatureExtractor(self.config)
        return self._torch_feature_extractor


def list_images(folder: Path) -> list[Path]:
    """递归收集目录中的图像文件。"""
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def extract_patch_embeddings(
    image: np.ndarray,
    config: PatchCoreConfig,
    *,
    target_mask: np.ndarray | None = None,
    ignore_mask: np.ndarray | None = None,
    feature_extractor: "_TorchPatchFeatureExtractor | None" = None,
) -> tuple[np.ndarray, _PatchBatch]:
    """Extract valid patch features using the configured backend."""
    backend = config.backend.strip().lower()
    if backend == "full":
        extractor = feature_extractor or _TorchPatchFeatureExtractor(config)
        return extractor.extract(
            image,
            target_mask=target_mask,
            ignore_mask=ignore_mask,
        )
    if backend != "handcrafted":
        raise ValueError(f"Unsupported PatchCore backend: {config.backend}")
    return extract_handcrafted_patch_embeddings(
        image,
        config,
        target_mask=target_mask,
        ignore_mask=ignore_mask,
    )


def extract_handcrafted_patch_embeddings(
    image: np.ndarray,
    config: PatchCoreConfig,
    *,
    target_mask: np.ndarray | None = None,
    ignore_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, _PatchBatch]:
    """Extract valid handcrafted patch features using target/ignore masks."""
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


class _TorchPatchFeatureExtractor:
    """CNN 特征提取器，用于完整 PatchCore 后端。"""

    def __init__(self, config: PatchCoreConfig) -> None:
        if torch is None or torch_f is None:
            raise RuntimeError(
                "完整 PatchCore 依赖 torch 和 torchvision，请先安装可用运行环境，"
                "或把 patchcore.backend 切换为 handcrafted。"
            )
        self.config = config
        self.device = _resolve_torch_device(config.backbone_device)
        self.model = _load_torch_backbone(config)
        self.model.to(self.device)
        self.model.eval()
        self.layer_names = [layer.strip() for layer in config.feature_layers if layer.strip()]
        if not self.layer_names:
            raise ValueError("完整 PatchCore 至少需要一个 feature_layers")
        self._features: dict[str, Any] = {}
        self._handles = [
            _resolve_submodule(self.model, layer_name).register_forward_hook(self._make_hook(layer_name))
            for layer_name in self.layer_names
        ]

    def extract(
        self,
        image: np.ndarray,
        *,
        target_mask: np.ndarray | None = None,
        ignore_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, _PatchBatch]:
        resized_image = cv2.resize(
            image,
            (self.config.image_size, self.config.image_size),
            interpolation=cv2.INTER_AREA,
        )
        input_tensor = _prepare_torch_input(resized_image, self.config).to(self.device)
        with torch.inference_mode():
            self._features.clear()
            _ = self.model(input_tensor)

        feature_maps = [self._features[layer_name] for layer_name in self.layer_names]
        embedding_map = _generate_deep_embedding_map(feature_maps, self.config)
        _, _, grid_rows, grid_cols = embedding_map.shape

        target_grid = _resize_mask_to_grid(
            target_mask,
            grid_rows,
            grid_cols,
            interpolation=cv2.INTER_AREA,
        )
        ignore_grid = _resize_mask_to_grid(
            ignore_mask,
            grid_rows,
            grid_cols,
            interpolation=cv2.INTER_AREA,
        )
        valid_mask = np.logical_and(
            target_grid >= float(self.config.min_target_coverage),
            ignore_grid <= float(self.config.max_ignore_overlap),
        )
        valid_indices = np.flatnonzero(valid_mask.reshape(-1)).astype(np.int32)
        embedding_array = embedding_map[0].detach().cpu().numpy()
        flattened = np.moveaxis(embedding_array, 0, -1).reshape(-1, embedding_array.shape[0])
        embeddings = flattened[valid_indices].astype(np.float32)

        return embeddings, _PatchBatch(
            grid_shape=(grid_rows, grid_cols),
            valid_indices=valid_indices,
            valid_patch_count=int(valid_indices.size),
            total_patch_count=int(grid_rows * grid_cols),
        )

    def _make_hook(self, layer_name: str):
        def _hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if torch.is_tensor(output):
                self._features[layer_name] = output.detach()
                return
            raise TypeError(f"Layer `{layer_name}` output is not a tensor")

        return _hook


def _prepare_torch_input(image: np.ndarray, config: PatchCoreConfig) -> Any:
    texture_mode = config.texture_input.strip().lower()
    if texture_mode == "gray":
        primary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        rgb = np.repeat(primary[:, :, None], 3, axis=2)
    elif texture_mode == "lab_l":
        primary = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32) / 255.0
        rgb = np.repeat(primary[:, :, None], 3, axis=2)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(np.transpose(normalized, (2, 0, 1))).unsqueeze(0).float()
    return tensor


def _generate_deep_embedding_map(feature_maps: list[Any], config: PatchCoreConfig) -> Any:
    if torch_f is None:
        raise RuntimeError("torch.nn.functional 不可用，无法生成完整 PatchCore embedding")

    pooled_features: list[Any] = []
    kernel_size = max(1, int(config.feature_pool_kernel_size))
    padding = kernel_size // 2
    for feature_map in feature_maps:
        pooled = torch_f.avg_pool2d(
            feature_map,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        pooled_features.append(pooled)

    target_height = max(item.shape[2] for item in pooled_features)
    target_width = max(item.shape[3] for item in pooled_features)
    aligned = [
        torch_f.interpolate(
            item,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        if item.shape[2:] != (target_height, target_width)
        else item
        for item in pooled_features
    ]
    return torch.cat(aligned, dim=1)


def _resolve_torch_device(requested_device: str) -> Any:
    normalized = requested_device.strip().lower()
    if normalized.startswith("cuda"):
        if torch is not None and torch.cuda.is_available():
            return torch.device(requested_device)
        return torch.device("cpu")
    if normalized == "mps":
        if torch is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cpu")


def _load_torch_backbone(config: PatchCoreConfig) -> Any:
    builder, default_weights = _resolve_backbone_builder(config.backbone_name)
    if config.backbone_weights_path:
        model = builder(weights=None)
        state_dict = torch.load(config.backbone_weights_path, map_location="cpu")
        state_dict = _unwrap_state_dict(state_dict)
        normalized_state_dict = _normalize_state_dict_keys(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(normalized_state_dict, strict=False)
        critical_missing = [key for key in missing_keys if not key.startswith("fc.")]
        if critical_missing:
            raise RuntimeError(
                "backbone_weights_path 无法正确恢复 PatchCore backbone，"
                f"缺少关键参数数量: {len(critical_missing)}"
            )
        if unexpected_keys:
            raise RuntimeError(
                "backbone_weights_path 包含无法匹配的参数，"
                f"请确认权重与 backbone_name 对齐，异常参数数量: {len(unexpected_keys)}"
            )
        return model

    if not config.backbone_pretrained:
        raise RuntimeError(
            "完整 PatchCore 不能使用随机初始化 backbone。"
            " 请设置 patchcore.backbone_pretrained=true，"
            "或配置 patchcore.backbone_weights_path 指向本地预训练权重，"
            "或把 patchcore.backend 切换为 handcrafted。"
        )

    torch.hub.set_dir(str(Path.cwd() / ".torch_cache"))
    try:
        return builder(weights=default_weights)
    except Exception as exc:  # pragma: no cover - depends on local cache/network
        raise RuntimeError(
            "完整 PatchCore 已启用 backbone_pretrained=True，但当前环境无法加载预训练权重。"
            "可选方案：1) 配置 patchcore.backbone_weights_path 指向本地权重；"
            "2) 先把 torchvision 权重缓存到项目 .torch_cache；"
            "3) 临时将 patchcore.backend 切换为 handcrafted 做功能联调。"
        ) from exc


def _resolve_backbone_builder(backbone_name: str) -> tuple[Any, Any]:
    normalized = backbone_name.strip().lower()
    builders = {
        "resnet18": (resnet18, ResNet18_Weights.DEFAULT if ResNet18_Weights is not None else None),
        "resnet50": (resnet50, ResNet50_Weights.DEFAULT if ResNet50_Weights is not None else None),
        "wide_resnet50_2": (
            wide_resnet50_2,
            Wide_ResNet50_2_Weights.DEFAULT if Wide_ResNet50_2_Weights is not None else None,
        ),
    }
    builder = builders.get(normalized)
    if builder is None or builder[0] is None:
        supported = ", ".join(sorted(builders))
        raise ValueError(f"Unsupported PatchCore backbone `{backbone_name}`，当前支持：{supported}")
    return builder


def _unwrap_state_dict(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("backbone_weights_path 必须加载到 state_dict 字典")
    for candidate_key in ("state_dict", "model", "backbone", "network"):
        nested = payload.get(candidate_key)
        if isinstance(nested, dict):
            return nested
    return payload


def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    prefixes = ("module.", "model.", "backbone.", "network.")
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
                    changed = True
        normalized[normalized_key] = value
    return normalized


def _resolve_submodule(model: Any, layer_name: str) -> Any:
    current = model
    for part in layer_name.split("."):
        if not hasattr(current, part):
            raise ValueError(f"PatchCore feature layer `{layer_name}` 在 backbone 中不存在")
        current = getattr(current, part)
    return current


def _resize_mask(mask: np.ndarray | None, image_size: int) -> np.ndarray:
    if mask is None:
        return np.ones((image_size, image_size), dtype=np.float32)
    resized = cv2.resize(
        (mask > 0).astype(np.float32),
        (image_size, image_size),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(np.float32)


def _resize_mask_to_grid(
    mask: np.ndarray | None,
    grid_rows: int,
    grid_cols: int,
    *,
    interpolation: int,
) -> np.ndarray:
    if mask is None:
        return np.ones((grid_rows, grid_cols), dtype=np.float32)
    resized = cv2.resize(
        (mask > 0).astype(np.float32),
        (grid_cols, grid_rows),
        interpolation=interpolation,
    )
    return resized.astype(np.float32)


def _determine_memory_bank_size(
    embeddings: np.ndarray,
    config: PatchCoreConfig,
) -> int:
    ratio = float(np.clip(config.coreset_sampling_ratio, 0.0, 1.0))
    if ratio > 0.0:
        ratio_target = max(1, int(round(len(embeddings) * ratio)))
    else:
        ratio_target = max(64, min(config.max_memory, len(embeddings) // 4 or 1))
    return min(len(embeddings), max(1, min(config.max_memory, ratio_target)))


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


def _positive_margin(value: float) -> float:
    return max(float(value), 1e-6)


def _decide_patchcore_anomaly(
    *,
    score: float,
    threshold: float,
    evidence: dict[str, float | int],
    config: PatchCoreConfig,
) -> tuple[bool, str]:
    """Combine balanced evidence rule with a strong-defect fast path."""
    decision_threshold = float(threshold) * _positive_margin(config.decision_score_margin)
    critical_score_threshold = float(threshold) * _positive_margin(config.critical_score_margin)
    critical_peak_threshold = float(threshold) * _positive_margin(config.critical_peak_score_margin)

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

    if normal_trigger and critical_trigger:
        return True, "normal_and_critical"
    if critical_trigger:
        return True, "critical_rule"
    if normal_trigger:
        return True, "normal_rule"
    return False, "none"


def _analyze_patch_evidence(
    patch_map: np.ndarray,
    *,
    score: float,
    threshold: float,
    valid_patch_count: int,
    config: PatchCoreConfig,
) -> dict[str, float | int]:
    peak_patch_score = float(patch_map.max()) if patch_map.size > 0 else 0.0
    if patch_map.size == 0 or valid_patch_count <= 0:
        return {
            "peak_patch_score": peak_patch_score,
            "strong_patch_count": 0,
            "largest_component_patch_count": 0,
            "strong_patch_ratio": 0.0,
            "largest_component_patch_ratio": 0.0,
        }

    strong_patch_floor = max(
        float(threshold),
        float(score) * float(np.clip(config.strong_patch_score_ratio, 0.0, 1.0)),
    )
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
