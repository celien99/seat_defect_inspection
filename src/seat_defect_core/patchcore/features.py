"""PatchCore 特征提取细节。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as torch_f
except ImportError:  # pragma: no cover - fallback for minimal runtime
    torch = None
    torch_f = None

try:
    from torchvision.models import (
        ResNet18_Weights,
        ResNet50_Weights,
        Wide_ResNet50_2_Weights,
        resnet18,
        resnet50,
        wide_resnet50_2,
    )
except ImportError:  # pragma: no cover - fallback for minimal runtime
    ResNet18_Weights = None
    ResNet50_Weights = None
    Wide_ResNet50_2_Weights = None
    resnet18 = None
    resnet50 = None
    wide_resnet50_2 = None

try:
    from torchvision.models import (
        ViT_B_16_Weights,
        ViT_B_32_Weights,
        ViT_L_16_Weights,
        ViT_L_32_Weights,
        vit_b_16,
        vit_b_32,
        vit_l_16,
        vit_l_32,
    )
except ImportError:  # pragma: no cover - depends on torchvision version
    ViT_B_16_Weights = None
    ViT_B_32_Weights = None
    ViT_L_16_Weights = None
    ViT_L_32_Weights = None
    vit_b_16 = None
    vit_b_32 = None
    vit_l_16 = None
    vit_l_32 = None

from ..config import PatchCoreConfig

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(slots=True)
class _PatchBatch:
    """记录当前图像 patch 网格与有效 patch 统计。"""

    grid_shape: tuple[int, int]
    valid_indices: np.ndarray
    valid_patch_count: int
    total_patch_count: int


def extract_patch_embeddings(
    image: np.ndarray,
    config: PatchCoreConfig,
    *,
    target_mask: np.ndarray | None = None,
    ignore_mask: np.ndarray | None = None,
    feature_extractor: "_TorchPatchFeatureExtractor | _TorchTransformerPatchFeatureExtractor | None" = None,
) -> tuple[np.ndarray, _PatchBatch]:
    """按配置后端提取有效 patch embedding。"""
    backend = config.backend.strip().lower()
    if backend == "full":
        extractor = feature_extractor or _TorchPatchFeatureExtractor(config)
        return extractor.extract(
            image,
            target_mask=target_mask,
            ignore_mask=ignore_mask,
        )
    if backend == "transformer":
        extractor = feature_extractor or _TorchTransformerPatchFeatureExtractor(config)
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
    """用掩膜筛出有效 patch，并提取轻量手工纹理特征。"""
    feature_image = _prepare_feature_image(image, target_mask=target_mask)
    resized_image = cv2.resize(
        feature_image,
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


def _prepare_feature_inputs(image: np.ndarray, texture_mode: str) -> np.ndarray:
    """把纹理输入统一成更适合手工特征的单通道图。"""
    if texture_mode == "gray":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if texture_mode == "lab_l":
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    if texture_mode == "hsv_v":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    if texture_mode == "ycrcb_y":
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _build_patch_feature(
    prepared: np.ndarray,
    *,
    top: int,
    left: int,
    patch_size: int,
    texture_mode: str,
) -> np.ndarray:
    """拼装单个 patch 的手工统计特征。"""
    patch = prepared[top : top + patch_size, left : left + patch_size].astype(np.float32)
    if patch.size == 0:
        raise ValueError("Patch feature 不能为空")

    normalized_patch = patch / 255.0
    grad_x = cv2.Sobel(normalized_patch, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(normalized_patch, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    laplacian = cv2.Laplacian(normalized_patch, cv2.CV_32F, ksize=3)
    thumb = cv2.resize(normalized_patch, (4, 4), interpolation=cv2.INTER_AREA).reshape(-1)

    if texture_mode == "lab_l":
        channel_stats = np.asarray(
            [
                normalized_patch.mean(),
                normalized_patch.std(),
                np.percentile(normalized_patch, 10),
                np.percentile(normalized_patch, 90),
            ],
            dtype=np.float32,
        )
    else:
        channel_stats = np.asarray(
            [
                normalized_patch.mean(),
                normalized_patch.std(),
            ],
            dtype=np.float32,
        )

    return np.concatenate(
        [
            channel_stats,
            np.asarray(
                [
                    grad_mag.mean(),
                    grad_mag.std(),
                    laplacian.var(),
                    float(np.abs(laplacian).mean()),
                    grad_x.std(),
                    grad_y.std(),
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
                "完整 PatchCore 依赖 torch 和 torchvision，请先安装可用运行环境。"
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
        """提取完整 PatchCore 使用的深度 embedding。"""
        feature_image = _prepare_feature_image(image, target_mask=target_mask)
        resized_image = cv2.resize(
            feature_image,
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
        """保存中间层输出，供 embedding 拼接使用。"""

        def _hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if torch.is_tensor(output):
                self._features[layer_name] = output.detach()
                return
            raise TypeError(f"Layer `{layer_name}` output is not a tensor")

        return _hook


class _TorchTransformerPatchFeatureExtractor:
    """Vision Transformer 特征提取器，用于 Transformer PatchCore 后端。"""

    def __init__(self, config: PatchCoreConfig) -> None:
        if torch is None:
            raise RuntimeError(
                "Transformer PatchCore 依赖 torch 和 torchvision，请先安装可用运行环境。"
            )
        self.config = config
        self.device = _resolve_torch_device(config.backbone_device)
        self.model = _load_torch_transformer_backbone(config)
        self.model.to(self.device)
        self.model.eval()

    def extract(
        self,
        image: np.ndarray,
        *,
        target_mask: np.ndarray | None = None,
        ignore_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, _PatchBatch]:
        """提取 ViT patch token embedding，并按 mask 筛出有效 token。"""
        feature_image = _prepare_feature_image(image, target_mask=target_mask)
        resized_image = cv2.resize(
            feature_image,
            (self.config.image_size, self.config.image_size),
            interpolation=cv2.INTER_AREA,
        )
        input_tensor = _prepare_torch_input(resized_image, self.config).to(self.device)
        with torch.inference_mode():
            token_embeddings = _extract_vit_patch_tokens(self.model, input_tensor)

        token_array = token_embeddings[0].detach().cpu().numpy().astype(np.float32)
        token_count = int(token_array.shape[0])
        grid_rows, grid_cols = _resolve_square_token_grid(token_count)

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
        embeddings = token_array[valid_indices].astype(np.float32)

        return embeddings, _PatchBatch(
            grid_shape=(grid_rows, grid_cols),
            valid_indices=valid_indices,
            valid_patch_count=int(valid_indices.size),
            total_patch_count=int(token_count),
        )


def _prepare_torch_input(image: np.ndarray, config: PatchCoreConfig) -> Any:
    """把 ROI 图像转成 torchvision backbone 可直接消费的输入。"""
    image = _prepare_feature_image(image)
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
    """对多层特征图做池化、对齐并拼成统一 embedding map。"""
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
    """尽量尊重配置，但在设备不可用时自动回退到 CPU。"""
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
    """加载完整 PatchCore 使用的 torchvision backbone。"""
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
            "或配置 patchcore.backbone_weights_path 指向本地预训练权重。"
        )

    torch.hub.set_dir(str(Path.cwd() / ".torch_cache"))
    try:
        return builder(weights=default_weights)
    except Exception as exc:  # pragma: no cover - depends on local cache/network
        raise RuntimeError(
            "完整 PatchCore 已启用 backbone_pretrained=True，但当前环境无法加载预训练权重。"
            "可选方案：1) 配置 patchcore.backbone_weights_path 指向本地权重；"
            "2) 先把 torchvision 权重缓存到项目 .torch_cache。"
        ) from exc


def _load_torch_transformer_backbone(config: PatchCoreConfig) -> Any:
    """加载 Transformer PatchCore 使用的 torchvision ViT backbone。"""
    builder, default_weights, default_image_size = _resolve_transformer_builder(config.backbone_name)
    if config.backbone_weights_path:
        model = builder(weights=None, image_size=int(config.image_size))
        state_dict = torch.load(config.backbone_weights_path, map_location="cpu")
        state_dict = _unwrap_state_dict(state_dict)
        normalized_state_dict = _normalize_state_dict_keys(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(normalized_state_dict, strict=False)
        critical_missing = [key for key in missing_keys if not key.startswith("heads.")]
        if critical_missing:
            raise RuntimeError(
                "backbone_weights_path 无法正确恢复 Transformer PatchCore backbone，"
                f"缺少关键参数数量: {len(critical_missing)}"
            )
        if unexpected_keys:
            raise RuntimeError(
                "backbone_weights_path 包含无法匹配的 Transformer 参数，"
                f"请确认权重与 backbone_name 对齐，异常参数数量: {len(unexpected_keys)}"
            )
        return model

    if not config.backbone_pretrained:
        raise RuntimeError(
            "Transformer PatchCore 不能使用随机初始化 backbone。"
            " 请设置 patchcore.backbone_pretrained=true，"
            "或配置 patchcore.backbone_weights_path 指向本地预训练权重。"
        )
    if int(config.image_size) != int(default_image_size):
        raise RuntimeError(
            "torchvision 预训练 ViT 权重要求 patchcore.image_size="
            f"{default_image_size}，当前为 {config.image_size}。"
            " 请把 image_size 调整为该值，或配置 backbone_weights_path 使用本地匹配权重。"
        )

    torch.hub.set_dir(str(Path.cwd() / ".torch_cache"))
    try:
        return builder(weights=default_weights)
    except Exception as exc:  # pragma: no cover - depends on local cache/network
        raise RuntimeError(
            "Transformer PatchCore 已启用 backbone_pretrained=True，但当前环境无法加载预训练权重。"
            "可选方案：1) 配置 patchcore.backbone_weights_path 指向本地权重；"
            "2) 先把 torchvision ViT 权重缓存到项目 .torch_cache。"
        ) from exc


def _resolve_backbone_builder(backbone_name: str) -> tuple[Any, Any]:
    """根据 backbone 名称返回构造器和默认权重枚举。"""
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


def _resolve_transformer_builder(backbone_name: str) -> tuple[Any, Any, int]:
    """根据 Transformer backbone 名称返回构造器、默认权重和输入尺寸。"""
    normalized = backbone_name.strip().lower()
    builders = {
        "vit_b_16": (vit_b_16, ViT_B_16_Weights.DEFAULT if ViT_B_16_Weights is not None else None, 224),
        "vit_b_32": (vit_b_32, ViT_B_32_Weights.DEFAULT if ViT_B_32_Weights is not None else None, 224),
        "vit_l_16": (vit_l_16, ViT_L_16_Weights.DEFAULT if ViT_L_16_Weights is not None else None, 224),
        "vit_l_32": (vit_l_32, ViT_L_32_Weights.DEFAULT if ViT_L_32_Weights is not None else None, 224),
    }
    builder = builders.get(normalized)
    if builder is None or builder[0] is None:
        supported = ", ".join(sorted(builders))
        raise ValueError(f"Unsupported Transformer PatchCore backbone `{backbone_name}`，当前支持：{supported}")
    return builder


def _extract_vit_patch_tokens(model: Any, input_tensor: Any) -> Any:
    """Return final ViT patch tokens without the classification head."""
    if not hasattr(model, "_process_input") or not hasattr(model, "class_token") or not hasattr(model, "encoder"):
        raise ValueError("Transformer PatchCore 当前只支持 torchvision VisionTransformer 模型")
    tokens = model._process_input(input_tensor)
    batch_size = tokens.shape[0]
    class_token = model.class_token.expand(batch_size, -1, -1)
    tokens = torch.cat([class_token, tokens], dim=1)
    encoded = model.encoder(tokens)
    return encoded[:, 1:, :]


def _resolve_square_token_grid(token_count: int) -> tuple[int, int]:
    """Resolve torchvision ViT's square patch-token grid."""
    grid_size = int(round(token_count ** 0.5))
    if grid_size * grid_size != token_count:
        raise ValueError(f"Transformer patch token 数量不是方形网格: {token_count}")
    return grid_size, grid_size


def _unwrap_state_dict(payload: Any) -> dict[str, Any]:
    """Extract state_dict from common training-framework checkpoint wrappers."""
    if not isinstance(payload, dict):
        raise TypeError("backbone_weights_path 必须加载到 state_dict 字典")
    for candidate_key in ("state_dict", "model", "backbone", "network"):
        nested = payload.get(candidate_key)
        if isinstance(nested, dict):
            return nested
    return payload


def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """去掉常见包装前缀，尽量把权重键名压平。"""
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
    """沿着点路径找到 backbone 中的目标层。"""
    current = model
    for part in layer_name.split("."):
        if not hasattr(current, part):
            raise ValueError(f"PatchCore feature layer `{layer_name}` 在 backbone 中不存在")
        current = getattr(current, part)
    return current


def _resize_mask(mask: np.ndarray | None, image_size: int) -> np.ndarray:
    """把原始掩膜缩放到 PatchCore 输入尺寸。"""
    if mask is None:
        return np.ones((image_size, image_size), dtype=np.float32)
    binary_mask = _mask_to_binary(mask)
    resized = cv2.resize(
        binary_mask,
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
    """把像素级掩膜压到 patch 网格上。"""
    if mask is None:
        return np.ones((grid_rows, grid_cols), dtype=np.float32)
    binary_mask = _mask_to_binary(mask)
    resized = cv2.resize(
        binary_mask,
        (grid_cols, grid_rows),
        interpolation=interpolation,
    )
    return resized.astype(np.float32)


def _prepare_feature_image(
    image: np.ndarray,
    *,
    target_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convert transparent PatchCore inputs to BGR without treating alpha-0 pixels as black."""
    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim != 3:
        raise ValueError("PatchCore image must be a 2D or 3D array")
    if array.shape[2] == 1:
        return cv2.cvtColor(array[:, :, 0], cv2.COLOR_GRAY2BGR)
    if array.shape[2] == 3:
        return array
    if array.shape[2] != 4:
        raise ValueError(f"PatchCore image has unsupported channel count: {array.shape[2]}")

    bgr = array[:, :, :3].copy()
    alpha_mask = array[:, :, 3] > 0
    if not alpha_mask.any() and target_mask is not None:
        alpha_mask = _mask_to_binary(target_mask, output_shape=array.shape[:2]) > 0
    if alpha_mask.all():
        return bgr
    if alpha_mask.any():
        fill_color = np.median(bgr[alpha_mask], axis=0)
    else:
        fill_color = np.zeros((3,), dtype=np.float32)
    bgr[~alpha_mask] = np.clip(fill_color, 0, 255).astype(bgr.dtype)
    return bgr


def _mask_to_binary(
    mask: np.ndarray,
    *,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Normalize a binary mask or transparent image mask to a float 0/1 array."""
    array = np.asarray(mask)
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = array[:, :, 3]
        else:
            array = np.any(array > 0, axis=2)
    binary = (array > 0).astype(np.float32)
    if output_shape is not None and binary.shape != output_shape:
        binary = cv2.resize(
            binary,
            (output_shape[1], output_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return binary.astype(np.float32)
