"""缺陷分类器核心训练逻辑。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..config import ClassificationConfig


class DefectClassifierTrainer:
    """缺陷分类器训练器。

    使用 Focal Loss 处理类别不均衡，支持 EfficientNet 和 MobileNetV3 backbone。
    """

    def __init__(
        self,
        config: "ClassificationConfig",
        *,
        backbone_name: str = "efficientnet_b0",
        image_size: int = 224,
        class_names: list[str] | None = None,
        device: str = "cpu",
    ) -> None:
        self._config = config
        self._backbone_name = backbone_name
        self._image_size = image_size
        self._class_names = class_names or _DEFAULT_CLASS_NAMES
        self._device = device
        self._model: object | None = None

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    def fit(
        self,
        samples: list[tuple[np.ndarray, np.ndarray, str]],
        *,
        val_split: float = 0.15,
        class_weights: dict[str, float] | None = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        focal_loss_gamma: float = 2.0,
        patience: int = 10,
        augment: bool = True,
    ) -> dict[str, object]:
        """训练分类模型。

        Args:
            samples: (heatmap, roi_image, label) 三元组列表。
            val_split: 验证集比例。
            class_weights: 类别权重字典。
            epochs: 训练轮数。
            batch_size: 批次大小。
            learning_rate: 学习率。
            weight_decay: 权重衰减。
            focal_loss_gamma: Focal Loss gamma 参数。
            patience: 早停耐心值。
            augment: 是否启用数据增强。

        Returns:
            训练指标字典，包含 best_val_acc, best_val_f1, best_epoch。
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Subset, TensorDataset

        from .engine import _build_classifier_model

        if len(samples) == 0:
            raise ValueError("No training samples provided")

        # 构建标签索引
        label_to_idx = {name: i for i, name in enumerate(self._class_names)}

        # 预处理所有样本
        all_heatmaps = []
        all_rois = []
        all_labels = []
        for heatmap, roi, label in samples:
            prepared_heatmap, prepared_roi = _prepare_sample(
                heatmap, roi, self._image_size
            )
            all_heatmaps.append(prepared_heatmap)
            all_rois.append(prepared_roi)
            all_labels.append(label_to_idx.get(label, label_to_idx.get("other", 0)))

        heatmap_tensor = torch.from_numpy(np.stack(all_heatmaps)).float()
        roi_tensor = torch.from_numpy(np.stack(all_rois)).float()
        label_tensor = torch.tensor(all_labels, dtype=torch.long)
        input_tensor = torch.stack([heatmap_tensor, roi_tensor], dim=1)
        dataset = TensorDataset(input_tensor, label_tensor)

        # 训练/验证划分
        n_val = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        indices = torch.randperm(len(dataset)).tolist()
        train_ds = Subset(dataset, indices[:n_train])
        val_ds = Subset(dataset, indices[n_train:])

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # 模型
        num_classes = len(self._class_names)
        model = _build_classifier_model(
            backbone_name=self._backbone_name,
            num_classes=num_classes,
            in_channels=2,
        )
        model.to(self._device)
        self._model = model

        # 类别权重
        if class_weights is not None:
            weight_list = [class_weights.get(name, 1.0) for name in self._class_names]
            alpha = torch.tensor(weight_list, device=self._device)
        else:
            alpha = None

        criterion = FocalLoss(gamma=focal_loss_gamma, alpha=alpha)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_epoch = 0
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0.0
            for batch_input, batch_labels in train_loader:
                batch_input = batch_input.to(self._device)
                batch_labels = batch_labels.to(self._device)
                if augment:
                    batch_input = _augment_batch(batch_input)

                optimizer.zero_grad()
                logits = model(batch_input)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()
            avg_train_loss = train_loss / max(len(train_loader), 1)

            # 验证
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for batch_input, batch_labels in val_loader:
                    batch_input = batch_input.to(self._device)
                    batch_labels = batch_labels.to(self._device)
                    logits = model(batch_input)
                    loss = criterion(logits, batch_labels)
                    val_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    correct += (preds == batch_labels).sum().item()
                    total += batch_labels.size(0)
                    all_preds.extend(preds.cpu().tolist())
                    all_targets.extend(batch_labels.cpu().tolist())

            val_acc = correct / max(total, 1)
            val_f1 = _compute_macro_f1(all_targets, all_preds, num_classes)
            avg_val_loss = val_loss / max(len(val_loader), 1)

            if val_f1 > best_val_f1:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return {
            "best_val_acc": float(best_val_acc),
            "best_val_f1": float(best_val_f1),
            "best_epoch": int(best_epoch),
            "num_train": n_train,
            "num_val": n_val,
            "num_classes": num_classes,
        }

    def save(self, path: str | Path, *, metadata: dict | None = None) -> None:
        """保存训练好的分类模型。"""
        import torch

        if self._model is None:
            raise RuntimeError("No model to save. Call fit() first.")

        save_data = {
            "model_state_dict": self._model.state_dict(),
            "metadata": {
                "backbone": self._backbone_name,
                "image_size": self._image_size,
                "class_names": self._class_names,
                "version": metadata.get("version", time.strftime("%Y%m%d_%H%M%S")),
                **(metadata or {}),
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_data, str(path))


_DEFAULT_CLASS_NAMES = [
    "none",
    "scratch",
    "stain",
    "wrinkle",
    "thread_jump",
    "foreign_matter",
    "dent",
    "color_shift",
    "other",
]


class FocalLoss:
    """Focal Loss for class imbalance."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: object = None,
        reduction: str = "mean",
    ) -> None:
        import torch.nn.functional as F  # noqa: N813

        self._gamma = gamma
        self._alpha = alpha
        self._reduction = reduction

    def __call__(self, inputs: object, targets: object) -> object:
        import torch.nn.functional as F

        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self._alpha)
        pt = (-ce_loss).exp()
        focal_loss = (1 - pt) ** self._gamma * ce_loss
        if self._reduction == "mean":
            return focal_loss.mean()
        elif self._reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def _prepare_sample(
    heatmap: np.ndarray,
    roi_image: np.ndarray,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """预处理单个训练样本：resize + normalize。"""
    import cv2

    heatmap_resized = cv2.resize(
        heatmap.astype(np.float32),
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )
    heatmap_norm = np.clip(heatmap_resized / max(heatmap_resized.max(), 1e-8), 0, 2)

    if roi_image.ndim == 3 and roi_image.shape[2] >= 3:
        roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    elif roi_image.ndim == 3:
        roi_gray = roi_image[:, :, 0]
    else:
        roi_gray = roi_image
    roi_resized = cv2.resize(
        roi_gray.astype(np.float32),
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )
    roi_norm = roi_resized / 255.0
    return heatmap_norm, roi_norm


def _augment_batch(batch_input: object) -> object:
    """对分类器输入应用轻量数据增强。"""
    import torch
    import torch.nn.functional as F

    if torch.rand(1).item() < 0.5:
        batch_input = torch.flip(batch_input, dims=[-1])
    angle = torch.randint(-10, 10, (1,)).item()
    if abs(angle) > 1:
        rad = angle * (3.14159265 / 180.0)
        cos_a, sin_a = rad.cos(), rad.sin()
        theta = torch.tensor(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], device=batch_input.device
        ).float().unsqueeze(0).expand(batch_input.size(0), -1, -1)
        grid = F.affine_grid(theta, batch_input.size(), align_corners=False)
        batch_input = F.grid_sample(
            batch_input, grid, align_corners=False, mode="bilinear"
        )
    return batch_input


def _compute_macro_f1(
    targets: list[int],
    preds: list[int],
    num_classes: int,
) -> float:
    """计算 macro F1 分数。"""
    from collections import Counter

    tp = Counter()
    fp = Counter()
    fn_count = Counter()
    for t, p in zip(targets, preds):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn_count[t] += 1

    f1_scores = []
    for c in range(num_classes):
        tp_c = tp.get(c, 0)
        fp_c = fp.get(c, 0)
        fn_c = fn_count.get(c, 0)
        precision = tp_c / max(tp_c + fp_c, 1)
        recall = tp_c / max(tp_c + fn_c, 1)
        if precision + recall > 0:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1_scores)) if f1_scores else 0.0
