"""模型注册中心。

管理模型版本、元数据和生命周期（active/staging/archived）。
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelCard:
    """模型版本元数据卡片。"""

    model_version: str
    """语义版本号，格式：YYYYMMDD_HHMMSS 或 v1.2.3。"""

    architecture: str
    """模型架构标识，如 patchcore_wrn50、classifier_mbv3。"""

    training_date: str
    """训练完成日期。"""

    training_sample_count: int
    """训练样本总数。"""

    metrics: dict[str, float] = field(default_factory=dict)
    """评估指标：precision、recall、f1、fp_rate 等。"""

    config_hash: str = ""
    """训练时配置的哈希值。"""

    pipeline_signature: str = ""
    """检测流水线签名。"""

    parent_version: str | None = None
    """父版本号，用于版本溯源。"""

    status: str = "active"
    """模型状态：active、staging、archived、deprecated。"""

    label_counts: dict[str, int] = field(default_factory=dict)
    """训练样本类别分布。"""

    notes: str = ""
    """人工备注。"""


class ModelRegistry:
    """基于目录的模型注册中心。

    目录结构：
        {registry_dir}/
            {camera_id}/
                {region_id}/
                    {version}/
                        model.npz (或 classifier.pt)
                        card.json

    region_id 为 "__full__" 时表示完整 ROI 模型。
    """

    def __init__(self, registry_dir: str | Path) -> None:
        self._root = Path(registry_dir)

    @property
    def root(self) -> Path:
        return self._root

    def register(
        self,
        *,
        camera_id: str,
        region_id: str = "__full__",
        model_path: str | Path,
        card: ModelCard,
    ) -> Path:
        """注册一个新模型版本。

        将模型文件复制到注册中心并写入卡片。

        Returns:
            模型在注册中心中的版本目录路径。
        """
        version_dir = self._root / camera_id / region_id / card.model_version
        version_dir.mkdir(parents=True, exist_ok=True)

        # 复制模型文件
        src = Path(model_path)
        dest = version_dir / src.name
        if src != dest:
            shutil.copy2(str(src), str(dest))

        # 写入卡片
        card_path = version_dir / "card.json"
        card_path.write_text(
            json.dumps(_card_to_dict(card), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 更新 active 符号链接
        active_link = self._root / camera_id / region_id / "active"
        if active_link.is_symlink() or active_link.exists():
            active_link.unlink()
        active_link.symlink_to(version_dir.name, target_is_directory=True)

        return version_dir

    def get_active(self, camera_id: str, region_id: str = "__full__") -> ModelCard | None:
        """获取当前活跃模型版本。"""
        active_link = self._root / camera_id / region_id / "active"
        if not active_link.is_symlink():
            return None
        version = active_link.resolve().name
        return self.get_card(camera_id, region_id, version)

    def get_card(
        self,
        camera_id: str,
        region_id: str,
        version: str,
    ) -> ModelCard | None:
        """读取指定版本的模型卡片。"""
        card_path = self._root / camera_id / region_id / version / "card.json"
        if not card_path.is_file():
            return None
        data = json.loads(card_path.read_text(encoding="utf-8"))
        return _card_from_dict(data)

    def list_versions(
        self,
        camera_id: str,
        region_id: str = "__full__",
    ) -> list[ModelCard]:
        """列出指定机位/区域的所有模型版本。"""
        versions_dir = self._root / camera_id / region_id
        if not versions_dir.is_dir():
            return []

        cards = []
        for item in sorted(versions_dir.iterdir()):
            if item.is_symlink() or not item.is_dir():
                continue
            card = self.get_card(camera_id, region_id, item.name)
            if card is not None:
                cards.append(card)
        return cards

    def promote(
        self,
        camera_id: str,
        region_id: str,
        version: str,
    ) -> bool:
        """将指定版本提升为 active。"""
        version_dir = self._root / camera_id / region_id / version
        if not version_dir.is_dir():
            return False

        active_link = self._root / camera_id / region_id / "active"
        if active_link.is_symlink() or active_link.exists():
            # 归档旧版本
            old_version = active_link.resolve().name if active_link.is_symlink() else None
            if old_version:
                old_card = self.get_card(camera_id, region_id, old_version)
                if old_card and old_card.status == "active":
                    old_card.status = "archived"
                    self._write_card(camera_id, region_id, old_version, old_card)

        active_link.unlink(missing_ok=True)
        active_link.symlink_to(version, target_is_directory=True)

        # 更新新版本状态
        card = self.get_card(camera_id, region_id, version)
        if card:
            card.status = "active"
            self._write_card(camera_id, region_id, version, card)
        return True

    def rollback(self, camera_id: str, region_id: str = "__full__") -> ModelCard | None:
        """回滚到上一个版本。"""
        cards = self.list_versions(camera_id, region_id)
        # 找最近的 archived 版本
        archived = [c for c in cards if c.status == "archived"]
        if not archived:
            # 降级：找最近的非 active 版本
            archived = [c for c in cards if c.status != "active"]
        if not archived:
            return None

        prev = archived[-1]  # 最近的
        self.promote(camera_id, region_id, prev.model_version)
        return prev

    def _write_card(
        self,
        camera_id: str,
        region_id: str,
        version: str,
        card: ModelCard,
    ) -> None:
        card_path = self._root / camera_id / region_id / version / "card.json"
        card_path.write_text(
            json.dumps(_card_to_dict(card), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _card_to_dict(card: ModelCard) -> dict[str, Any]:
    return {
        "model_version": card.model_version,
        "architecture": card.architecture,
        "training_date": card.training_date,
        "training_sample_count": card.training_sample_count,
        "metrics": card.metrics,
        "config_hash": card.config_hash,
        "pipeline_signature": card.pipeline_signature,
        "parent_version": card.parent_version,
        "status": card.status,
        "label_counts": card.label_counts,
        "notes": card.notes,
    }


def _card_from_dict(data: dict[str, Any]) -> ModelCard:
    return ModelCard(
        model_version=data.get("model_version", "unknown"),
        architecture=data.get("architecture", "unknown"),
        training_date=data.get("training_date", ""),
        training_sample_count=data.get("training_sample_count", 0),
        metrics=data.get("metrics", {}),
        config_hash=data.get("config_hash", ""),
        pipeline_signature=data.get("pipeline_signature", ""),
        parent_version=data.get("parent_version"),
        status=data.get("status", "active"),
        label_counts=data.get("label_counts", {}),
        notes=data.get("notes", ""),
    )
