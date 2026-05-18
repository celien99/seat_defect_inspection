"""飞轮缓冲区管理器。

管理缓冲区的样本数量阈值、自动归档和重训练触发逻辑。
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import FlywheelConfig


class BufferManager:
    """缓冲区管理器。

    职责：
    - 监控各类别样本数量，判断是否达到重训练阈值
    - 自动归档过期或超量样本
    - 提供重训练触发决策
    """

    def __init__(self, config: "FlywheelConfig") -> None:
        self._config = config
        self._buffer_root = Path(config.buffer_dir)
        self._last_retrain_time: float = 0.0

    @property
    def buffer_root(self) -> Path:
        return self._buffer_root

    def should_retrain(self, camera_id: str | None = None) -> tuple[bool, str]:
        """判断是否应触发重训练。

        Returns:
            (should_retrain, reason) 元组。
        """
        if not self._config.enabled:
            return False, "flywheel_disabled"

        # 检查冷却时间
        elapsed = time.time() - self._last_retrain_time
        cooldown_seconds = self._config.retrain_cooldown_hours * 3600
        if elapsed < cooldown_seconds:
            remaining_h = (cooldown_seconds - elapsed) / 3600
            return False, f"cooldown_remaining_{remaining_h:.1f}h"

        # 获取统计
        stats = self.get_class_counts(camera_id=camera_id)

        # 计算 TP 样本总数
        tp_total = sum(
            count
            for key, count in stats.items()
            if key.startswith("tp/")
        )

        if tp_total >= self._config.min_samples_before_retrain:
            return True, f"tp_samples_reached_{tp_total}"

        # 硬负样本也计入触发
        hard_count = stats.get("hard", 0)
        if hard_count >= self._config.min_samples_before_retrain:
            return True, f"hard_samples_reached_{hard_count}"

        return False, f"insufficient_samples_tp_{tp_total}_hard_{hard_count}"

    def get_class_counts(
        self,
        camera_id: str | None = None,
        seat_model_id: str | None = None,
    ) -> dict[str, int]:
        """统计缓冲区各分类的样本数量。"""
        counts: dict[str, int] = {}
        if not self._buffer_root.is_dir():
            return counts

        search_dir = self._buffer_root
        if camera_id:
            search_dir = search_dir / camera_id
        if seat_model_id and camera_id:
            search_dir = search_dir / seat_model_id

        for sample_type_dir in _iter_sample_dirs(search_dir):
            count = len(list(sample_type_dir.glob("*.npz")))
            if count > 0:
                key = sample_type_dir.name
                if seat_model_id and camera_id:
                    pass  # key is just the sample type
                elif camera_id:
                    # key includes model name
                    parent = sample_type_dir.parent.name
                    key = f"{parent}/{sample_type_dir.name}"
                counts[key] = count

        return counts

    def archive_old_samples(self, max_age_days: int = 90) -> int:
        """归档超过最大保留天数的样本。

        Returns:
            归档的样本文件数量。
        """
        archived = 0
        archive_root = self._buffer_root / "_archive"
        cutoff = time.time() - max_age_days * 86400

        for npz_path in self._buffer_root.rglob("*.npz"):
            if "_archive" in npz_path.parts:
                continue
            try:
                mtime = npz_path.stat().st_mtime
                if mtime < cutoff:
                    rel_path = npz_path.relative_to(self._buffer_root)
                    dest = archive_root / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(npz_path), str(dest))
                    archived += 1
            except OSError:
                continue

        return archived

    def enforce_max_samples(self) -> dict[str, int]:
        """对超量的类别执行样本裁剪，保留最新的 N 个。

        Returns:
            各类别裁剪掉的样本数量。
        """
        pruned: dict[str, int] = {}
        max_per_class = self._config.max_samples_per_class

        for sample_type_dir in _iter_sample_dirs(self._buffer_root):
            npz_files = sorted(
                sample_type_dir.glob("*.npz"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if len(npz_files) > max_per_class:
                to_remove = npz_files[max_per_class:]
                for f in to_remove:
                    f.unlink()
                key = str(sample_type_dir.relative_to(self._buffer_root))
                pruned[key] = len(to_remove)

        return pruned

    def record_retrain(self) -> None:
        """记录一次重训练完成时间。"""
        self._last_retrain_time = time.time()

    def get_last_retrain_time(self) -> float:
        return self._last_retrain_time


def _iter_sample_dirs(root: Path):
    """遍历缓冲区中所有样本类型目录。"""
    if not root.is_dir():
        return
    for dirpath in root.rglob("*"):
        if dirpath.is_dir() and any(dirpath.glob("*.npz")):
            yield dirpath
