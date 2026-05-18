"""检测数据采集服务（异步模式）。

在每次检测完成后将 NG 样本和采样的 OK 样本持久化到飞轮缓冲区。
使用后台线程异步写入磁盘，主检测流程零阻塞。
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..config import FlywheelConfig
    from ..types.results import CameraInspectionResult, InspectionResult

_logger = logging.getLogger(__name__)


class DataCollectorService:
    """检测数据采集器（异步磁盘写入）。

    将检测结果及中间产物写入结构化缓冲区，供后续自学习训练使用。
    collect() 方法仅入队数据并立即返回，后台线程负责磁盘 I/O。
    """

    _QUEUE_DEPTH_WARN = 500

    def __init__(self, config: "FlywheelConfig") -> None:
        self._config = config
        self._buffer_root = Path(config.buffer_dir)
        self._rng = np.random.RandomState(42)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._running = False
        self._start_worker()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def collect(
        self,
        result: "InspectionResult",
        camera_results: list["CameraInspectionResult"],
        roi_images: dict[str, np.ndarray] | None = None,
        heatmaps: dict[str, np.ndarray] | None = None,
    ) -> dict[str, int]:
        """入队一次检测的样本数据，立即返回（不等待磁盘写入）。

        Returns:
            入队统计字典（非实际写入数）。
        """
        if not self._config.enabled:
            return {}

        stats: dict[str, int] = {"ok": 0, "ng": 0, "reject": 0, "fp": 0, "total": 0}
        timestamp = result.timestamp or time.strftime("%Y%m%d_%H%M%S")
        seat_model_id = result.seat_model_id or "default"

        for cam_result in camera_results:
            camera_id = cam_result.camera_id
            roi_img = (roi_images or {}).get(camera_id)
            heatmap = (heatmaps or {}).get(camera_id)

            if cam_result.status == "OK":
                if self._rng.random() < self._config.sampling_rate_ok:
                    self._enqueue_sample(
                        camera_id=camera_id,
                        seat_model_id=seat_model_id,
                        timestamp=timestamp,
                        part_id=result.part_id,
                        sample_type="ok",
                        roi_image=roi_img,
                        heatmap=heatmap,
                        metadata=_extract_metadata(cam_result),
                    )
                    stats["ok"] += 1

            elif cam_result.status == "NG":
                sample_type = self._classify_ng_sample(cam_result)
                self._enqueue_sample(
                    camera_id=camera_id,
                    seat_model_id=seat_model_id,
                    timestamp=timestamp,
                    part_id=result.part_id,
                    sample_type=sample_type,
                    roi_image=roi_img,
                    heatmap=heatmap,
                    metadata=_extract_metadata(cam_result),
                )
                stats["ng"] += 1
                if sample_type == "fp":
                    stats["fp"] += 1

            elif cam_result.status == "REJECT":
                stats["reject"] += 1

            stats["total"] += 1

        # 队列堆积过多告警
        qsize = self._queue.qsize()
        if qsize > self._QUEUE_DEPTH_WARN:
            _logger.warning(
                "飞轮采集队列堆积 %d 条，可能存在磁盘 I/O 瓶颈", qsize
            )

        return stats

    def get_buffer_stats(self) -> dict[str, dict[str, int]]:
        """获取当前缓冲区各分类的样本数量统计。"""
        stats: dict[str, dict[str, int]] = {}
        if not self._buffer_root.is_dir():
            return stats

        for camera_dir in self._buffer_root.iterdir():
            if not camera_dir.is_dir():
                continue
            camera_stats: dict[str, int] = {}
            for model_dir in camera_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                for sample_dir in model_dir.iterdir():
                    if not sample_dir.is_dir():
                        continue
                    count = len(list(sample_dir.glob("*.npz")))
                    key = f"{model_dir.name}/{sample_dir.name}"
                    camera_stats[key] = camera_stats.get(key, 0) + count
            if camera_stats:
                stats[camera_dir.name] = camera_stats

        return stats

    def flush(self, timeout: float = 30.0) -> None:
        """等待队列中所有样本写入完成。"""
        if not self._running:
            return
        # 入队哨兵并等待
        self._queue.put(None)
        # 简单等待队列排空
        deadline = time.monotonic() + timeout
        while self._queue.qsize() > 0 and time.monotonic() < deadline:
            time.sleep(0.05)

    def shutdown(self) -> None:
        """停止后台写入线程。"""
        self._running = False
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=5.0)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _start_worker(self) -> None:
        self._running = True
        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="flywheel-writer",
        )
        self._worker.start()

    def _worker_loop(self) -> None:
        """后台线程：从队列取出样本并写入磁盘。"""
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:  # 哨兵
                self._queue.task_done()
                continue

            try:
                self._write_sample(item)
            except Exception:
                _logger.warning(
                    "飞轮样本写入失败 camera=%s type=%s",
                    item.get("camera_id", "?"),
                    item.get("sample_type", "?"),
                    exc_info=True,
                )
            finally:
                self._queue.task_done()

    def _enqueue_sample(self, **kwargs: Any) -> None:
        """将样本数据入队（深拷贝 numpy 数组以避免跨线程引用问题）。"""
        item = {}
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                item[key] = value.copy()
            else:
                item[key] = value
        self._queue.put(item)

    def _write_sample(self, item: dict[str, Any]) -> None:
        """将单个样本写入 .npz 文件。"""
        camera_id = item["camera_id"]
        seat_model_id = item["seat_model_id"]
        timestamp = item["timestamp"]
        part_id = item["part_id"]
        sample_type = item["sample_type"]
        roi_image = item.get("roi_image")
        heatmap = item.get("heatmap")
        metadata = item.get("metadata")

        save_dir = self._buffer_root / camera_id / seat_model_id / sample_type
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{timestamp}_{part_id}.npz"

        save_data: dict[str, Any] = {}
        if roi_image is not None:
            save_data["roi_image"] = roi_image
        if heatmap is not None:
            save_data["heatmap"] = heatmap
        if metadata is not None:
            save_data["metadata_json"] = json.dumps(metadata, ensure_ascii=False)

        np.savez_compressed(str(save_path), **save_data)

    def _classify_ng_sample(self, cam_result: "CameraInspectionResult") -> str:
        """根据分类器结果将 NG 样本归入对应子目录。"""
        texture = cam_result.texture_result
        if texture and texture.classification_results:
            primary = texture.classification_results[0]
            if primary.veto_applied:
                return "fp"
            if primary.confidence >= self._config.auto_label_threshold:
                return f"tp/{primary.defect_type.value}"
            if primary.confidence >= self._config.human_validation_threshold:
                return "hard"
            return "fp"

        if cam_result.region_results:
            for region in cam_result.region_results:
                if region.texture_result and region.texture_result.classification_results:
                    primary = region.texture_result.classification_results[0]
                    if primary.veto_applied:
                        continue
                    if primary.confidence >= self._config.auto_label_threshold:
                        return f"tp/{primary.defect_type.value}"
                    if primary.confidence >= self._config.human_validation_threshold:
                        return "hard"
            return "fp"

        return "hard"


def _extract_metadata(cam_result: "CameraInspectionResult") -> dict[str, Any]:
    """从 CameraInspectionResult 提取关键元数据。"""
    meta: dict[str, Any] = {
        "camera_id": cam_result.camera_id,
        "status": cam_result.status,
        "reason": cam_result.reason,
    }
    texture = cam_result.texture_result
    if texture is not None:
        meta["patchcore_score"] = texture.score
        meta["patchcore_threshold"] = texture.threshold
        meta["decision_mode"] = texture.decision_mode
        if texture.classification_results:
            meta["classification"] = [
                {
                    "defect_type": item.defect_type.value,
                    "confidence": item.confidence,
                    "veto_applied": item.veto_applied,
                }
                for item in texture.classification_results
            ]
    return meta
