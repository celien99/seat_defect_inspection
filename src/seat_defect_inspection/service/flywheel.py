"""飞轮自学习训练编排。

监控缓冲区阈值，触发缺陷分类器微调、PatchCore 增量更新和模型版本管理。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from seat_defect_core.service.core import InspectionService

    from ..config import InspectionConfig


def check_and_retrain_if_needed(
    service: "InspectionService",
    *,
    seat_model_id: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """检查飞轮缓冲区并在条件满足时触发重训练。

    Args:
        service: 检测服务实例。
        seat_model_id: 限定处理的座椅型号。
        dry_run: 仅检查条件，不实际训练。

    Returns:
        决策和训练结果摘要。
    """
    config = service.config
    if not config.flywheel.enabled:
        return {"status": "flywheel_disabled"}

    buffer = service.get_flywheel_buffer()
    if buffer is None:
        return {"status": "no_buffer"}

    should, reason = buffer.should_retrain()
    if not should:
        # 仍有周期维护工作可以做
        archived = buffer.archive_old_samples(max_age_days=90)
        pruned = buffer.enforce_max_samples()
        return {
            "status": "not_triggered",
            "reason": reason,
            "archived_samples": archived,
            "pruned_samples": sum(pruned.values()),
        }

    if dry_run:
        stats = buffer.get_class_counts()
        return {
            "status": "would_retrain",
            "reason": reason,
            "buffer_stats": stats,
        }

    # 执行重训练
    context = service.resolve_context(seat_model_id)
    collector = service.get_flywheel_collector()
    registry = _get_registry(config)

    summary: dict[str, Any] = {
        "status": "retraining",
        "reason": reason,
        "seat_model_id": context.seat_model_id,
        "cameras": [],
    }

    for camera in context.cameras:
        camera_summary = _retrain_camera(
            service,
            camera,
            context,
            collector,
            registry,
        )
        summary["cameras"].append(camera_summary)

    buffer.record_retrain()
    buffer.archive_old_samples(max_age_days=90)
    buffer.enforce_max_samples()

    summary["status"] = "completed"
    return summary


def _retrain_camera(
    service: "InspectionService",
    camera: object,
    context: object,
    collector: object | None,
    registry: object | None,
) -> dict[str, Any]:
    """对单个机位执行飞轮重训练。"""
    camera_summary: dict[str, Any] = {
        "camera_id": camera.camera_id,
        "classifier_fine_tuned": False,
        "patchcore_updated": False,
    }

    if not camera.classification.enabled:
        return camera_summary

    # 1. 缺陷分类器微调
    classifier_metrics = _fine_tune_classifier(
        service,
        camera,
        context.seat_model_id,
        collector,
        registry,
    )
    if classifier_metrics:
        camera_summary["classifier_fine_tuned"] = True
        camera_summary["classifier_metrics"] = classifier_metrics

    # 2. PatchCore 增量更新（可选）
    if service.config.flywheel.incremental_patchcore_enabled:
        patchcore_updated = _incremental_patchcore_update(
            service,
            camera,
            context,
            collector,
        )
        camera_summary["patchcore_updated"] = patchcore_updated

    return camera_summary


def _fine_tune_classifier(
    service: "InspectionService",
    camera: object,
    seat_model_id: str | None,
    collector: object | None,
    registry: object | None,
) -> dict[str, Any] | None:
    """从 TP 和 FP 缓冲区微调分类器。"""
    if collector is None:
        return None

    from seat_defect_core.classifier.training import DefectClassifierTrainer

    buffer_root = Path(service.config.flywheel.buffer_dir)
    camera_dir = buffer_root / camera.camera_id / (seat_model_id or "default")

    # 收集训练样本
    samples = []
    label_counts: dict[str, int] = {}

    for tp_dir_name in _iter_tp_dirs(camera_dir):
        defect_type = tp_dir_name.split("/", 1)[-1] if "/" in tp_dir_name else tp_dir_name
        npz_files = list((camera_dir / tp_dir_name).glob("*.npz"))
        label_counts[defect_type] = len(npz_files)
        for npz_path in npz_files[:5000]:  # 上限
            try:
                data = np.load(npz_path, allow_pickle=False)
                if "heatmap" in data and "roi_image" in data:
                    samples.append((data["heatmap"], data["roi_image"], defect_type))
            except Exception:
                continue

    # 添加 FP 样本
    fp_dir = camera_dir / "fp"
    if fp_dir.is_dir():
        fp_count = len(list(fp_dir.glob("*.npz")))
        label_counts["none"] = label_counts.get("none", 0) + fp_count
        for npz_path in list(fp_dir.glob("*.npz"))[:5000]:
            try:
                data = np.load(npz_path, allow_pickle=False)
                if "roi_image" in data:
                    heatmap = data.get("heatmap", np.zeros_like(data["roi_image"]))
                    samples.append((heatmap, data["roi_image"], "none"))
            except Exception:
                continue

    if len(samples) < 20:
        return None

    classifier_config = camera.classification
    trainer = DefectClassifierTrainer(
        config=classifier_config,
        backbone_name="efficientnet_b0",
        image_size=224,
    )

    version = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(classifier_config.model_path)
    fine_tuned_path = output_path.parent / f"{output_path.stem}_{version}{output_path.suffix}"

    metrics = trainer.fit(
        samples,
        val_split=0.15,
        epochs=30,  # 微调用较少 epochs
        batch_size=16,
        learning_rate=0.0001,  # 更低的学习率
        focal_loss_gamma=2.0,
        patience=5,
        augment=True,
    )

    trainer.save(
        fine_tuned_path,
        metadata={
            "version": version,
            "label_counts": label_counts,
            "parent_version": getattr(service.get_classifier_service(camera), "version", None),
        },
    )

    # 注册新版本
    if registry is not None:
        from seat_defect_core.model_registry import ModelCard

        card = ModelCard(
            model_version=version,
            architecture=f"classifier_{trainer._backbone_name}",
            training_date=time.strftime("%Y-%m-%d"),
            training_sample_count=len(samples),
            metrics=metrics,
            parent_version=getattr(service.get_classifier_service(camera), "version", None),
            label_counts=label_counts,
        )
        registry.register(
            camera_id=camera.camera_id,
            region_id="__classifier__",
            model_path=fine_tuned_path,
            card=card,
        )

    return metrics


def _incremental_patchcore_update(
    service: "InspectionService",
    camera: object,
    context: object,
    collector: object | None,
) -> bool:
    """增量更新 PatchCore memory bank。

    只有在新 TP 样本被确认后才有意义 — 这些样本代表被验证过的正常纹理变化，
    将其 embedding 追加到 memory bank 并重新 coreset 采样。
    """
    # 增量 PatchCore 更新需要：
    # 1. 从 tp/none/ 缓冲区加载被确认为正常的新样本
    # 2. 通过当前模型中提取 embedding
    # 3. 追加到 memory bank，重新 coreset
    # 4. 重新计算阈值
    #
    # 这需要在 core 层的 PatchCoreService 中暴露 fit_from_embeddings 接口。
    # 当前 PatchCore 模型的 .npz 格式已经支持独立操作。
    # 由于涉及较多底层 PatchCore 重构，标记为占位，待 Phase 3 完整实现。
    return False


def _get_registry(config: "InspectionConfig"):
    """获取或创建模型注册中心。"""
    if config.model_registry_dir is None:
        return None
    from seat_defect_core.model_registry import ModelRegistry

    return ModelRegistry(config.model_registry_dir)


def _iter_tp_dirs(camera_dir: Path):
    """遍历 TP 样本目录。"""
    if not camera_dir.is_dir():
        return
    for item in sorted(camera_dir.iterdir()):
        if item.is_dir() and item.name.startswith("tp"):
            yield item.name
            # 也遍历 tp/ 下的子目录
            for sub in sorted(item.iterdir()):
                if sub.is_dir():
                    yield f"{item.name}/{sub.name}"
