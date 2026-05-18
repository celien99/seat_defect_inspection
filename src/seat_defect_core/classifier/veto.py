"""基于启发式规则的误报过滤器。

在 PatchCore 判定异常后、最终决策前执行，过滤掉物理上不合理的检出。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..config import FalsePositiveVetoConfig


@dataclass
class VetoDecision:
    """误报过滤判定结果。"""

    vetoed: bool
    """是否被否决（True 表示该异常为误报）。"""

    reason: str
    """否决原因描述。"""

    applied_rules: list[str]
    """实际触发的规则名称列表。"""


def apply_veto(
    heatmap: np.ndarray,
    heatmap_threshold: float = 1.0,
    *,
    config: "FalsePositiveVetoConfig",
) -> VetoDecision:
    """对 PatchCore 热力图执行启发式误报过滤。

    Args:
        heatmap: ROI 坐标系下的异常热力图 (H, W) float32。
        heatmap_threshold: 判定为异常的阈值，默认为 1.0。
        config: 误报过滤配置。

    Returns:
        VetoDecision 对象。
    """
    if not config.enabled:
        return VetoDecision(vetoed=False, reason="veto_disabled", applied_rules=[])

    if heatmap.size == 0:
        return VetoDecision(vetoed=True, reason="empty_heatmap", applied_rules=["empty_heatmap"])

    anomaly_mask = (heatmap > heatmap_threshold).astype(np.uint8)
    if anomaly_mask.sum() == 0:
        return VetoDecision(vetoed=True, reason="no_anomaly_pixels", applied_rules=["no_anomaly_pixels"])

    total_area = anomaly_mask.shape[0] * anomaly_mask.shape[1]
    applied_rules: list[str] = []
    vetoed = False
    reason = ""

    rule = _check_area_ratio(anomaly_mask, total_area, config)
    if rule:
        vetoed = True
        reason = rule
        applied_rules.append("min_area_ratio")

    if not vetoed:
        rule = _check_aspect_ratio(anomaly_mask, config)
        if rule:
            vetoed = True
            reason = rule
            applied_rules.append("max_aspect_ratio")

    if not vetoed:
        rule = _check_edge_proximity(anomaly_mask, config)
        if rule:
            vetoed = True
            reason = rule
            applied_rules.append("edge_proximity")

    if vetoed:
        return VetoDecision(vetoed=True, reason=reason, applied_rules=applied_rules)
    return VetoDecision(vetoed=False, reason="all_checks_passed", applied_rules=[])


def _find_anomaly_components(
    anomaly_mask: np.ndarray,
) -> list[dict]:
    """在异常 mask 中寻找连通域，返回每个连通域的统计信息。"""
    import cv2

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        anomaly_mask, connectivity=8
    )
    components = []
    for i in range(1, num_labels):  # 跳过背景 (label 0)
        area = int(stats[i, cv2.CC_STAT_AREA])
        left = int(stats[i, cv2.CC_STAT_LEFT])
        top = int(stats[i, cv2.CC_STAT_TOP])
        width = int(stats[i, cv2.CC_STAT_WIDTH])
        height = int(stats[i, cv2.CC_STAT_HEIGHT])
        components.append(
            {
                "area": area,
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "right": left + width,
                "bottom": top + height,
            }
        )
    return components


def _check_area_ratio(
    anomaly_mask: np.ndarray,
    total_area: int,
    config: "FalsePositiveVetoConfig",
) -> str:
    """检查异常面积是否过小（斑点噪声）。"""
    anomaly_area = int(anomaly_mask.sum())
    ratio = anomaly_area / total_area
    if ratio < config.min_defect_area_ratio:
        return (
            f"anomaly_area_ratio_too_small: {ratio:.6f} < "
            f"{config.min_defect_area_ratio:.6f}"
        )
    return ""


def _check_aspect_ratio(
    anomaly_mask: np.ndarray,
    config: "FalsePositiveVetoConfig",
) -> str:
    """检查异常连通域长宽比（光照条带状伪影通常极长极窄）。"""
    components = _find_anomaly_components(anomaly_mask)
    if not components:
        return ""

    # 检查最大连通域的长宽比
    largest = max(components, key=lambda c: c["area"])
    if largest["width"] == 0 or largest["height"] == 0:
        return ""
    aspect_ratio = min(largest["width"], largest["height"]) / max(
        largest["width"], largest["height"]
    )
    if aspect_ratio < config.max_defect_aspect_ratio:
        return (
            f"anomaly_aspect_ratio_too_extreme: {aspect_ratio:.4f} < "
            f"{config.max_defect_aspect_ratio:.4f}"
        )
    return ""


def _check_edge_proximity(
    anomaly_mask: np.ndarray,
    config: "FalsePositiveVetoConfig",
) -> str:
    """检查异常是否过于贴近 ROI 边界（边界伪影）。"""
    height, width = anomaly_mask.shape
    edge_x = int(width * config.edge_proximity_ratio)
    edge_y = int(height * config.edge_proximity_ratio)
    if edge_x <= 0 or edge_y <= 0:
        return ""

    components = _find_anomaly_components(anomaly_mask)
    anomaly_pixels = int(anomaly_mask.sum())
    if anomaly_pixels == 0 or not components:
        return ""

    # 计算在边界区域内的异常像素比例
    edge_mask = np.zeros_like(anomaly_mask)
    edge_mask[:edge_y, :] = 1
    edge_mask[-edge_y:, :] = 1
    edge_mask[:, :edge_x] = 1
    edge_mask[:, -edge_x:] = 1

    edge_pixels = int((anomaly_mask & edge_mask).sum())
    edge_ratio = edge_pixels / anomaly_pixels

    # 如果超过 80% 的异常像素在边界区域，判为边界伪影
    if edge_ratio > 0.8:
        return (
            f"anomaly_too_close_to_edge: {edge_ratio:.2f} of anomaly pixels "
            f"within {config.edge_proximity_ratio:.2f} of border"
        )
    return ""
