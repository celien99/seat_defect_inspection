"""主检测流程输入类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class FramePacket:
    """单机位内部帧数据，供相机检测 pipeline 使用。"""

    camera_id: str
    """当前帧所属机位 ID。"""

    frame_id: str
    """当前帧或当前检测批次的编号。"""

    part_id: str
    """当前被检工件编号。"""

    source: str
    """输入来源标识，例如外部路径、URL 或 external://camera_id。"""

    source_kind: str
    """输入来源类型，例如 external_image。"""

    timestamp: str
    """当前帧时间戳。"""

    image: Any
    """BGR 图像数据。"""

    image_path: Optional[str] = None
    """当输入来自图片文件时保留原始路径。"""


@dataclass
class InspectionFrame:
    """外部系统传入主检测流程的单机位图片。"""

    camera_id: str
    """该图片对应的配置机位 ID。"""

    image: Any
    """外部已经采集好的 BGR 图像数据。"""

    source: Optional[str] = None
    """可选来源标识，仅用于报告和调试追踪。"""

    frame_id: Optional[str] = None
    """可选帧编号；为空时主流程自动生成。"""

    timestamp: Optional[str] = None
    """可选时间戳；为空时主流程自动生成。"""

    source_kind: str = "external_image"
    """来源类型，默认表示外部传入图片。"""

    error_reason: Optional[str] = None
    """上游采图或输入准备失败原因；存在时 core 直接生成 REJECT。"""


__all__ = [
    "FramePacket",
    "InspectionFrame",
]
