"""几何类值对象。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BoundingBox:
    """图像坐标系中的水平矩形框。"""

    x1: float
    """左上角 x 坐标。"""

    y1: float
    """左上角 y 坐标。"""

    x2: float
    """右下角 x 坐标。"""

    y2: float
    """右下角 y 坐标。"""

    @property
    def width(self) -> float:
        """矩形宽度，坐标异常时返回 0。"""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """矩形高度，坐标异常时返回 0。"""
        return max(0.0, self.y2 - self.y1)


__all__ = ["BoundingBox"]
