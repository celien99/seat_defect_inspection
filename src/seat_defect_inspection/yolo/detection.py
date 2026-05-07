"""兼容旧导入路径的 YOLO 检测转发层。"""

from __future__ import annotations

import sys

from seat_defect_core.yolo import detection as _detection

sys.modules[__name__] = _detection
