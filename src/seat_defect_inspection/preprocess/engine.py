"""兼容旧导入路径的预处理转发层。"""

from __future__ import annotations

import sys

from seat_defect_core.preprocess import engine as _engine

sys.modules[__name__] = _engine
