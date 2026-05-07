"""兼容旧导入路径的 PatchCore runtime 转发层。"""

from __future__ import annotations

import sys

from seat_defect_core.patchcore import engine as _engine

sys.modules[__name__] = _engine
