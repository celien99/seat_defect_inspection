"""兼容旧导入路径的 PatchCore 特征提取转发层。"""

from __future__ import annotations

import sys

from seat_defect_core.patchcore import features as _features

sys.modules[__name__] = _features
