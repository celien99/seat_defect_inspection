"""兼容旧导入路径的 PatchCore 打分转发层。"""

from __future__ import annotations

import sys

from seat_defect_core.patchcore import scoring as _scoring

sys.modules[__name__] = _scoring
