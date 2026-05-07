"""兼容旧导入路径的调试产物转发层。"""

from __future__ import annotations

import sys

from seat_defect_core.cvops import debug_artifacts as _debug_artifacts

sys.modules[__name__] = _debug_artifacts
