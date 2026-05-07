"""兼容旧导入路径的 ROI runtime 转发层。"""

from __future__ import annotations

import sys

from seat_defect_core.cvops import roi as _roi

sys.modules[__name__] = _roi
