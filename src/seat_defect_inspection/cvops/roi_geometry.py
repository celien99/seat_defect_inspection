"""兼容旧导入路径的 ROI 几何工具转发层。"""

from __future__ import annotations

import sys

from seat_defect_core.cvops import roi_geometry as _roi_geometry

sys.modules[__name__] = _roi_geometry
