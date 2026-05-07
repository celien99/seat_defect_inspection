"""兼容旧导入路径的图像质量门控转发层。"""

from __future__ import annotations

import sys

from seat_defect_core.cvops import quality as _quality

sys.modules[__name__] = _quality
