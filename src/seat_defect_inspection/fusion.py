"""兼容旧导入路径的多机位融合转发层。"""

from __future__ import annotations

import sys

from seat_defect_core import fusion as _fusion

sys.modules[__name__] = _fusion
