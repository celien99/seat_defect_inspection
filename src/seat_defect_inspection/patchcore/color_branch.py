"""兼容旧导入路径的颜色分支转发层。"""

from __future__ import annotations

import sys

from seat_defect_core.patchcore import color_branch as _color_branch

sys.modules[__name__] = _color_branch
