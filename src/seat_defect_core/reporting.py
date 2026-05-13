"""JSON 输出工具。"""

from __future__ import annotations

from pathlib import Path

from .serialization import inspection_result_to_dict
from .types import InspectionResult
from .util import write_json


def export_inspection_report(result: InspectionResult, output_path: str) -> Path:
    """写出一次检测任务的结果 JSON。"""
    path = Path(output_path)
    payload = inspection_result_to_dict(result)
    write_json(path, payload)
    return path
