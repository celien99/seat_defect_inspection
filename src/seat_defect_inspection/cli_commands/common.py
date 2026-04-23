"""CLI 子命令共用参数。"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_CONFIG_PATH = str(
    Path(__file__).resolve().parents[3] / "configs" / "seat_defect_inspection.mvs.json",
)


def add_config_argument(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    """为子命令补充统一的配置文件入口。"""
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=help_text,
    )


def add_seat_model_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: str,
) -> None:
    """为子命令补充统一的型号路由参数。"""
    parser.add_argument(
        "--seat-model-id",
        help=help_text,
    )
