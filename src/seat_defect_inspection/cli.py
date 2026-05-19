"""座椅缺陷检测项目命令行入口。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .cli_commands import (
    register_benchmark_command,
    register_capture_command,
    register_inspect_command,
    register_inspect_folder_command,
    register_train_patchcore_command,
    register_train_yolo_command,
)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。"""
    parser = argparse.ArgumentParser(
        description="汽车座椅缺陷检测独立项目入口",
    )
    # 主入口只负责把各个子命令挂起来，避免业务继续堆回这里。
    subparsers = parser.add_subparsers(required=True)
    register_benchmark_command(subparsers)
    register_train_patchcore_command(subparsers)
    register_capture_command(subparsers)
    register_inspect_command(subparsers)
    register_inspect_folder_command(subparsers)
    register_train_yolo_command(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """命令行主入口。"""
    parser = build_parser()
    # argv 为 None 时沿用 argparse 默认行为，直接读取当前进程命令行参数。
    args = parser.parse_args(None if argv is None else list(argv))
    args.run(args)
