"""检测命令。"""

from __future__ import annotations

import argparse

from ..runtime_config import load_config
from .common import add_config_argument, add_seat_model_argument


def register_inspect_command(subparsers) -> None:
    """注册检测命令。"""
    parser = subparsers.add_parser(
        "inspect",
        help="抓取每个机位一帧并执行融合检测",
    )
    parser.set_defaults(run=run_inspect_command)
    add_config_argument(parser, help_text="缺陷检测配置 JSON 路径")
    parser.add_argument(
        "--part-id",
        help="本次检测的部件编号，可覆盖配置中的默认值",
    )
    add_seat_model_argument(parser, help_text="指定本次检测使用的座椅型号路由")


def run_inspect_command(args: argparse.Namespace) -> None:
    """执行检测命令并打印摘要。"""
    from ..service import run_inspection

    config = load_config(args.config)
    result = run_inspection(
        config,
        part_id=args.part_id,
        seat_model_id=args.seat_model_id,
    )
    print(
        f"缺陷检测完成，型号：{result.seat_model_id or 'default'}，"
        f"融合结果：{result.status}，报告已保存到：{config.output_json_path}",
    )
