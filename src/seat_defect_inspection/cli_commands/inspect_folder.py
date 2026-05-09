"""离线图片文件夹检测命令。"""

from __future__ import annotations

import argparse

from ..runtime_config import load_config
from .common import add_config_argument, add_seat_model_argument


def register_inspect_folder_command(subparsers) -> None:
    """注册离线图片文件夹检测命令。"""
    parser = subparsers.add_parser(
        "inspect-folder",
        help="从本地图片文件夹批量执行离线检测，不调用真机",
    )
    parser.set_defaults(run=run_inspect_folder_command)
    add_config_argument(parser, help_text="缺陷检测配置 JSON/INI 路径")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="离线图片根目录，支持单样本、按样本分目录、按机位分目录三种布局",
    )
    parser.add_argument(
        "--output-dir",
        help="离线检测输出根目录；未传时默认输出到 results.json 同级 offline_inspect 目录",
    )
    parser.add_argument(
        "--part-id",
        help="单样本目录下可手动指定工件编号；批量目录下不允许使用",
    )
    add_seat_model_argument(parser, help_text="指定本次离线检测使用的座椅型号路由")


def run_inspect_folder_command(args: argparse.Namespace) -> None:
    """执行离线图片文件夹检测并打印摘要。"""
    from ..service import inspect_image_folder

    config = load_config(args.config)
    summary = inspect_image_folder(
        config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        part_id=args.part_id,
        seat_model_id=args.seat_model_id,
    )
    print(
        f"离线检测完成，样本数：{summary['sample_count']}，"
        f"OK：{summary['ok_count']}，NG：{summary['ng_count']}，REJECT：{summary['reject_count']}，"
        f"汇总：{summary['summary_path']}",
    )
