"""采图命令。"""

from __future__ import annotations

import argparse

from ..runtime_config import load_config
from .common import add_config_argument, add_seat_model_argument


def register_capture_command(subparsers) -> None:
    """注册采图命令。"""
    parser = subparsers.add_parser(
        "capture",
        help="从全部启用机位抓取一帧或多帧并保存",
    )
    parser.set_defaults(run=run_capture_command)
    add_config_argument(parser, help_text="缺陷检测配置 JSON/INI 路径")
    parser.add_argument(
        "--part-id",
        help="本次采图的部件编号，可覆盖配置中的默认值",
    )
    parser.add_argument(
        "--output-dir",
        help="采图输出目录，可覆盖配置中的 capture_dir",
    )
    add_seat_model_argument(parser, help_text="指定本次采图使用的座椅型号路由")
    parser.add_argument(
        "--save-to-train-good-dir",
        action="store_true",
        help="同时把图像复制到各机位的 train_good_dir",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="每个机位连续采集的张数，默认 1",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=0,
        help="同一机位连续采集之间的等待毫秒数，默认 0",
    )


def run_capture_command(args: argparse.Namespace) -> None:
    """执行采图命令并打印摘要。"""
    from ..service import capture_samples

    # 命令层只负责装配参数，采图细节继续留在 service。
    config = load_config(args.config)
    summary = capture_samples(
        config,
        part_id=args.part_id,
        output_dir=args.output_dir,
        seat_model_id=args.seat_model_id,
        save_to_train_good_dir=args.save_to_train_good_dir,
        count=args.count,
        interval_ms=args.interval_ms,
    )
    success_count = sum(1 for item in summary.records if item.status == "OK")
    failure_count = len(summary.records) - success_count
    print(
        f"采图完成，成功 {success_count} 路，失败 {failure_count} 路，"
        f"每机位张数：{args.count}，型号：{summary.seat_model_id or 'default'}，"
        f"manifest：{summary.manifest_path}",
    )
