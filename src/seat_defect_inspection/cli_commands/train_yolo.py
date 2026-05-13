"""YOLO 训练命令。"""

from __future__ import annotations

import argparse

from ..runtime_config import load_yolo_training_config
from .common import add_config_argument, add_seat_model_argument


def register_train_yolo_command(subparsers) -> None:
    """注册 YOLO 训练命令。"""
    parser = subparsers.add_parser(
        "train-yolo",
        help="训练用于座椅定位的 YOLO 分割模型",
    )
    parser.set_defaults(run=run_train_yolo_command)
    add_config_argument(parser, help_text="包含 yolo_training 配置块的 JSON/INI 路径")
    add_seat_model_argument(
        parser,
        help_text="指定要使用的座椅型号训练配置；未传时优先使用顶层 yolo_training",
    )


def run_train_yolo_command(args: argparse.Namespace) -> None:
    """执行 YOLO 训练命令并打印摘要。"""
    from ..yolo import train_yolo_model

    config = load_yolo_training_config(args.config, seat_model_id=args.seat_model_id)
    summary = train_yolo_model(config)
    print(
        f"YOLO 训练完成，型号：{summary.get('seat_model_id') or 'default'}，"
        f"最佳权重：{summary['best_weights_path']}，输出目录：{summary['save_dir']}",
    )
