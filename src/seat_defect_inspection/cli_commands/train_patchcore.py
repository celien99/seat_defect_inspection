"""PatchCore 训练命令。"""

from __future__ import annotations

import argparse

from ..runtime_config import load_config
from .common import add_config_argument, add_seat_model_argument


def register_train_patchcore_command(subparsers) -> None:
    """注册 PatchCore 训练命令。"""
    parser = subparsers.add_parser(
        "train-patchcore",
        help="按配置为每个机位训练 PatchCore 模型",
    )
    parser.set_defaults(run=run_train_patchcore_command)
    add_config_argument(parser, help_text="缺陷检测配置 JSON/INI 路径")
    add_seat_model_argument(
        parser,
        help_text="指定要训练的座椅型号；多型号配置下不传时默认训练全部型号",
    )


def run_train_patchcore_command(args: argparse.Namespace) -> None:
    """执行 PatchCore 训练命令并打印摘要。"""
    from ..service import train_patchcore_models

    config = load_config(args.config)
    summaries = train_patchcore_models(config, seat_model_id=args.seat_model_id)
    # 训练可能覆盖多个型号，这里压成一行输出，便于命令行快速查看。
    model_scope = ",".join(
        sorted(
            {
                item["seat_model_id"]
                for item in summaries
                if item.get("seat_model_id") is not None
            }
        )
    ) or "default"
    print(
        f"PatchCore 训练完成，共生成 {len(summaries)} 个机位模型，"
        f"型号范围：{model_scope}，配置来源：{args.config}",
    )
