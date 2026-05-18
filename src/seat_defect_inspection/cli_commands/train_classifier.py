"""缺陷分类器训练命令。"""

from __future__ import annotations

import argparse

from .common import add_config_argument, add_seat_model_argument


def register_train_classifier_command(subparsers) -> None:
    """注册缺陷分类器训练命令。"""
    parser = subparsers.add_parser(
        "train-classifier",
        help="按标注数据集训练缺陷分类器模型",
    )
    parser.set_defaults(run=run_train_classifier_command)
    add_config_argument(parser, help_text="缺陷检测配置 JSON/INI 路径")
    add_seat_model_argument(
        parser,
        help_text="指定要训练的座椅型号；多型号配置下不传时默认训练全部型号",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="标注数据集目录，包含 scratch/ stain/ wrinkle/ 等子目录",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="分类器模型输出路径（默认使用配置中的 classification.model_path）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="训练轮数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批次大小",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b0",
        choices=["efficientnet_b0", "efficientnet_b1", "mobilenet_v3_small"],
        help="分类器骨干网络",
    )


def run_train_classifier_command(args: argparse.Namespace) -> None:
    """执行缺陷分类器训练命令。"""
    from ..runtime_config import load_config
    from ..service import train_classifier_models

    config = load_config(args.config)
    summary = train_classifier_models(
        config,
        seat_model_id=args.seat_model_id,
        dataset_dir=args.dataset_dir,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        backbone=args.backbone,
    )
    print(f"分类器训练完成: {summary}")
