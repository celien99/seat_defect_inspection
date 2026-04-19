"""座椅缺陷检测项目命令行入口。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .runtime_config import load_config, load_yolo_training_config
from .service import capture_samples, run_inspection, train_patchcore_models
from .yolo_training import train_yolo_model

DEFAULT_CONFIG_PATH = str(
    Path(__file__).resolve().parents[2] / "configs" / "seat_defect_inspection.example.json",
)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。"""
    parser = argparse.ArgumentParser(
        description="汽车座椅缺陷检测独立项目入口",
    )
    subparsers = parser.add_subparsers(required=True)

    train_patchcore_parser = subparsers.add_parser(
        "train-patchcore",
        help="按配置为每个机位训练 PatchCore 模型",
    )
    train_patchcore_parser.set_defaults(run=_run_train_patchcore)
    train_patchcore_parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="缺陷检测配置 JSON 路径",
    )

    capture_parser = subparsers.add_parser(
        "capture",
        help="从全部启用机位各抓取一帧并保存",
    )
    capture_parser.set_defaults(run=_run_capture)
    capture_parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="缺陷检测配置 JSON 路径",
    )
    capture_parser.add_argument(
        "--part-id",
        help="本次采图的部件编号，可覆盖配置中的默认值",
    )
    capture_parser.add_argument(
        "--output-dir",
        help="采图输出目录，可覆盖配置中的 capture_dir",
    )
    capture_parser.add_argument(
        "--save-to-train-good-dir",
        action="store_true",
        help="同时把图像拷贝到各机位的 train_good_dir",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="抓取每个机位一帧并执行融合检测",
    )
    inspect_parser.set_defaults(run=_run_inspect)
    inspect_parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="缺陷检测配置 JSON 路径",
    )
    inspect_parser.add_argument(
        "--part-id",
        help="本次检测的部件编号，可覆盖配置中的默认值",
    )

    train_yolo_parser = subparsers.add_parser(
        "train-yolo",
        help="训练用于座椅定位的 YOLO 模型",
    )
    train_yolo_parser.set_defaults(run=_run_train_yolo)
    train_yolo_parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="包含 yolo_training 配置块的 JSON 路径",
    )

    return parser


def _run_train_patchcore(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    summaries = train_patchcore_models(config)
    print(
        f"PatchCore 训练完成，共生成 {len(summaries)} 个机位模型，配置来源：{args.config}",
    )


def _run_capture(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    summary = capture_samples(
        config,
        part_id=args.part_id,
        output_dir=args.output_dir,
        save_to_train_good_dir=args.save_to_train_good_dir,
    )
    success_count = sum(1 for item in summary.records if item.status == "OK")
    failure_count = len(summary.records) - success_count
    print(
        f"采图完成，成功 {success_count} 路，失败 {failure_count} 路，"
        f"manifest：{summary.manifest_path}",
    )


def _run_inspect(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    result = run_inspection(config, part_id=args.part_id)
    print(
        f"缺陷检测完成，融合结果：{result.status}，报告已保存到：{config.output_json_path}",
    )


def _run_train_yolo(args: argparse.Namespace) -> None:
    config = load_yolo_training_config(args.config)
    summary = train_yolo_model(config)
    print(
        f"YOLO 训练完成，最佳权重：{summary['best_weights_path']}，输出目录：{summary['save_dir']}",
    )


def main(argv: Sequence[str] | None = None) -> None:
    """命令行主入口。"""
    parser = build_parser()
    args = parser.parse_args() if argv is None else parser.parse_args(list(argv))
    args.run(args)
