"""座椅缺陷检测项目命令行入口。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .runtime_config import load_config, load_yolo_training_config
from .service import capture_samples, run_inspection, train_patchcore_models
from .yolo_training import train_yolo_model

DEFAULT_CONFIG_PATH = str(
    Path(__file__).resolve().parents[2] / "configs" / "seat_defect_inspection.mvs.json",
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
    train_patchcore_parser.add_argument(
        "--seat-model-id",
        help="指定要训练的座椅型号；多型号配置下不传时默认训练全部型号",
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
        "--seat-model-id",
        help="指定本次采图使用的座椅型号路由",
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
    inspect_parser.add_argument(
        "--seat-model-id",
        help="指定本次检测使用的座椅型号路由",
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
    train_yolo_parser.add_argument(
        "--seat-model-id",
        help="指定要使用的座椅型号训练配置；未传时优先使用顶层 yolo_training",
    )

    return parser


def _run_train_patchcore(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    summaries = train_patchcore_models(config, seat_model_id=args.seat_model_id)
    trained_models = sorted(
        {
            item["seat_model_id"]
            for item in summaries
            if item.get("seat_model_id") is not None
        },
    )
    model_scope = ",".join(trained_models) if trained_models else "default"
    print(
        f"PatchCore 训练完成，共生成 {len(summaries)} 个机位模型，"
        f"型号范围：{model_scope}，配置来源：{args.config}",
    )


def _run_capture(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    summary = capture_samples(
        config,
        part_id=args.part_id,
        output_dir=args.output_dir,
        seat_model_id=args.seat_model_id,
        save_to_train_good_dir=args.save_to_train_good_dir,
    )
    success_count = sum(1 for item in summary.records if item.status == "OK")
    failure_count = len(summary.records) - success_count
    print(
        f"采图完成，成功 {success_count} 路，失败 {failure_count} 路，"
        f"型号：{summary.seat_model_id or 'default'}，manifest：{summary.manifest_path}",
    )


def _run_inspect(args: argparse.Namespace) -> None:
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


def _run_train_yolo(args: argparse.Namespace) -> None:
    config = load_yolo_training_config(args.config, seat_model_id=args.seat_model_id)
    summary = train_yolo_model(config)
    print(
        f"YOLO 训练完成，型号：{summary.get('seat_model_id') or 'default'}，"
        f"最佳权重：{summary['best_weights_path']}，输出目录：{summary['save_dir']}",
    )


def main(argv: Sequence[str] | None = None) -> None:
    """命令行主入口。"""
    parser = build_parser()
    args = parser.parse_args() if argv is None else parser.parse_args(list(argv))
    args.run(args)
