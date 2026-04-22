"""座椅缺陷检测项目命令行入口。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .runtime_config import load_config, load_yolo_training_config

DEFAULT_CONFIG_PATH = str(
    Path(__file__).resolve().parents[2] / "configs" / "seat_defect_inspection.mvs.json",
)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。"""
    parser = argparse.ArgumentParser(
        description="汽车座椅缺陷检测独立项目入口",
    )
    # 仍然按命令顺序手写展开，避免再套一层命令注册框架。
    subparsers = parser.add_subparsers(required=True)
    _build_train_patchcore_parser(subparsers)
    _build_capture_parser(subparsers)
    _build_inspect_parser(subparsers)
    _build_train_yolo_parser(subparsers)
    return parser


def _build_train_patchcore_parser(subparsers) -> None:
    """注册 PatchCore 训练命令。"""
    parser = subparsers.add_parser(
        "train-patchcore",
        help="按配置为每个机位训练 PatchCore 模型",
    )
    parser.set_defaults(run=_run_train_patchcore)
    _add_config_argument(parser, help_text="缺陷检测配置 JSON 路径")
    _add_seat_model_argument(
        parser,
        help_text="指定要训练的座椅型号；多型号配置下不传时默认训练全部型号",
    )


def _build_capture_parser(subparsers) -> None:
    """注册采图命令。"""
    parser = subparsers.add_parser(
        "capture",
        help="从全部启用机位抓取一帧或多帧并保存",
    )
    parser.set_defaults(run=_run_capture)
    _add_config_argument(parser, help_text="缺陷检测配置 JSON 路径")
    parser.add_argument(
        "--part-id",
        help="本次采图的部件编号，可覆盖配置中的默认值",
    )
    parser.add_argument(
        "--output-dir",
        help="采图输出目录，可覆盖配置中的 capture_dir",
    )
    _add_seat_model_argument(parser, help_text="指定本次采图使用的座椅型号路由")
    parser.add_argument(
        "--save-to-train-good-dir",
        action="store_true",
        help="同时把图像拷贝到各机位的 train_good_dir",
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


def _build_inspect_parser(subparsers) -> None:
    """注册检测命令。"""
    parser = subparsers.add_parser(
        "inspect",
        help="抓取每个机位一帧并执行融合检测",
    )
    parser.set_defaults(run=_run_inspect)
    _add_config_argument(parser, help_text="缺陷检测配置 JSON 路径")
    parser.add_argument(
        "--part-id",
        help="本次检测的部件编号，可覆盖配置中的默认值",
    )
    _add_seat_model_argument(parser, help_text="指定本次检测使用的座椅型号路由")


def _build_train_yolo_parser(subparsers) -> None:
    """注册 YOLO 训练命令。"""
    parser = subparsers.add_parser(
        "train-yolo",
        help="训练用于座椅定位的 YOLO 模型",
    )
    parser.set_defaults(run=_run_train_yolo)
    _add_config_argument(parser, help_text="包含 yolo_training 配置块的 JSON 路径")
    _add_seat_model_argument(
        parser,
        help_text="指定要使用的座椅型号训练配置；未传时优先使用顶层 yolo_training",
    )


def _add_config_argument(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    """为子命令补充统一的配置文件入口。"""
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=help_text,
    )


def _add_seat_model_argument(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    """为子命令补充统一的型号路由参数。"""
    parser.add_argument(
        "--seat-model-id",
        help=help_text,
    )


def _run_train_patchcore(args: argparse.Namespace) -> None:
    """执行 PatchCore 训练命令并打印摘要。"""
    from .service import train_patchcore_models

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


def _run_capture(args: argparse.Namespace) -> None:
    """执行采图命令并打印摘要。"""
    from .service import capture_samples

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


def _run_inspect(args: argparse.Namespace) -> None:
    """执行检测命令并打印摘要。"""
    from .service import run_inspection

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
    """执行 YOLO 训练命令并打印摘要。"""
    from .yolo import train_yolo_model

    config = load_yolo_training_config(args.config, seat_model_id=args.seat_model_id)
    summary = train_yolo_model(config)
    print(
        f"YOLO 训练完成，型号：{summary.get('seat_model_id') or 'default'}，"
        f"最佳权重：{summary['best_weights_path']}，输出目录：{summary['save_dir']}",
    )


def main(argv: Sequence[str] | None = None) -> None:
    """命令行主入口。"""
    parser = build_parser()
    # argv 为 None 时沿用 argparse 默认行为，直接读取当前进程命令行参数。
    args = parser.parse_args(None if argv is None else list(argv))
    args.run(args)
