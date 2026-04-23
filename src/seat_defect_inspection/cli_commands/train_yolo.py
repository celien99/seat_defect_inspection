"""YOLO 训练命令。"""

from __future__ import annotations

import argparse
from dataclasses import replace

from ..config import CameraConfig, InspectionConfig, PreprocessConfig
from ..runtime_config import load_config, load_yolo_training_config
from .common import add_config_argument, add_seat_model_argument


def register_train_yolo_command(subparsers) -> None:
    """注册 YOLO 训练命令。"""
    parser = subparsers.add_parser(
        "train-yolo",
        help="训练用于座椅定位的 YOLO 分割模型",
    )
    parser.set_defaults(run=run_train_yolo_command)
    add_config_argument(parser, help_text="包含 yolo_training 配置块的 JSON 路径")
    add_seat_model_argument(
        parser,
        help_text="指定要使用的座椅型号训练配置；未传时优先使用顶层 yolo_training",
    )


def run_train_yolo_command(args: argparse.Namespace) -> None:
    """执行 YOLO 训练命令并打印摘要。"""
    from ..yolo import train_yolo_model

    config = load_yolo_training_config(args.config, seat_model_id=args.seat_model_id)
    inspection_config = _try_load_inspection_config(args.config)
    if config.preprocess is None and inspection_config is not None:
        # 训练配置没单独写 preprocess 时，尽量复用线上检测配置，避免两套口径分叉。
        preprocess = _resolve_yolo_training_preprocess(
            inspection_config,
            args.seat_model_id,
        )
        if preprocess is not None:
            config = replace(config, preprocess=preprocess)

    summary = train_yolo_model(config)
    print(
        f"YOLO 训练完成，型号：{summary.get('seat_model_id') or 'default'}，"
        f"最佳权重：{summary['best_weights_path']}，输出目录：{summary['save_dir']}",
    )


def _try_load_inspection_config(path: str) -> InspectionConfig | None:
    """尽量加载完整检测配置，失败时回退到仅使用 yolo_training 配置。"""
    try:
        return load_config(path)
    except Exception:
        return None


def _resolve_yolo_training_preprocess(
    config: InspectionConfig,
    seat_model_id: str | None,
) -> PreprocessConfig | None:
    """从当前型号对应机位里推导一份统一的 preprocess。"""
    cameras = _resolve_training_cameras(config, seat_model_id)
    if not cameras:
        return None

    preprocess = cameras[0].preprocess
    if all(camera.preprocess == preprocess for camera in cameras[1:]):
        return preprocess

    raise ValueError(
        "train-yolo 检测到多个机位的 preprocess 配置不一致。"
        " 请在 yolo_training.preprocess 中显式配置一份统一预处理，"
        "或先统一相关机位的 preprocess 参数。"
    )


def _resolve_training_cameras(
    config: InspectionConfig,
    seat_model_id: str | None,
) -> list[CameraConfig]:
    """解析 train-yolo 当前应参考的机位列表。"""
    if config.seat_models:
        resolved_id = (
            seat_model_id
            or config.default_seat_model_id
            or config.seat_models[0].seat_model_id
        )
        for seat_model in config.seat_models:
            if seat_model.seat_model_id == resolved_id:
                return [camera for camera in seat_model.cameras if camera.enabled]
        available = ", ".join(item.seat_model_id for item in config.seat_models)
        raise ValueError(f"未知 seat_model_id `{resolved_id}`，可选值：{available}")
    return [camera for camera in config.cameras if camera.enabled]
