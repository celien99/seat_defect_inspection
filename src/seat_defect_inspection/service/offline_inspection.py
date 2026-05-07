"""离线图片文件夹检测流程。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from seat_defect_core.util import write_json

from ..reporting import resolve_inspection_archive_path
from .inspection import run_inspection

if TYPE_CHECKING:
    from ..config import CameraConfig
    from .core import InspectionService

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(slots=True)
class _OfflineSample:
    """单个离线样本对应的机位图片集合。"""

    part_id: str
    source_map: dict[str, str]


def inspect_image_folder(
    service: "InspectionService",
    input_dir: str,
    *,
    seat_model_id: str | None = None,
    output_dir: str | None = None,
    part_id: str | None = None,
) -> dict[str, Any]:
    """从图片文件夹批量执行离线检测。"""
    input_root = Path(input_dir)
    if not input_root.is_dir():
        raise NotADirectoryError(f"离线检测输入目录不存在：{input_root}")

    context = service.resolve_context(seat_model_id)
    if not context.cameras:
        raise ValueError("当前配置没有启用机位，无法执行离线检测")

    camera_ids = [camera.camera_id for camera in context.cameras]
    samples = _discover_offline_samples(
        input_root,
        camera_ids,
        part_id=part_id,
    )
    if not samples:
        raise FileNotFoundError(f"输入目录中没有可检测样本：{input_root}")

    run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    run_root = _build_run_root(service, output_dir, run_id)
    latest_output_path = run_root / "reports" / "latest.json"
    debug_dir = run_root / "debug"
    original_output_json_path = service.config.output_json_path
    original_debug_dir = service.config.debug_dir
    original_sources = {
        camera.camera_id: camera.source
        for camera in context.cameras
    }

    # 离线批量检测只改输入源和输出路径，核心模型缓存与流程缓存继续复用。
    service.config.output_json_path = str(latest_output_path)
    service.config.debug_dir = str(debug_dir)

    try:
        records: list[dict[str, Any]] = []
        for sample in samples:
            _apply_sample_sources(context.cameras, sample.source_map)
            result = run_inspection(
                service,
                part_id=sample.part_id,
                seat_model_id=seat_model_id,
            )
            archive_path = resolve_inspection_archive_path(latest_output_path, result)
            records.append(
                {
                    "part_id": sample.part_id,
                    "status": result.status,
                    "decision_reason": result.decision_reason,
                    "report_path": str(archive_path),
                    "camera_sources": dict(sample.source_map),
                }
            )

        status_counter = Counter(record["status"] for record in records)
        summary_path = run_root / "summary.json"
        summary = {
            "run_id": run_id,
            "input_dir": str(input_root),
            "seat_model_id": context.seat_model_id,
            "sample_count": len(records),
            "ok_count": int(status_counter.get("OK", 0)),
            "ng_count": int(status_counter.get("NG", 0)),
            "reject_count": int(status_counter.get("REJECT", 0)),
            "run_root": str(run_root),
            "reports_dir": str(latest_output_path.parent),
            "debug_dir": str(debug_dir),
            "summary_path": str(summary_path),
            "records": records,
        }
        write_json(summary_path, summary)
        return summary
    finally:
        service.config.output_json_path = original_output_json_path
        service.config.debug_dir = original_debug_dir
        _restore_camera_sources(context.cameras, original_sources)


def _discover_offline_samples(
    input_root: Path,
    camera_ids: list[str],
    *,
    part_id: str | None,
) -> list[_OfflineSample]:
    """同时支持单样本、按样本分目录、按机位分目录三种离线输入布局。"""
    if _looks_like_camera_layout(input_root, camera_ids):
        return _discover_camera_layout_samples(
            input_root,
            camera_ids,
            part_id=part_id,
        )

    single_sample = _try_build_single_sample(input_root, camera_ids, part_id=part_id)
    if single_sample is not None:
        return [single_sample]

    sample_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    if not sample_dirs:
        raise FileNotFoundError(
            "离线检测目录中未发现样本。"
            " 需要满足以下任一结构："
            "1) 根目录直接放每个机位的图片；"
            "2) 根目录下每个子目录对应一个样本；"
            "3) 根目录下每个机位一个子目录。"
        )
    if part_id is not None:
        raise ValueError("--part-id 仅适用于单样本目录，不适用于批量子目录")
    return [
        _OfflineSample(
            part_id=sample_dir.name,
            source_map=_resolve_sample_source_map(sample_dir, camera_ids),
        )
        for sample_dir in sample_dirs
    ]


def _looks_like_camera_layout(input_root: Path, camera_ids: list[str]) -> bool:
    """判断根目录是否采用“每个机位一个子目录”的布局。"""
    child_dir_names = {path.name for path in input_root.iterdir() if path.is_dir()}
    return bool(camera_ids) and set(camera_ids).issubset(child_dir_names)


def _discover_camera_layout_samples(
    input_root: Path,
    camera_ids: list[str],
    *,
    part_id: str | None,
) -> list[_OfflineSample]:
    """解析按机位分目录的批量图片布局。"""
    indexed_images = {
        camera_id: _index_camera_dir(input_root / camera_id, camera_id)
        for camera_id in camera_ids
    }
    if not indexed_images:
        return []

    all_part_ids = set().union(*(mapping.keys() for mapping in indexed_images.values()))
    if not all_part_ids:
        raise FileNotFoundError(f"离线检测目录中没有图片：{input_root}")

    # 每个机位目录都只有一张图时，直接把它们视为一个样本，不强制文件名完全一致。
    if all(len(mapping) == 1 for mapping in indexed_images.values()):
        return [
            _OfflineSample(
                part_id=part_id or input_root.name,
                source_map={
                    camera_id: str(next(iter(indexed_images[camera_id].values())))
                    for camera_id in camera_ids
                },
            )
        ]

    if part_id is not None:
        raise ValueError("--part-id 仅适用于单样本目录或每机位单图目录，不适用于批量机位目录")

    for camera_id, mapping in indexed_images.items():
        missing = sorted(all_part_ids - set(mapping.keys()))
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"机位目录 `{camera_id}` 缺少对齐样本：{preview}"
                "。按机位分目录时，所有机位目录下的文件名（去后缀）必须一一对应。"
            )

    return [
        _OfflineSample(
            part_id=part_id,
            source_map={
                camera_id: str(indexed_images[camera_id][part_id])
                for camera_id in camera_ids
            },
        )
        for part_id in sorted(all_part_ids)
    ]


def _index_camera_dir(camera_dir: Path, camera_id: str) -> dict[str, Path]:
    """把单机位目录索引成 part_id -> image_path。"""
    if not camera_dir.is_dir():
        raise NotADirectoryError(f"离线检测缺少机位目录：{camera_dir}")

    indexed: dict[str, Path] = {}
    for image_path in sorted(_iter_image_files(camera_dir)):
        part_id = image_path.relative_to(camera_dir).with_suffix("").as_posix()
        if part_id in indexed:
            raise ValueError(
                f"机位 `{camera_id}` 的离线目录存在重复样本键：{part_id}"
                "。请避免同名不同后缀的图片同时出现。"
            )
        indexed[part_id] = image_path
    return indexed


def _try_build_single_sample(
    input_root: Path,
    camera_ids: list[str],
    *,
    part_id: str | None,
) -> _OfflineSample | None:
    """根目录本身就能凑齐所有机位图片时，直接当作单样本。"""
    try:
        source_map = _resolve_sample_source_map(input_root, camera_ids)
    except (FileNotFoundError, NotADirectoryError, ValueError):
        return None
    return _OfflineSample(
        part_id=part_id or input_root.name,
        source_map=source_map,
    )


def _resolve_sample_source_map(
    sample_dir: Path,
    camera_ids: list[str],
) -> dict[str, str]:
    """在一个样本目录内查找每个机位对应的图片。"""
    if not sample_dir.is_dir():
        raise NotADirectoryError(f"样本目录不存在：{sample_dir}")

    source_map: dict[str, str] = {}
    for camera_id in camera_ids:
        image_path = _find_camera_image(sample_dir, camera_id)
        if image_path is None:
            raise FileNotFoundError(
                f"样本目录 `{sample_dir}` 缺少机位 `{camera_id}` 的图片。"
                " 支持两种写法："
                f"`{camera_id}.jpg` 这类直接文件，或 `{camera_id}/xxx.jpg` 子目录。"
            )
        source_map[camera_id] = str(image_path)
    return source_map


def _find_camera_image(sample_dir: Path, camera_id: str) -> Path | None:
    """优先匹配 `camera_id.*`，其次匹配 `camera_id/` 子目录中的唯一图片。"""
    direct_candidates = sorted(
        path
        for path in sample_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.stem == camera_id
    )

    nested_candidates: list[Path] = []
    camera_dir = sample_dir / camera_id
    if camera_dir.is_dir():
        nested_candidates = sorted(_iter_image_files(camera_dir))

    candidates = direct_candidates + nested_candidates
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"样本目录 `{sample_dir}` 下机位 `{camera_id}` 匹配到多张图片，"
        "无法判断该使用哪一张。请保证每个机位每个样本只保留一张图。"
    )


def _iter_image_files(folder: Path):
    """递归遍历目录中的图片文件。"""
    return (
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _build_run_root(
    service: "InspectionService",
    output_dir: str | None,
    run_id: str,
) -> Path:
    """为一次离线批测构造独立输出目录。"""
    if output_dir is not None:
        base_dir = Path(output_dir)
    else:
        base_dir = Path(service.config.output_json_path).parent / "offline_inspect"
    run_root = base_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def _restore_camera_sources(
    cameras: list["CameraConfig"],
    original_sources: dict[str, str],
) -> None:
    """Restore camera sources after offline inspection so the caller can reuse the service."""
    for camera in cameras:
        camera.source = original_sources[camera.camera_id]


def _apply_sample_sources(cameras: list["CameraConfig"], source_map: dict[str, str]) -> None:
    """把当前样本的图片路径写回机位 source，复用原始检测主流程。"""
    for camera in cameras:
        camera.source = source_map[camera.camera_id]
