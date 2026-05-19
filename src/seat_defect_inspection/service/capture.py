"""采图流程。"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List

from media_inputs import infer_source_kind
from seat_defect_core.types import FramePacket
from seat_defect_core.util import build_model_scoped_root, write_image

from ..config import CameraConfig
from ..reporting import export_capture_manifest
from ..schemas import CaptureRecord, CaptureSummary

if TYPE_CHECKING:
    from .core import InspectionService


def capture_samples(
    service: "InspectionService",
    part_id: str | None = None,
    *,
    output_dir: str | None = None,
    seat_model_id: str | None = None,
    save_to_train_good_dir: bool = False,
    count: int = 1,
    interval_ms: int = 0,
) -> CaptureSummary:
    """每个启用机位抓取一帧或多帧并落盘。"""
    context = service.resolve_context(seat_model_id)
    resolved_part_id = part_id or service.config.part_id
    sample_count = max(1, int(count))
    wait_seconds = max(0, int(interval_ms)) / 1000.0
    run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    capture_root = (
        build_model_scoped_root(Path(output_dir or service.config.capture_dir), context.seat_model_id)
        / resolved_part_id
        / run_id
    )
    capture_root.mkdir(parents=True, exist_ok=True)

    records: List[CaptureRecord] = []
    for camera in context.cameras:
        for sample_index in range(sample_count):
            sample_number = sample_index + 1
            try:
                frame_packet = service.acquisition.capture(
                    camera.camera_id,
                    camera.source,
                    resolved_part_id,
                )
                output_path = _save_captured_frame(capture_root, frame_packet)
                train_good_path = None
                if save_to_train_good_dir:
                    train_good_path = _save_train_good_frame(camera, frame_packet)
                records.append(
                    CaptureRecord(
                        camera_id=frame_packet.camera_id,
                        frame_id=frame_packet.frame_id,
                        part_id=frame_packet.part_id,
                        source=frame_packet.source,
                        source_kind=frame_packet.source_kind,
                        timestamp=frame_packet.timestamp,
                        status="OK",
                        seat_model_id=context.seat_model_id,
                        output_path=output_path,
                        train_good_path=train_good_path,
                    )
                )
            except Exception as exc:
                records.append(
                    _build_capture_error_record(
                        camera,
                        part_id=resolved_part_id,
                        seat_model_id=context.seat_model_id,
                        sample_number=sample_number,
                        reason=str(exc),
                    )
                )
            if wait_seconds > 0 and sample_index < sample_count - 1:
                time.sleep(wait_seconds)

    summary = CaptureSummary(
        part_id=resolved_part_id,
        run_id=run_id,
        output_dir=str(capture_root),
        manifest_path=str(capture_root / "manifest.json"),
        seat_model_id=context.seat_model_id,
        records=records,
    )
    export_capture_manifest(summary)
    return summary


def _build_capture_error_record(
    camera: CameraConfig,
    *,
    part_id: str,
    seat_model_id: str | None,
    sample_number: int,
    reason: str,
) -> CaptureRecord:
    """补一条采图失败记录，保持 manifest 结构完整。"""
    return CaptureRecord(
        camera_id=camera.camera_id,
        frame_id="",
        part_id=part_id,
        source=camera.source,
        source_kind=infer_source_kind(camera.source),
        timestamp=datetime.now().astimezone().isoformat(),
        status="ERROR",
        seat_model_id=seat_model_id,
        reason=f"sample_{sample_number}:{reason}",
    )


def _save_captured_frame(capture_root: Path, frame_packet: FramePacket) -> str:
    """把采图结果写到批次目录。"""
    camera_dir = capture_root / frame_packet.camera_id
    camera_dir.mkdir(parents=True, exist_ok=True)
    output_path = camera_dir / f"{frame_packet.frame_id}.png"
    write_image(output_path, frame_packet.image)
    return str(output_path)


def _save_train_good_frame(camera: CameraConfig, frame_packet: FramePacket) -> str:
    """按机位配置把样本补写到 train_good_dir。"""
    if not camera.train_good_dir:
        raise ValueError(f"机位 `{camera.camera_id}` 未配置 `train_good_dir`")
    train_good_dir = Path(camera.train_good_dir)
    train_good_dir.mkdir(parents=True, exist_ok=True)
    output_path = train_good_dir / (
        f"{frame_packet.part_id}_{frame_packet.camera_id}_{frame_packet.frame_id}.png"
    )
    write_image(output_path, frame_packet.image)
    return str(output_path)
