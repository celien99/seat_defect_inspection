"""Inference execution pipeline for benchmark evaluation."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, List, Optional

import cv2

from seat_defect_core.service.inspection import inspect_frames
from seat_defect_core.types import InspectionFrame

from .schemas import (
    BenchmarkRecord,
    BenchmarkSample,
    CameraBenchmarkRecord,
)

if TYPE_CHECKING:
    from .config import BenchmarkConfig
    from ..service.core import InspectionService


def run_round(
    service: "InspectionService",
    config: "BenchmarkConfig",
    round_name: str,
    samples: List[BenchmarkSample],
) -> List[BenchmarkRecord]:
    """Run inference on an entire benchmark round."""
    records: List[BenchmarkRecord] = []
    sample_count = len(samples)

    for idx, sample in enumerate(samples):
        frames = _build_frames(sample)
        t0 = perf_counter()
        result = inspect_frames(service, frames, part_id=sample.part_id)
        elapsed_ms = (perf_counter() - t0) * 1000.0

        cam_records = _build_camera_records(
            result.camera_results,
            artifacts_dir=config.artifacts_dir,
            round_name=round_name,
            part_id=sample.part_id,
        )
        predicted_status = result.status
        if predicted_status not in ("OK", "NG", "REJECT"):
            predicted_status = "NG" if predicted_status else "OK"

        record = BenchmarkRecord(
            sample=sample,
            predicted_status=predicted_status,
            decision_reason=result.decision_reason,
            camera_records=cam_records,
            inference_timing_ms=elapsed_ms,
        )
        records.append(record)

        marker = "OK" if predicted_status == "OK" else "x"
        print(f"  [{idx + 1:04d}/{sample_count}] {marker} {predicted_status}  part_id={sample.part_id}")

    return records


def _build_frames(sample: BenchmarkSample) -> List[InspectionFrame]:
    frames: List[InspectionFrame] = []
    for cid, path in sample.image_paths.items():
        image = cv2.imread(path)
        if image is None:
            frames.append(InspectionFrame(
                camera_id=cid,
                image=None,
                source=path,
                source_kind="file_read_error",
                frame_id=Path(path).stem,
                error_reason=f"Failed to read image: {path}",
            ))
        else:
            frames.append(InspectionFrame(
                camera_id=cid,
                image=image,
                source=path,
                source_kind="external_image",
                frame_id=Path(path).stem,
            ))
    return frames


def _build_camera_records(
    camera_results,
    artifacts_dir: Optional[str] = None,
    round_name: str = "",
    part_id: str = "",
) -> List[CameraBenchmarkRecord]:
    records: List[CameraBenchmarkRecord] = []
    for cam in camera_results:
        rec = CameraBenchmarkRecord(
            camera_id=cam.camera_id,
            predicted_status=cam.status,
        )
        if cam.texture_result is not None:
            t = cam.texture_result
            rec.anomaly_score = t.score
            rec.anomaly_threshold = t.threshold
        elif cam.region_results:
            scores = [
                r.texture_result.score
                for r in cam.region_results
                if r.texture_result is not None
            ]
            if scores:
                rec.anomaly_score = max(scores)

        # Export overlay image if artifacts_dir is configured
        if artifacts_dir:
            overlay_image = getattr(cam, "overlay_image", None)
            if overlay_image is not None:
                output_path = (
                    Path(artifacts_dir)
                    / f"{round_name}_{part_id}_{cam.camera_id}_overlay.png"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if cv2.imwrite(str(output_path), overlay_image):
                    rec.overlay_path = str(output_path)

        records.append(rec)
    return records
