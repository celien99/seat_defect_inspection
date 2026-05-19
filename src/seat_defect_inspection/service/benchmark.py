"""Benchmark inspection pipeline with good/defect/mixed datasets."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from .core import InspectionService


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

BENCHMARK_DATA_DIR = Path(__file__).resolve().parents[4] / "benchmark_data"
ROUNDS = ("good", "defect", "mixed")


def run_benchmark(
    service: "InspectionService",
    rounds: tuple[str, ...] = ROUNDS,
) -> Dict[str, dict]:
    """Run benchmark inspection on selected rounds and report metrics.

    Parameters
    ----------
    rounds:
        Which rounds to run, e.g. ``("good",)`` or ``("good", "defect")``.
        Defaults to all three.
    """
    if not BENCHMARK_DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Benchmark data directory not found: {BENCHMARK_DATA_DIR}\n"
            "Expected structure:\n"
            "  benchmark_data/\n"
            "  ├── good/        (all OK samples)\n"
            "  │   ├── cam_0/\n"
            "  │   └── cam_1/\n"
            "  ├── defect/      (all NG samples)\n"
            "  │   ├── cam_0/\n"
            "  │   └── cam_1/\n"
            "  └── mixed/       (mixed OK/NG samples)\n"
            "      ├── cam_0/\n"
            "      └── cam_1/"
        )

    context = service.resolve_context(None)
    if not context.cameras:
        raise ValueError("No cameras configured")

    camera_ids = [c.camera_id for c in context.cameras]

    original_sources = {c.camera_id: c.source for c in context.cameras}

    results: Dict[str, dict] = {}
    for round_name in rounds:
        round_dir = BENCHMARK_DATA_DIR / round_name
        if not round_dir.is_dir():
            print(f"[benchmark] Skipping '{round_name}' — directory not found: {round_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"  Benchmark round: {round_name}")
        print(f"{'='*60}")

        counts = _run_single_round(service, context.cameras, round_dir, camera_ids)
        results[round_name] = counts

    _restore_sources(context.cameras, original_sources)

    _print_summary(results, camera_ids, rounds)
    return results


def _run_single_round(
    service: "InspectionService",
    cameras,
    round_dir: Path,
    camera_ids: List[str],
) -> dict:
    """Run inspection on all samples in a round directory."""
    from .inspection import run_inspection

    camera_images = _collect_camera_images(round_dir, camera_ids)
    sample_count = len(next(iter(camera_images.values())))

    status_counter: Counter = Counter()
    stats = {"total": sample_count, "ok": 0, "ng": 0, "reject": 0}

    for idx in range(sample_count):
        source_map = {
            cid: str(camera_images[cid][idx]) for cid in camera_ids
        }
        _apply_sample_sources(cameras, source_map)

        result = run_inspection(service, part_id=f"{round_dir.name}_{idx:04d}")
        status_counter[result.status] += 1

        marker = "✓" if result.status == "OK" else "✗"
        print(f"  [{idx+1:04d}/{sample_count}] {marker} {result.status}")

    stats["ok"] = status_counter.get("OK", 0)
    stats["ng"] = status_counter.get("NG", 0)
    stats["reject"] = status_counter.get("REJECT", 0)

    ok_rate = stats["ok"] / stats["total"] * 100 if stats["total"] else 0
    ng_rate = stats["ng"] / stats["total"] * 100 if stats["total"] else 0
    reject_rate = stats["reject"] / stats["total"] * 100 if stats["total"] else 0

    print(f"\n  Results: OK={stats['ok']} ({ok_rate:.1f}%), "
          f"NG={stats['ng']} ({ng_rate:.1f}%), "
          f"REJECT={stats['reject']} ({reject_rate:.1f}%)")

    return stats


def _collect_camera_images(
    round_dir: Path,
    camera_ids: List[str],
) -> Dict[str, List[Path]]:
    """Collect images from each camera subdirectory, sorted by name."""
    result: Dict[str, List[Path]] = {}
    counts: List[int] = []

    for cid in camera_ids:
        cam_dir = round_dir / cid
        if not cam_dir.is_dir():
            raise FileNotFoundError(
                f"Camera directory not found: {cam_dir}\n"
                f"Expected subdirectory '{cid}' under {round_dir}"
            )
        images = sorted(
            p for p in cam_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise FileNotFoundError(f"No images found in {cam_dir}")
        result[cid] = images
        counts.append(len(images))

    if len(set(counts)) != 1:
        detail = ", ".join(f"{cid}={len(result[cid])}" for cid in camera_ids)
        raise ValueError(
            f"Camera image counts must be equal, got: {detail}"
        )

    return result


def _apply_sample_sources(cameras, source_map: Dict[str, str]) -> None:
    for camera in cameras:
        camera.source = source_map[camera.camera_id]


def _restore_sources(cameras, original_sources: Dict[str, str]) -> None:
    for camera in cameras:
        camera.source = original_sources[camera.camera_id]


def _print_summary(results: dict, camera_ids: List[str], rounds: tuple[str, ...]) -> None:
    """Print final benchmark summary with combined metrics."""
    print(f"\n{'='*60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  Cameras: {', '.join(camera_ids)}")

    good = results.get("good")
    defect = results.get("defect")

    # ---- per-round detail ----
    for round_name in rounds:
        r = results.get(round_name)
        if r is None:
            continue
        total = r["total"]
        ok = r["ok"]
        ng = r["ng"]
        reject = r["reject"]

        if round_name == "good":
            label = "Good (all OK)"
            fp_rate = (ng + reject) / total * 100 if total else 0
            print(f"\n  [{label}]")
            print(f"    Samples: {total}")
            print(f"    OK: {ok}  |  NG: {ng}  |  REJECT: {reject}")
            print(f"    False positive rate: {fp_rate:.1f}%")
        elif round_name == "defect":
            label = "Defect (all NG)"
            miss_rate = ok / total * 100 if total else 0
            detection_rate = ng / total * 100 if total else 0
            print(f"\n  [{label}]")
            print(f"    Samples: {total}")
            print(f"    OK: {ok}  |  NG: {ng}  |  REJECT: {reject}")
            print(f"    Miss rate: {miss_rate:.1f}%  |  Detection rate: {detection_rate:.1f}%")
        else:
            label = "Mixed"
            print(f"\n  [{label}]")
            print(f"    Samples: {total}")
            print(f"    OK: {ok}  |  NG: {ng}  |  REJECT: {reject}")
            print(f"    OK rate: {ok/total*100:.1f}%  |  NG rate: {ng/total*100:.1f}%")

    # ---- combined metrics from Good + Defect ----
    if good is not None and defect is not None:
        # Good round: all expected OK → TN = OK, FP = NG + REJECT
        TN = good["ok"]
        FP = good["ng"] + good["reject"]
        # Defect round: all expected NG → TP = NG, FN = OK
        TP = defect["ng"]
        FN = defect["ok"]

        precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (TP + TN) / (TP + TN + FP + FN) * 100 if (TP + TN + FP + FN) > 0 else 0.0

        print(f"\n  [Combined Metrics (Good + Defect)]")
        print(f"    TP={TP}  TN={TN}  FP={FP}  FN={FN}")
        print(f"    Precision (精准率): {precision:.1f}%")
        print(f"    Recall    (召回率): {recall:.1f}%")
        print(f"    F1 Score  (F1 值):  {f1:.1f}%")
        print(f"    Accuracy  (准确率): {accuracy:.1f}%")

    print(f"\n{'='*60}\n")
