from __future__ import annotations

from seat_defect_inspection.cvops import resolve_debug_artifact_names


def test_standard_artifact_mode_keeps_only_core_outputs() -> None:
    assert resolve_debug_artifact_names("standard") == {
        "raw",
        "detections",
        "roi",
        "patchcore_input",
        "overlay",
    }


def test_full_artifact_mode_keeps_all_debug_outputs() -> None:
    assert resolve_debug_artifact_names("full") == {
        "raw",
        "preprocessed",
        "detections",
        "roi",
        "roi_texture",
        "patchcore_input",
        "foreground_weight",
        "target_mask",
        "ignore_mask",
        "valid_mask",
        "heatmap",
        "overlay",
    }


def test_artifact_mode_aliases_are_supported() -> None:
    assert resolve_debug_artifact_names("core") == resolve_debug_artifact_names("standard")
    assert resolve_debug_artifact_names("all") == resolve_debug_artifact_names("full")


def test_unknown_artifact_mode_raises() -> None:
    try:
        resolve_debug_artifact_names("unexpected")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unexpected debug_artifact_mode")
