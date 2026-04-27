"""调试图档位定义与解析。"""

from __future__ import annotations

DEFAULT_DEBUG_ARTIFACT_MODE = "standard"

_DEBUG_ARTIFACT_GROUPS: dict[str, tuple[str, ...]] = {
    "standard": (
        "raw",
        "detections",
        "roi",
        "patchcore_input",
        "overlay",
    ),
    "full": (
        "raw",
        "preprocessed",
        "detections",
        "roi",
        "roi_texture",
        "patchcore_input",
        "foreground_weight",
        "target_mask",
        "valid_mask",
        "heatmap",
        "overlay",
    ),
}

_MODE_ALIASES = {
    "core": "standard",
    "minimal": "standard",
    "all": "full",
}


def resolve_debug_artifact_names(mode: str | None) -> set[str]:
    """返回当前调试图档位对应的文件键集合。"""
    normalized = (mode or DEFAULT_DEBUG_ARTIFACT_MODE).strip().lower()
    normalized = _MODE_ALIASES.get(normalized, normalized)
    if normalized not in _DEBUG_ARTIFACT_GROUPS:
        supported = ", ".join(sorted(_DEBUG_ARTIFACT_GROUPS))
        raise ValueError(f"未知 debug_artifact_mode `{mode}`，可选值：{supported}")
    return set(_DEBUG_ARTIFACT_GROUPS[normalized])
