"""Configuration file loading for JSON and INI project configs."""

from __future__ import annotations

import configparser
import json
from pathlib import Path
from typing import Any

_INI_SUFFIXES = {".ini", ".cfg"}
_LIST_KEYS = {"box", "debug_artifact_names", "feature_layers"}
_BOOL_KEYS = {
    "backbone_pretrained",
    "cache",
    "color_insensitive_mode",
    "debug_artifacts_enabled",
    "defect_overrides_reject",
    "enabled",
    "pretrained",
    "reject_on_any_reject",
}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def load_inspection_payload(path: str) -> tuple[Path, dict[str, Any]]:
    """Read a JSON or INI config file and return the inspection payload."""
    config_path = Path(path).resolve()
    if config_path.suffix.lower() in _INI_SUFFIXES:
        return config_path.parent, _load_ini_inspection_payload(config_path)

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"配置文件顶层必须是对象：{config_path}")
    inspection_payload = payload.get("seat_defect_inspection", payload)
    if not isinstance(inspection_payload, dict):
        raise TypeError(f"`seat_defect_inspection` 必须是对象：{config_path}")
    return config_path.parent, inspection_payload


def _load_ini_inspection_payload(config_path: Path) -> dict[str, Any]:
    parser = configparser.ConfigParser(interpolation=None)
    read_files = parser.read(config_path, encoding="utf-8")
    if not read_files:
        raise FileNotFoundError(f"配置文件不存在或无法读取：{config_path}")

    payload: dict[str, Any] = {}
    for section_name in parser.sections():
        normalized_section = _normalize_section_name(section_name)
        items = _section_items(parser, section_name)
        _apply_ini_section(payload, normalized_section, items, config_path)

    return payload


def _normalize_section_name(section_name: str) -> str:
    normalized = section_name.strip()
    prefix = "seat_defect_inspection."
    if normalized.startswith(prefix):
        return normalized[len(prefix):]
    return normalized


def _section_items(
    parser: configparser.ConfigParser,
    section_name: str,
) -> dict[str, Any]:
    return {
        key: _parse_ini_value(key, value)
        for key, value in parser.items(section_name)
    }


def _apply_ini_section(
    payload: dict[str, Any],
    section_name: str,
    items: dict[str, Any],
    config_path: Path,
) -> None:
    if section_name in {"seat_defect_inspection", "inspection"}:
        payload.update(items)
        return
    if section_name == "fusion":
        payload.setdefault("fusion", {}).update(items)
        return
    if section_name == "yolo_training":
        payload.setdefault("yolo_training", {}).update(items)
        return
    if section_name.startswith("camera."):
        _apply_camera_section(payload, section_name.split("."), items, config_path)
        return
    if section_name.startswith("seat_model."):
        _apply_seat_model_section(payload, section_name.split("."), items, config_path)
        return
    raise ValueError(f"INI 配置包含未知 section `{section_name}`：{config_path}")


def _apply_seat_model_section(
    payload: dict[str, Any],
    parts: list[str],
    items: dict[str, Any],
    config_path: Path,
) -> None:
    if len(parts) < 2 or not parts[1]:
        raise ValueError(f"INI seat_model section 缺少型号 ID：{config_path}")
    seat_model_id = parts[1]
    seat_model = _ensure_named_item(
        payload.setdefault("seat_models", []),
        id_key="seat_model_id",
        expected_id=seat_model_id,
        section_items={},
        config_path=config_path,
    )
    rest = parts[2:]
    if not rest:
        seat_model.update(
            _with_required_id(
                items,
                id_key="seat_model_id",
                expected_id=seat_model_id,
                config_path=config_path,
            )
        )
        return
    if rest[0] == "yolo_training" and len(rest) == 1:
        seat_model.setdefault("yolo_training", {}).update(items)
        return
    if rest[0] == "camera":
        _apply_camera_section(seat_model, rest, items, config_path)
        return
    raise ValueError(f"INI seat_model section 不受支持 `{'.'.join(parts)}`：{config_path}")


def _apply_camera_section(
    container: dict[str, Any],
    parts: list[str],
    items: dict[str, Any],
    config_path: Path,
) -> None:
    if len(parts) < 2 or not parts[1]:
        raise ValueError(f"INI camera section 缺少机位 ID：{config_path}")
    camera_id = parts[1]
    camera = _ensure_named_item(
        container.setdefault("cameras", []),
        id_key="camera_id",
        expected_id=camera_id,
        section_items={},
        config_path=config_path,
    )
    rest = parts[2:]
    _apply_camera_rest(camera, camera_id, rest, items, config_path)


def _apply_camera_rest(
    camera: dict[str, Any],
    camera_id: str,
    rest: list[str],
    items: dict[str, Any],
    config_path: Path,
) -> None:
    if not rest:
        camera.update(
            _with_required_id(
                items,
                id_key="camera_id",
                expected_id=camera_id,
                config_path=config_path,
            )
        )
        return
    if rest[0] in {"quality", "detection", "patchcore", "color_branch"} and len(rest) == 1:
        camera.setdefault(rest[0], {}).update(items)
        return
    if rest[0] == "roi":
        _apply_roi_section(camera, rest[1:], items, config_path)
        return
    if rest[0] in {"region", "regions"}:
        _apply_region_section(camera, rest[1:], items, config_path)
        return
    raise ValueError(f"INI camera section 不受支持 `{'.'.join(['camera', camera_id, *rest])}`：{config_path}")


def _apply_roi_section(
    camera: dict[str, Any],
    rest: list[str],
    items: dict[str, Any],
    config_path: Path,
) -> None:
    roi = camera.setdefault("roi", {})
    if not rest:
        roi.update(items)
        return
    if rest == ["alignment"]:
        roi.setdefault("alignment", {}).update(items)
        return
    raise ValueError(f"INI roi section 不受支持：{config_path}")


def _apply_region_section(
    camera: dict[str, Any],
    rest: list[str],
    items: dict[str, Any],
    config_path: Path,
) -> None:
    if not rest or not rest[0]:
        raise ValueError(f"INI region section 缺少区域 ID：{config_path}")
    region_id = rest[0]
    region = _ensure_named_item(
        camera.setdefault("regions", []),
        id_key="region_id",
        expected_id=region_id,
        section_items={},
        config_path=config_path,
    )
    tail = rest[1:]
    if not tail:
        region.update(
            _with_required_id(
                items,
                id_key="region_id",
                expected_id=region_id,
                config_path=config_path,
            )
        )
        return
    if tail == ["patchcore"]:
        region.setdefault("patchcore", {}).update(items)
        return
    raise ValueError(f"INI region section 不受支持：{config_path}")


def _ensure_named_item(
    collection: list[dict[str, Any]],
    *,
    id_key: str,
    expected_id: str,
    section_items: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    payload = _with_required_id(
        section_items,
        id_key=id_key,
        expected_id=expected_id,
        config_path=config_path,
    )
    for item in collection:
        if item.get(id_key) == expected_id:
            item.update(payload)
            return item
    collection.append(payload)
    return payload


def _with_required_id(
    items: dict[str, Any],
    *,
    id_key: str,
    expected_id: str,
    config_path: Path,
) -> dict[str, Any]:
    current_id = items.get(id_key)
    if current_id is not None and str(current_id) != expected_id:
        raise ValueError(
            f"INI section ID `{expected_id}` 与字段 `{id_key}={current_id}` 不一致：{config_path}"
        )
    payload = dict(items)
    payload[id_key] = expected_id
    return payload


def _parse_ini_value(key: str, value: str) -> Any:
    stripped = _strip_quotes(value.strip())
    if key in _LIST_KEYS:
        return _parse_ini_list(stripped)
    if key in _BOOL_KEYS:
        return _parse_ini_bool(stripped)
    return stripped


def _parse_ini_list(value: str) -> list[str]:
    stripped = value.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1].strip()
    if not stripped:
        return []
    return [_strip_quotes(item.strip()) for item in stripped.split(",")]


def _parse_ini_bool(value: str) -> bool | str:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return value


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


__all__ = [
    "load_inspection_payload",
]
