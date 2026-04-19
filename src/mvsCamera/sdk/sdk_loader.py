from __future__ import annotations

import ctypes
import os
import shutil
import sys
from pathlib import Path

DLL_NAME = "MvCameraControl.dll"
ENV_HINTS = (
    "MVS_HOME",
    "MVS_ROOT",
    "HIKROBOT_MVS_HOME",
    "MVCAM_COMMON_RUNENV",
)


class MvsSdkLoadError(RuntimeError):
    """海康 MVS SDK DLL 无法加载时抛出的异常。"""


def load_mvs_sdk_library(local_dll_path: Path | None = None):
    """从项目目录或系统已安装的 MVS 环境中加载控制 DLL。"""
    if os.name != "nt":
        raise MvsSdkLoadError("MVS SDK loading is only supported on Windows")

    attempts: list[str] = []
    for dll_path in _iter_candidate_dlls(local_dll_path):
        _register_dependency_dirs(dll_path)
        try:
            return ctypes.WinDLL(str(dll_path))
        except OSError as exc:
            attempts.append(f"{dll_path} -> {exc}")

    search_summary = "\n".join(f"- {item}" for item in attempts) or "- no DLL candidates were found"
    raise MvsSdkLoadError(
        "Could not load MvCameraControl.dll. Checked these candidates:\n"
        f"{search_summary}\n"
        "Install Hikrobot MVS, confirm Python and SDK bitness match, and ensure the SDK runtime "
        "directories are available on this machine.",
    )


def describe_mvs_sdk_candidates(local_dll_path: Path | None = None) -> list[str]:
    """返回当前会尝试加载的 DLL 候选路径，便于 Windows 现场排障。"""
    return [str(path) for path in _iter_candidate_dlls(local_dll_path)]


def _iter_candidate_dlls(local_dll_path: Path | None) -> list[Path]:
    """汇总所有可能的 DLL 位置，并去重。"""
    candidates: list[Path] = []
    seen: set[str] = set()

    def add_candidate(path: Path | None) -> None:
        if path is None:
            return
        resolved = str(path.resolve()) if path.exists() else str(path)
        key = resolved.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    add_candidate(local_dll_path)
    for root in _candidate_sdk_roots(local_dll_path):
        add_candidate(root / DLL_NAME)
        for dll_path in _find_dlls_under_root(root):
            add_candidate(dll_path)
    return [path for path in candidates if path.exists()]


def _candidate_sdk_roots(local_dll_path: Path | None) -> list[Path]:
    """收集可能的 SDK 根目录。"""
    roots: list[Path] = []
    seen: set[str] = set()

    def add_root(path: Path | None) -> None:
        if path is None or not path.exists():
            return
        try:
            resolved = str(path.resolve())
        except OSError:
            resolved = str(path)
        key = resolved.lower()
        if key in seen:
            return
        seen.add(key)
        roots.append(path)

    if local_dll_path is not None:
        add_root(local_dll_path.parent)

    for env_name in ENV_HINTS:
        # 兼容不同机器上常见的环境变量命名。
        env_value = os.environ.get(env_name)
        if env_value:
            add_root(Path(env_value))

    for discovered_root in _iter_registry_install_locations():
        add_root(discovered_root)

    for common_root in _iter_common_install_locations():
        add_root(common_root)

    mvs_executable = shutil.which("MVS.exe")
    if mvs_executable:
        add_root(Path(mvs_executable).resolve().parent)

    return roots


def _iter_common_install_locations() -> list[Path]:
    """枚举常见的 MVS 默认安装目录。"""
    program_files = [
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
    ]
    suffixes = [
        Path("MVS"),
        Path("Hikrobot") / "MVS",
        Path("Common Files") / "MVS",
    ]
    locations: list[Path] = []
    for base in program_files:
        if not base:
            continue
        for suffix in suffixes:
            locations.append(Path(base) / suffix)
    return locations


def _iter_registry_install_locations() -> list[Path]:
    """从 Windows 注册表中尝试发现 MVS 安装目录。"""
    if os.name != "nt":
        return []

    try:
        import winreg
    except ImportError:
        return []

    subkeys = [
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
        r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\MVS.exe",
    ]
    roots: list[Path] = []

    def read_value(key, value_name: str) -> str | None:
        try:
            value, _ = winreg.QueryValueEx(key, value_name)
        except OSError:
            return None
        return value if isinstance(value, str) and value else None

    for subkey in subkeys:
        try:
            root_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, subkey)
        except OSError:
            continue

        if subkey.endswith(r"App Paths\MVS.exe"):
            executable_path = read_value(root_key, "")
            if executable_path:
                roots.append(Path(executable_path).resolve().parent)
            continue

        index = 0
        while True:
            try:
                child_name = winreg.EnumKey(root_key, index)
            except OSError:
                break
            index += 1
            try:
                child_key = winreg.OpenKey(root_key, child_name)
            except OSError:
                continue

            display_name = read_value(child_key, "DisplayName") or ""
            install_location = read_value(child_key, "InstallLocation")
            if "MVS" not in display_name and "Hikrobot" not in display_name and "MvCamera" not in display_name:
                continue
            if install_location:
                roots.append(Path(install_location))
    return roots


def _find_dlls_under_root(root: Path) -> list[Path]:
    """在候选根目录下优先按常见目录结构查找 DLL，必要时再递归搜索。"""
    if not root.exists():
        return []

    direct_candidates = [
        root / DLL_NAME,
        root / "bin" / DLL_NAME,
        root / "Bin" / DLL_NAME,
        root / "Runtime" / DLL_NAME,
        root / "Development" / "Libraries" / _platform_dir_name() / DLL_NAME,
    ]
    discovered = [path for path in direct_candidates if path.exists()]
    if discovered:
        return discovered

    try:
        return list(root.rglob(DLL_NAME))
    except OSError:
        return []


def _register_dependency_dirs(dll_path: Path) -> None:
    """为 Python 3.8+ 注册 DLL 依赖目录，避免仅主 DLL 存在仍加载失败。"""
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return

    for directory in _dependency_directories(dll_path):
        try:
            add_dll_directory(str(directory))
        except OSError:
            continue


def _dependency_directories(dll_path: Path) -> list[Path]:
    """推断与主 DLL 同时需要加入搜索路径的运行时目录。"""
    directories: list[Path] = []
    seen: set[str] = set()

    def add_directory(path: Path | None) -> None:
        if path is None or not path.exists() or not path.is_dir():
            return
        key = str(path.resolve()).lower()
        if key in seen:
            return
        seen.add(key)
        directories.append(path)

    add_directory(dll_path.parent)

    root = _infer_sdk_root(dll_path)
    add_directory(root)
    for relative in (
        Path("Runtime"),
        Path("Runtime") / _platform_dir_name(),
        Path("Development") / "Libraries",
        Path("Development") / "Libraries" / _platform_dir_name(),
        Path("Bin"),
        Path("Applications"),
        Path("Applications") / _platform_dir_name(),
    ):
        add_directory(root / relative)

    for child in root.rglob("*"):
        if child.is_dir() and _platform_dir_name().lower() in child.name.lower():
            add_directory(child)

    return directories


def _infer_sdk_root(dll_path: Path) -> Path:
    """根据 DLL 所在位置反推 SDK 根目录。"""
    current = dll_path.parent
    markers = {"runtime", "development", "bin", "applications", "win64", "win32"}
    while current.parent != current:
        if current.name.lower() in markers:
            return current.parent
        current = current.parent
    return dll_path.parent


def _platform_dir_name() -> str:
    """根据 Python 位数选择 Win32/Win64 目录名。"""
    return "Win64" if sys.maxsize > 2**32 else "Win32"
