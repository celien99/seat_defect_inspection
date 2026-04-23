"""命令行子命令入口集合。"""

from .capture import register_capture_command
from .inspect import register_inspect_command
from .inspect_folder import register_inspect_folder_command
from .train_patchcore import register_train_patchcore_command
from .train_yolo import register_train_yolo_command

__all__ = [
    "register_capture_command",
    "register_inspect_command",
    "register_inspect_folder_command",
    "register_train_patchcore_command",
    "register_train_yolo_command",
]
