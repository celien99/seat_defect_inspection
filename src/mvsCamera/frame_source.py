"""MVS 相机源适配层。

上层只需要把海康相机当作一个 `cv2.VideoCapture` 风格输入源使用；
具体的设备选择、SDK 初始化和取流细节，都下沉到 `camera_controller.py`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse

from .camera_controller import CameraLocator, CameraPropertyConfig, HikCamera

MVS_SOURCE_SCHEME = "mvs"
DEFAULT_GRAB_TIMEOUT_MS = 1000


@dataclass(slots=True)
class MvsCameraSourceConfig:
    """工业相机源配置。
    支持的参数：
    - device_index: 设备索引
    - serial_number: 序列号
    - ip_address: IP 地址
    - mac_address: MAC 地址
    - grab_timeout_ms: 抓取超时时间（毫秒）
    - trigger_mode: 触发模式
    - pixel_format: 像素格式
    - exposure_auto: 曝光自动模式
    - exposure_time_us: 曝光时间（微秒）
    - gain_auto: 增益自动模式
    - gain: 增益
    - gamma: 伽马
    - acquisition_frame_rate_enable: 帧率使能
    - acquisition_frame_rate: 帧率
    - width: 宽度
    - height: 高度
    - offset_x: X 偏移
    - offset_y: Y 偏移
    - reverse_x: X 反转
    - reverse_y: Y 反转
    """

    device_index: int | None = 0
    serial_number: str | None = None
    ip_address: str | None = None
    mac_address: str | None = None
    grab_timeout_ms: int = DEFAULT_GRAB_TIMEOUT_MS
    trigger_mode: str = "continuous"
    pixel_format: str = "bgr8"
    exposure_auto: str | None = None
    exposure_time_us: float | None = None
    gain_auto: str | None = None
    gain: float | None = None
    gamma: float | None = None
    acquisition_frame_rate_enable: bool | None = None
    acquisition_frame_rate: float | None = None
    width: int | None = None
    height: int | None = None
    offset_x: int | None = None
    offset_y: int | None = None
    reverse_x: bool | None = None
    reverse_y: bool | None = None

    def to_locator(self) -> CameraLocator:
        """转换为相机定位配置。"""
        return CameraLocator(
            device_index=self.device_index,
            serial_number=self.serial_number,
            ip_address=self.ip_address,
            mac_address=self.mac_address,
        )

    def to_property_config(self) -> CameraPropertyConfig:
        """转换为相机属性配置。"""
        return CameraPropertyConfig(
            exposure_auto=self.exposure_auto,
            exposure_time_us=self.exposure_time_us,
            gain_auto=self.gain_auto,
            gain=self.gain,
            gamma=self.gamma,
            acquisition_frame_rate_enable=self.acquisition_frame_rate_enable,
            acquisition_frame_rate=self.acquisition_frame_rate,
            width=self.width,
            height=self.height,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
            reverse_x=self.reverse_x,
            reverse_y=self.reverse_y,
        )


def is_mvs_source(source: str) -> bool:
    """判断给定源是否为 `mvs://` 工业相机地址。"""
    return source.lower().startswith(f"{MVS_SOURCE_SCHEME}://")


def parse_mvs_source(source: str) -> MvsCameraSourceConfig:
    """解析 `mvs://` 源字符串为结构化配置。

    支持示例：
    - `mvs://0`
    - `mvs://index/1`
    - `mvs://sn/ABC123`
    - `mvs://ip/192.168.1.10`
    - `mvs://mac/AA:BB:CC:DD:EE:FF`
    - `mvs://0?timeout_ms=2000&trigger=software&pixel_format=bgr8`
    - `mvs://sn/ABC123?exposure_auto=off&exposure_time=6000&gain=8`
    """
    if not is_mvs_source(source):
        raise ValueError(f"Unsupported MVS source: {source}")

    parsed = urlparse(source)
    query = parse_qs(parsed.query)

    config = MvsCameraSourceConfig(
        grab_timeout_ms=int(query.get("timeout_ms", [DEFAULT_GRAB_TIMEOUT_MS])[0]),
        trigger_mode=query.get("trigger", ["continuous"])[0].lower(),
        pixel_format=query.get("pixel_format", ["bgr8"])[0].lower(),
        exposure_auto=_first_query_value(query, "exposure_auto"),
        exposure_time_us=_first_query_float(query, "exposure_time_us", "exposure_time"),
        gain_auto=_first_query_value(query, "gain_auto"),
        gain=_first_query_float(query, "gain"),
        gamma=_first_query_float(query, "gamma"),
        acquisition_frame_rate_enable=_first_query_bool(query, "frame_rate_enable", "acquisition_frame_rate_enable"),
        acquisition_frame_rate=_first_query_float(query, "fps", "frame_rate", "acquisition_frame_rate"),
        width=_first_query_int(query, "width"),
        height=_first_query_int(query, "height"),
        offset_x=_first_query_int(query, "offset_x"),
        offset_y=_first_query_int(query, "offset_y"),
        reverse_x=_first_query_bool(query, "reverse_x"),
        reverse_y=_first_query_bool(query, "reverse_y"),
    )

    _apply_selector_from_url(config, parsed)
    _apply_selector_from_query(config, query)
    _validate_selector(config)
    return config


def open_mvs_capture(source: str) -> "MvsCameraCapture":
    """根据字符串源创建工业相机取流对象。"""
    return MvsCameraCapture(parse_mvs_source(source))


class MvsCameraCapture:
    """对外暴露的相机取流适配器。"""

    def __init__(self, config: MvsCameraSourceConfig) -> None:
        self._config = config
        self._camera = HikCamera(
            locator=config.to_locator(),
            trigger_mode=config.trigger_mode,
            pixel_format=config.pixel_format,
            property_config=config.to_property_config(),
        )
        self._device_info = self._camera.open()
        self._camera.start_grabbing()

    def isOpened(self) -> bool:
        """返回相机是否已成功打开。"""
        return self._camera.opened

    def read(self) -> tuple[bool, Any]:
        """读取一帧 BGR 图像。"""
        frame = self._camera.get_frame(timeout_ms=self._config.grab_timeout_ms)
        if frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        """关闭相机。"""
        self._camera.close()

    def get(self, prop_id: int) -> float:
        """兼容 OpenCV 的属性查询。"""
        if prop_id == 3:
            return float(self._camera.width)
        if prop_id == 4:
            return float(self._camera.height)
        if prop_id == 5:
            return float(self._camera.fps)
        return 0.0

    @property
    def device_info(self):
        """返回当前连接设备的标准化信息。"""
        return self._device_info


def _apply_selector_from_url(config: MvsCameraSourceConfig, parsed) -> None:
    """从 URL 中应用选择器。"""
    selector = (parsed.netloc or "").strip()
    remainder = parsed.path.strip("/")

    if selector in {"index", "sn", "serial", "ip", "mac"} and remainder:
        _set_selector(config, selector, remainder)
        return

    candidate = selector or remainder
    if not candidate:
        return
    if candidate.isdigit():
        config.device_index = int(candidate)


def _apply_selector_from_query(config: MvsCameraSourceConfig, query: dict[str, list[str]]) -> None:
    """从查询参数中应用选择器。"""
    if "index" in query:
        config.device_index = int(query["index"][0])
    if "sn" in query:
        config.serial_number = query["sn"][0]
        config.device_index = None
    if "serial" in query:
        config.serial_number = query["serial"][0]
        config.device_index = None
    if "ip" in query:
        config.ip_address = query["ip"][0]
        config.device_index = None
    if "mac" in query:
        config.mac_address = query["mac"][0]
        config.device_index = None


def _set_selector(config: MvsCameraSourceConfig, selector: str, value: str) -> None:
    """设置选择器。"""
    if selector == "index":
        config.device_index = int(value)
        return
    config.device_index = None
    if selector in {"sn", "serial"}:
        config.serial_number = value
    elif selector == "ip":
        config.ip_address = value
    elif selector == "mac":
        config.mac_address = value


def _validate_selector(config: MvsCameraSourceConfig) -> None:
    """验证选择器。"""
    selected = [
        config.device_index is not None,
        bool(config.serial_number),
        bool(config.ip_address),
        bool(config.mac_address),
    ]
    if sum(selected) > 1:
        raise ValueError("Only one camera selector can be set among index / serial / ip / mac")


def _first_query_value(query: dict[str, list[str]], *keys: str) -> str | None:
    """获取第一个存在的键值。"""
    for key in keys:
        if key in query and query[key]:
            return query[key][0]
    return None


def _first_query_int(query: dict[str, list[str]], *keys: str) -> int | None:
    """获取第一个存在的键值并转换为 int。"""
    value = _first_query_value(query, *keys)
    if value is None:
        return None
    return int(value)


def _first_query_float(query: dict[str, list[str]], *keys: str) -> float | None:
    """获取第一个存在的键值并转换为 float。"""
    value = _first_query_value(query, *keys)
    if value is None:
        return None
    return float(value)


def _first_query_bool(query: dict[str, list[str]], *keys: str) -> bool | None:
    """获取第一个存在的键值并转换为 bool。"""
    value = _first_query_value(query, *keys)
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Unsupported boolean value '{value}', expected true/false")
