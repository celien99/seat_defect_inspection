"""面向业务封装的海康相机控制器。

该模块参考仓库中已在 Windows 上跑通的 `mvs` 目录调用链，
目标是为 `src/mvsCamera` 提供稳定、可复用的单相机能力：
- SDK 初始化/反初始化
- 设备枚举与按 index / SN / IP / MAC 选择
- 打开设备、设置触发模式、开始取流
- 读取 OpenCV 可直接使用的 BGR 图像
"""

from __future__ import annotations

import threading
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .sdk.CameraParams_const import MV_ACCESS_Control, MV_ACCESS_Exclusive, MV_GIGE_DEVICE, MV_USB_DEVICE
from .sdk.MvCameraControl_class import MvCamera
from .sdk.MvCameraControl_header import (
    MV_CC_DEVICE_INFO,
    MV_CC_DEVICE_INFO_LIST,
    MV_CC_PIXEL_CONVERT_PARAM,
    MV_FRAME_OUT_INFO_EX,
    MVCC_FLOATVALUE,
    MVCC_INTVALUE,
    PixelType_Gvsp_BGR8_Packed,
    PixelType_Gvsp_Mono8,
    PixelType_Gvsp_RGB8_Packed,
)
from .sdk import MvErrorDefine_const as error_constants
from .pixel_utils import char_array_to_string, int_to_ip, is_color_pixel_type, is_mono_pixel_type

PIXEL_FORMAT_MAP = {
    "mono8": PixelType_Gvsp_Mono8,
    "bgr8": PixelType_Gvsp_BGR8_Packed,
    "rgb8": PixelType_Gvsp_RGB8_Packed,
}
EXPOSURE_AUTO_MODE_MAP = {
    "off": 0,
    "once": 1,
    "continuous": 2,
}
GAIN_AUTO_MODE_MAP = {
    "off": 0,
    "once": 1,
    "continuous": 2,
}
TRIGGER_SOURCE_SOFTWARE = 7
_ERROR_NAME_BY_CODE = {
    value: name
    for name, value in vars(error_constants).items()
    if name == "MV_OK" or name.startswith("MV_E_")
}


class MvsCameraError(RuntimeError):
    """海康相机控制异常。"""


@dataclass(slots=True)
class MvsDeviceInfo:
    """标准化设备信息。

    字段：
    - index: 当前设备在 SDK 枚举结果中的索引
    - tlayer_type: 传输层类型，通常是 GigE 或 USB
    - serial_number: 序列号
    - mac_address: MAC 地址
    - ip_address: 当前 IP，仅 GigE 相机通常可用
    - model_name: 设备型号名
    - user_defined_name: 相机用户自定义名称
    """

    index: int
    tlayer_type: int
    serial_number: str | None = None
    mac_address: str | None = None
    ip_address: str | None = None
    model_name: str | None = None
    user_defined_name: str | None = None


@dataclass(slots=True)
class CameraLocator:
    """相机定位信息。

    字段：
    - device_index: 按枚举顺序选机
    - serial_number: 按序列号选机
    - ip_address: 按 IP 选机
    - mac_address: 按 MAC 选机

    约束：
    运行时应尽量只使用一种选择方式，避免现场设备顺序变化造成误选。
    """

    device_index: int | None = 0
    serial_number: str | None = None
    ip_address: str | None = None
    mac_address: str | None = None


@dataclass(slots=True)
class CameraPropertyConfig:
    """相机运行时属性配置。

    字段：
    - exposure_auto / exposure_time_us: 曝光模式与曝光时间
    - gain_auto / gain: 增益模式与增益值
    - gamma: 相机端 Gamma
    - acquisition_frame_rate_enable / acquisition_frame_rate: 帧率控制
    - width / height / offset_x / offset_y: 采集 ROI
    - reverse_x / reverse_y: 镜像翻转
    """

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


class HikCamera:
    """参考 Windows 已验证代码整理出的单相机控制器。

    生命周期：
    1. `__init__` 初始化 SDK 计数与基础状态
    2. `open` 枚举并打开目标相机，写入触发模式和属性
    3. `start_grabbing` 申请缓冲区并开始取流
    4. `get_frame` 读取单帧并转成 BGR
    5. `close` 停止取流、关闭设备、必要时反初始化 SDK

    该类是整个 MVS 接入链中最靠近海康 SDK 的业务封装层。
    """

    sdk_initialized = False
    instance_count = 0
    _sdk_lock = threading.RLock()

    def __init__(
        self,
        locator: CameraLocator | None = None,
        trigger_mode: str = "continuous",
        pixel_format: str = "bgr8",
        property_config: CameraPropertyConfig | None = None,
    ) -> None:
        self.locator = locator or CameraLocator()
        self.trigger_mode = trigger_mode
        self.pixel_format = pixel_format
        self.property_config = property_config or CameraPropertyConfig()
        self.cam = MvCamera()
        self.device_list = MV_CC_DEVICE_INFO_LIST()
        self.frame_info = MV_FRAME_OUT_INFO_EX()
        self.payload_size = 0
        self.data_buf: Any | None = None
        self.opened = False
        self.grabbing = False
        self.width = 0
        self.height = 0
        self.fps = 0.0

        with HikCamera._sdk_lock:
            self._initialize_sdk()
            HikCamera.instance_count += 1

    @classmethod
    def _initialize_sdk(cls) -> None:
        """按进程级别初始化海康 SDK。

        当前实现通过类变量 `sdk_initialized` 保证重复创建相机对象时不会重复初始化。
        """
        with cls._sdk_lock:
            if cls.sdk_initialized:
                return
            ret = MvCamera.MV_CC_Initialize()
            if ret != 0:
                raise MvsCameraError(f"SDK initialize failed: {parse_error(ret)}")
            cls.sdk_initialized = True

    @classmethod
    def _finalize_sdk(cls) -> None:
        """在最后一个相机对象释放后反初始化 SDK。"""
        with cls._sdk_lock:
            if not cls.sdk_initialized:
                return
            ret = MvCamera.MV_CC_Finalize()
            if ret == 0:
                cls.sdk_initialized = False

    def enumerate_devices(self) -> list[MvsDeviceInfo]:
        """枚举当前可见设备。"""
        tlayer_type = MV_GIGE_DEVICE | MV_USB_DEVICE
        memset(byref(self.device_list), 0, sizeof(self.device_list))
        ret = MvCamera.MV_CC_EnumDevices(tlayer_type, self.device_list)
        if ret != 0:
            raise MvsCameraError(f"Enum devices failed: {parse_error(ret)}")

        devices: list[MvsDeviceInfo] = []
        for index in range(self.device_list.nDeviceNum):
            devices.append(self._build_device_info(index))
        return devices

    def open(self) -> MvsDeviceInfo:
        """打开目标相机。

        调用顺序：
        1. `enumerate_devices`
        2. `_resolve_device`
        3. `MV_CC_CreateHandle`
        4. `MV_CC_OpenDevice`
        5. 设置心跳、最佳包大小、触发模式、像素格式和业务属性
        6. 读取 Width/Height/FPS 作为当前相机状态
        """
        devices = self.enumerate_devices()
        if not devices:
            raise MvsCameraError("No MVS camera devices were found")

        selected = self._resolve_device(devices)
        device_info = cast(
            self.device_list.pDeviceInfo[selected.index],
            POINTER(MV_CC_DEVICE_INFO),
        ).contents

        ret = self.cam.MV_CC_CreateHandle(device_info)
        if ret != 0:
            raise MvsCameraError(f"CreateHandle failed: {parse_error(ret)}")

        try:
            ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Control, 0)
            if ret != 0:
                raise MvsCameraError(f"OpenDevice failed: {parse_error(ret)}")

            # GigE 相机先设置心跳和最佳包大小，和已验证的 Windows 代码保持一致。
            self.cam.MV_CC_SetIntValue("GevHeartbeatTimeout", 5000)
            if device_info.nTLayerType == MV_GIGE_DEVICE:
                packet_size = self.cam.MV_CC_GetOptimalPacketSize()
                if int(packet_size) > 0:
                    self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", int(packet_size))

            self.set_trigger_mode(self.trigger_mode == "software")
            self._try_set_pixel_format(self.pixel_format)
            self.apply_property_config()

            self.width = self._get_int_value("Width")
            self.height = self._get_int_value("Height")
            self.fps = self._get_float_value("AcquisitionFrameRate")
            self.opened = True
            return selected
        except Exception:
            self._safe_destroy()
            raise

    def start_grabbing(self) -> None:
        """开始取流。

        关键动作：
        - 先读取 `PayloadSize`
        - 为 SDK 输出帧分配缓冲区
        - 调用 `MV_CC_StartGrabbing`
        """
        if not self.opened:
            raise MvsCameraError("Camera is not opened")

        self.payload_size = self._get_int_value("PayloadSize")
        self.data_buf = (c_ubyte * self.payload_size)()
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise MvsCameraError(f"StartGrabbing failed: {parse_error(ret)}")
        self.grabbing = True

    def stop_grabbing(self) -> None:
        """停止取流。"""
        if not self.grabbing:
            return
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            raise MvsCameraError(f"StopGrabbing failed: {parse_error(ret)}")
        self.grabbing = False

    def set_trigger_mode(self, enable: bool) -> None:
        """设置触发模式。

        - `False`: 连续采集
        - `True`: 软件触发，并把触发源切到 `TriggerSoftware`
        """
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", 1 if enable else 0)
        if ret != 0:
            raise MvsCameraError(f"Set TriggerMode failed: {parse_error(ret)}")
        if enable:
            ret = self.cam.MV_CC_SetEnumValue("TriggerSource", TRIGGER_SOURCE_SOFTWARE)
            if ret != 0:
                raise MvsCameraError(f"Set TriggerSource failed: {parse_error(ret)}")

    def trigger_once(self) -> None:
        """软件触发一次采图。"""
        ret = self.cam.MV_CC_SetCommandValue("TriggerSoftware")
        if ret != 0:
            raise MvsCameraError(f"TriggerSoftware failed: {parse_error(ret)}")

    def apply_property_config(self) -> None:
        """应用额外的相机属性配置。

        说明：
        该函数把 `CameraPropertyConfig` 中的字段逐项写入相机节点，
        是现场相机参数下发的总入口。
        """
        config = self.property_config

        # 先应用翻转、曝光、增益等成像参数，再应用 ROI，避免局部节点依赖顺序问题。
        if config.reverse_x is not None:
            self._set_bool_value("ReverseX", config.reverse_x)
        if config.reverse_y is not None:
            self._set_bool_value("ReverseY", config.reverse_y)
        if config.exposure_auto is not None:
            self.set_exposure_auto(config.exposure_auto)
        if config.exposure_time_us is not None:
            self.set_exposure_time(config.exposure_time_us)
        if config.gain_auto is not None:
            self.set_gain_auto(config.gain_auto)
        if config.gain is not None:
            self.set_gain(config.gain)
        if config.gamma is not None:
            self.set_gamma(config.gamma)
        if config.acquisition_frame_rate_enable is not None:
            self._set_bool_value("AcquisitionFrameRateEnable", config.acquisition_frame_rate_enable)
        if config.acquisition_frame_rate is not None:
            self.set_acquisition_frame_rate(config.acquisition_frame_rate)
        self._apply_roi_config(config)

    def set_exposure_auto(self, mode: str) -> None:
        """设置曝光自动模式。"""
        self._set_mapped_enum_value("ExposureAuto", mode, EXPOSURE_AUTO_MODE_MAP)

    def set_exposure_time(self, exposure_time_us: float) -> None:
        """设置曝光时间，单位微秒。"""
        self._set_float_value("ExposureTime", exposure_time_us)

    def set_gain_auto(self, mode: str) -> None:
        """设置增益自动模式。"""
        self._set_mapped_enum_value("GainAuto", mode, GAIN_AUTO_MODE_MAP)

    def set_gain(self, gain: float) -> None:
        """设置模拟增益。"""
        self._set_float_value("Gain", gain)

    def set_gamma(self, gamma: float) -> None:
        """设置 Gamma。"""
        self._set_float_value("Gamma", gamma)

    def set_acquisition_frame_rate(self, fps: float) -> None:
        """设置采集帧率。"""
        self._set_float_value("AcquisitionFrameRate", fps)

    def set_roi(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
        offset_x: int | None = None,
        offset_y: int | None = None,
    ) -> None:
        """设置 ROI 参数。

        通常用于产线现场把采集区域裁到只覆盖座椅所在区域，减少冗余带宽与处理开销。
        """
        if width is not None:
            self._set_int_value("Width", width)
        if height is not None:
            self._set_int_value("Height", height)
        if offset_x is not None:
            self._set_int_value("OffsetX", offset_x)
        if offset_y is not None:
            self._set_int_value("OffsetY", offset_y)

    def get_int_node(self, node_name: str) -> dict[str, int]:
        """读取整数节点的当前值与范围。

        适合现场调试时查询 `Width`、`Height`、`OffsetX` 等节点的当前值和可配置范围。
        """
        value = MVCC_INTVALUE()
        memset(byref(value), 0, sizeof(value))
        ret = self.cam.MV_CC_GetIntValue(node_name, value)
        if ret != 0:
            raise MvsCameraError(f"Read int node '{node_name}' failed: {parse_error(ret)}")
        return {
            "current": int(value.nCurValue),
            "min": int(value.nMin),
            "max": int(value.nMax),
            "inc": int(value.nInc),
        }

    def get_float_node(self, node_name: str) -> dict[str, float]:
        """读取浮点节点的当前值与范围。

        适合现场调试时查询 `ExposureTime`、`Gain`、`AcquisitionFrameRate` 等节点。
        """
        value = MVCC_FLOATVALUE()
        memset(byref(value), 0, sizeof(value))
        ret = self.cam.MV_CC_GetFloatValue(node_name, value)
        if ret != 0:
            raise MvsCameraError(f"Read float node '{node_name}' failed: {parse_error(ret)}")
        return {
            "current": float(value.fCurValue),
            "min": float(value.fMin),
            "max": float(value.fMax),
        }

    def get_frame(self, timeout_ms: int = 1000) -> np.ndarray | None:
        """读取一帧图像，统一返回 BGR。

        调用顺序：
        1. 若为软件触发模式，先执行 `trigger_once`
        2. 调用 `MV_CC_GetOneFrameTimeout`
        3. 更新当前宽高
        4. 调用 `_decode_frame` 转成 OpenCV 可直接使用的 BGR

        返回：
        - 成功：`np.ndarray`，BGR 图像
        - 超时或未取到帧：`None`
        """
        if not self.grabbing or self.data_buf is None:
            raise MvsCameraError("Camera is not grabbing")

        if self.trigger_mode == "software":
            self.trigger_once()

        memset(byref(self.frame_info), 0, sizeof(self.frame_info))
        ret = self.cam.MV_CC_GetOneFrameTimeout(
            self.data_buf,
            self.payload_size,
            self.frame_info,
            timeout_ms,
        )
        if ret != 0:
            return None

        self.width = int(self.frame_info.nWidth)
        self.height = int(self.frame_info.nHeight)
        return self._decode_frame(self.data_buf, self.frame_info)

    def close(self) -> None:
        """关闭相机并释放 SDK 资源。"""
        try:
            if self.grabbing:
                self.stop_grabbing()
        finally:
            self._safe_destroy()
            with HikCamera._sdk_lock:
                if HikCamera.instance_count > 0:
                    HikCamera.instance_count -= 1
                if HikCamera.instance_count == 0:
                    HikCamera._finalize_sdk()

    def _safe_destroy(self) -> None:
        """在异常或正常关闭时安全释放 Handle 和 Device。"""
        if self.opened:
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.opened = False

    def _resolve_device(self, devices: list[MvsDeviceInfo]) -> MvsDeviceInfo:
        """根据 `CameraLocator` 从枚举结果中选出目标相机。

        选择优先级：
        serial_number -> ip_address -> mac_address -> device_index
        """
        locator = self.locator
        if locator.serial_number:
            for device in devices:
                if (device.serial_number or "").upper() == locator.serial_number.upper():
                    return device
            raise MvsCameraError(f"Camera with serial number '{locator.serial_number}' not found")

        if locator.ip_address:
            for device in devices:
                if device.ip_address == locator.ip_address:
                    return device
            raise MvsCameraError(f"Camera with IP '{locator.ip_address}' not found")

        if locator.mac_address:
            for device in devices:
                if (device.mac_address or "").upper() == locator.mac_address.upper():
                    return device
            raise MvsCameraError(f"Camera with MAC '{locator.mac_address}' not found")

        index = locator.device_index if locator.device_index is not None else 0
        if not 0 <= index < len(devices):
            raise MvsCameraError(f"Camera index {index} is out of range for {len(devices)} device(s)")
        return devices[index]

    def _build_device_info(self, index: int) -> MvsDeviceInfo:
        """把 SDK 原始设备信息转换为业务层可读的 `MvsDeviceInfo`。"""
        device_info = cast(
            self.device_list.pDeviceInfo[index],
            POINTER(MV_CC_DEVICE_INFO),
        ).contents
        if device_info.nTLayerType == MV_GIGE_DEVICE:
            gige_info = device_info.SpecialInfo.stGigEInfo
            return MvsDeviceInfo(
                index=index,
                tlayer_type=device_info.nTLayerType,
                serial_number=char_array_to_string(gige_info.chSerialNumber),
                mac_address=_extract_mac_address(device_info),
                ip_address=int_to_ip(gige_info.nCurrentIp),
                model_name=char_array_to_string(gige_info.chModelName),
                user_defined_name=char_array_to_string(gige_info.chUserDefinedName),
            )

        if device_info.nTLayerType == MV_USB_DEVICE:
            usb_info = device_info.SpecialInfo.stUsb3VInfo
            return MvsDeviceInfo(
                index=index,
                tlayer_type=device_info.nTLayerType,
                serial_number=char_array_to_string(usb_info.chSerialNumber),
                mac_address=_extract_mac_address(device_info),
                ip_address=None,
                model_name=char_array_to_string(usb_info.chModelName),
                user_defined_name=char_array_to_string(usb_info.chUserDefinedName),
            )

        return MvsDeviceInfo(index=index, tlayer_type=device_info.nTLayerType)

    def _get_int_value(self, node_name: str) -> int:
        """读取整数节点当前值。"""
        value = MVCC_INTVALUE()
        memset(byref(value), 0, sizeof(value))
        ret = self.cam.MV_CC_GetIntValue(node_name, value)
        if ret != 0:
            raise MvsCameraError(f"Read int node '{node_name}' failed: {parse_error(ret)}")
        return int(value.nCurValue)

    def _get_float_value(self, node_name: str) -> float:
        """读取浮点节点当前值。

        某些节点或机型不支持时返回 0.0，而不是直接中断打开流程。
        """
        value = MVCC_FLOATVALUE()
        memset(byref(value), 0, sizeof(value))
        ret = self.cam.MV_CC_GetFloatValue(node_name, value)
        if ret != 0:
            return 0.0
        return float(value.fCurValue)

    def _set_int_value(self, node_name: str, value: int) -> None:
        """写入整数节点。"""
        ret = self.cam.MV_CC_SetIntValue(node_name, int(value))
        if ret != 0:
            raise MvsCameraError(f"Set int node '{node_name}' failed: {parse_error(ret)}")

    def _set_float_value(self, node_name: str, value: float) -> None:
        """写入浮点节点。"""
        ret = self.cam.MV_CC_SetFloatValue(node_name, float(value))
        if ret != 0:
            raise MvsCameraError(f"Set float node '{node_name}' failed: {parse_error(ret)}")

    def _set_bool_value(self, node_name: str, value: bool) -> None:
        """写入布尔节点。"""
        ret = self.cam.MV_CC_SetBoolValue(node_name, bool(value))
        if ret != 0:
            raise MvsCameraError(f"Set bool node '{node_name}' failed: {parse_error(ret)}")

    def _set_mapped_enum_value(self, node_name: str, value: str, mapping: dict[str, int]) -> None:
        """把字符串枚举值映射为 SDK 所需整数枚举后写入节点。"""
        normalized = value.strip().lower()
        enum_value = mapping.get(normalized)
        if enum_value is None:
            options = ", ".join(sorted(mapping))
            raise MvsCameraError(f"Unsupported value '{value}' for '{node_name}', expected one of: {options}")
        ret = self.cam.MV_CC_SetEnumValue(node_name, enum_value)
        if ret != 0:
            raise MvsCameraError(f"Set enum node '{node_name}' failed: {parse_error(ret)}")

    def _apply_roi_config(self, config: CameraPropertyConfig) -> None:
        """仅当 ROI 相关参数被配置时才下发 ROI。"""
        if (
            config.width is None
            and config.height is None
            and config.offset_x is None
            and config.offset_y is None
        ):
            return
        self.set_roi(
            width=config.width,
            height=config.height,
            offset_x=config.offset_x,
            offset_y=config.offset_y,
        )

    def _try_set_pixel_format(self, pixel_format: str) -> None:
        """尝试直接把相机输出像素格式切到目标格式。

        若机型不支持，则静默保留原始像素格式，后续在 `_decode_frame` 中做软件转换。
        """
        pixel_type = PIXEL_FORMAT_MAP.get(pixel_format.lower())
        if pixel_type is None:
            return
        ret = self.cam.MV_CC_SetEnumValue("PixelFormat", pixel_type)
        if ret != 0:
            # 某些机型不支持直接切到 BGR8，此时回退到原始格式 + 后续像素转换。
            return

    def _decode_frame(self, frame_buffer: Any, frame_info: Any) -> np.ndarray:
        """把 SDK 原始帧数据解码为 BGR 图像。

        分支：
        - 已经是 BGR8: 直接 reshape
        - RGB8: 转为 BGR
        - Mono8: 转为 3 通道 BGR
        - 其他彩色格式: 先调用 SDK 像素转换再输出 BGR
        - 其他灰度格式: 先转换为 Mono8，再升成 BGR
        """
        pixel_type = int(frame_info.enPixelType)
        width = int(frame_info.nWidth)
        height = int(frame_info.nHeight)

        if pixel_type == PixelType_Gvsp_BGR8_Packed:
            frame = np.frombuffer(
                frame_buffer,
                count=width * height * 3,
                dtype=np.uint8,
            ).reshape(height, width, 3)
            return frame.copy()

        if pixel_type == PixelType_Gvsp_RGB8_Packed:
            rgb = np.frombuffer(
                frame_buffer,
                count=width * height * 3,
                dtype=np.uint8,
            ).reshape(height, width, 3)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if pixel_type == PixelType_Gvsp_Mono8:
            mono = np.frombuffer(
                frame_buffer,
                count=width * height,
                dtype=np.uint8,
            ).reshape(height, width)
            return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

        if is_color_pixel_type(pixel_type):
            converted = self._convert_pixel_type(
                frame_buffer=frame_buffer,
                frame_info=frame_info,
                destination_pixel_type=PixelType_Gvsp_BGR8_Packed,
                destination_size=width * height * 3,
            )
            return np.frombuffer(
                converted,
                count=width * height * 3,
                dtype=np.uint8,
            ).reshape(height, width, 3).copy()

        if is_mono_pixel_type(pixel_type):
            converted = self._convert_pixel_type(
                frame_buffer=frame_buffer,
                frame_info=frame_info,
                destination_pixel_type=PixelType_Gvsp_Mono8,
                destination_size=width * height,
            )
            mono = np.frombuffer(
                converted,
                count=width * height,
                dtype=np.uint8,
            ).reshape(height, width)
            return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

        raise MvsCameraError(f"Unsupported pixel type: {pixel_type}")

    def _convert_pixel_type(
        self,
        frame_buffer: Any,
        frame_info: Any,
        destination_pixel_type: int,
        destination_size: int,
    ) -> Any:
        """调用海康 SDK 做像素格式转换。

        这是 Bayer/YUV 等非 OpenCV 直接友好格式的关键兜底转换入口。
        """
        convert_param = MV_CC_PIXEL_CONVERT_PARAM()
        memset(byref(convert_param), 0, sizeof(convert_param))
        convert_param.nWidth = frame_info.nWidth
        convert_param.nHeight = frame_info.nHeight
        convert_param.enSrcPixelType = frame_info.enPixelType
        convert_param.pSrcData = cast(frame_buffer, POINTER(c_ubyte))
        convert_param.nSrcDataLen = frame_info.nFrameLen
        convert_param.enDstPixelType = destination_pixel_type

        destination_buffer = (c_ubyte * destination_size)()
        convert_param.pDstBuffer = cast(destination_buffer, POINTER(c_ubyte))
        convert_param.nDstBufferSize = destination_size

        ret = self.cam.MV_CC_ConvertPixelType(convert_param)
        if ret != 0:
            raise MvsCameraError(f"Convert pixel type failed: {parse_error(ret)}")
        return destination_buffer


def parse_error(ret: int) -> str:
    """把 SDK 错误码转换成人类可读文本。"""
    code = ret & 0xFFFFFFFF
    name = _ERROR_NAME_BY_CODE.get(code, "UNKNOWN_ERROR")
    return f"{name} (0x{code:x})"


def _extract_mac_address(device_info) -> str | None:
    """从不同设备结构中抽取 MAC 地址。"""
    if hasattr(device_info, "nMacAddrHigh") and hasattr(device_info, "nMacAddrLow"):
        high = device_info.nMacAddrHigh
        low = device_info.nMacAddrLow
        mac_int = (high << 32) | low
        return ":".join(f"{(mac_int >> (8 * i)) & 0xFF:02X}" for i in reversed(range(6)))
    gige_info = getattr(device_info.SpecialInfo, "stGigEInfo", None)
    if gige_info is not None and hasattr(gige_info, "chMacAddr"):
        return ":".join(f"{value:02X}" for value in gige_info.chMacAddr[:6])
    return None
