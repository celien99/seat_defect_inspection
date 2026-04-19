"""面向业务封装的海康相机控制器。

该模块参考仓库中已在 Windows 上跑通的 `mvs` 目录调用链，
目标是为 `src/mvsCamera` 提供稳定、可复用的单相机能力：
- SDK 初始化/反初始化
- 设备枚举与按 index / SN / IP / MAC 选择
- 打开设备、设置触发模式、开始取流
- 读取 OpenCV 可直接使用的 BGR 图像
"""

from __future__ import annotations

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
    """标准化设备信息。"""

    index: int
    tlayer_type: int
    serial_number: str | None = None
    mac_address: str | None = None
    ip_address: str | None = None
    model_name: str | None = None
    user_defined_name: str | None = None


@dataclass(slots=True)
class CameraLocator:
    """相机定位信息。"""

    device_index: int | None = 0
    serial_number: str | None = None
    ip_address: str | None = None
    mac_address: str | None = None


@dataclass(slots=True)
class CameraPropertyConfig:
    """相机运行时属性配置。"""

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
    """参考 Windows 已验证代码整理出的单相机控制器。"""

    sdk_initialized = False
    instance_count = 0

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

        self._initialize_sdk()
        HikCamera.instance_count += 1

    @classmethod
    def _initialize_sdk(cls) -> None:
        if cls.sdk_initialized:
            return
        ret = MvCamera.MV_CC_Initialize()
        if ret != 0:
            raise MvsCameraError(f"SDK initialize failed: {parse_error(ret)}")
        cls.sdk_initialized = True

    @classmethod
    def _finalize_sdk(cls) -> None:
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
        """打开目标相机。"""
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
        """开始取流。"""
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
        """设置触发模式。"""
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
        """应用额外的相机属性配置。"""
        config = self.property_config

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
        """设置 ROI 参数。"""
        if width is not None:
            self._set_int_value("Width", width)
        if height is not None:
            self._set_int_value("Height", height)
        if offset_x is not None:
            self._set_int_value("OffsetX", offset_x)
        if offset_y is not None:
            self._set_int_value("OffsetY", offset_y)

    def get_int_node(self, node_name: str) -> dict[str, int]:
        """读取整数节点的当前值与范围。"""
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
        """读取浮点节点的当前值与范围。"""
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
        """读取一帧图像，统一返回 BGR。"""
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
            if HikCamera.instance_count > 0:
                HikCamera.instance_count -= 1
            if HikCamera.instance_count == 0:
                HikCamera._finalize_sdk()

    def _safe_destroy(self) -> None:
        if self.opened:
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.opened = False

    def _resolve_device(self, devices: list[MvsDeviceInfo]) -> MvsDeviceInfo:
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
        value = MVCC_INTVALUE()
        memset(byref(value), 0, sizeof(value))
        ret = self.cam.MV_CC_GetIntValue(node_name, value)
        if ret != 0:
            raise MvsCameraError(f"Read int node '{node_name}' failed: {parse_error(ret)}")
        return int(value.nCurValue)

    def _get_float_value(self, node_name: str) -> float:
        value = MVCC_FLOATVALUE()
        memset(byref(value), 0, sizeof(value))
        ret = self.cam.MV_CC_GetFloatValue(node_name, value)
        if ret != 0:
            return 0.0
        return float(value.fCurValue)

    def _set_int_value(self, node_name: str, value: int) -> None:
        ret = self.cam.MV_CC_SetIntValue(node_name, int(value))
        if ret != 0:
            raise MvsCameraError(f"Set int node '{node_name}' failed: {parse_error(ret)}")

    def _set_float_value(self, node_name: str, value: float) -> None:
        ret = self.cam.MV_CC_SetFloatValue(node_name, float(value))
        if ret != 0:
            raise MvsCameraError(f"Set float node '{node_name}' failed: {parse_error(ret)}")

    def _set_bool_value(self, node_name: str, value: bool) -> None:
        ret = self.cam.MV_CC_SetBoolValue(node_name, bool(value))
        if ret != 0:
            raise MvsCameraError(f"Set bool node '{node_name}' failed: {parse_error(ret)}")

    def _set_mapped_enum_value(self, node_name: str, value: str, mapping: dict[str, int]) -> None:
        normalized = value.strip().lower()
        enum_value = mapping.get(normalized)
        if enum_value is None:
            options = ", ".join(sorted(mapping))
            raise MvsCameraError(f"Unsupported value '{value}' for '{node_name}', expected one of: {options}")
        ret = self.cam.MV_CC_SetEnumValue(node_name, enum_value)
        if ret != 0:
            raise MvsCameraError(f"Set enum node '{node_name}' failed: {parse_error(ret)}")

    def _apply_roi_config(self, config: CameraPropertyConfig) -> None:
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
        pixel_type = PIXEL_FORMAT_MAP.get(pixel_format.lower())
        if pixel_type is None:
            return
        ret = self.cam.MV_CC_SetEnumValue("PixelFormat", pixel_type)
        if ret != 0:
            # 某些机型不支持直接切到 BGR8，此时回退到原始格式 + 后续像素转换。
            return

    def _decode_frame(self, frame_buffer: Any, frame_info: Any) -> np.ndarray:
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
    if hasattr(device_info, "nMacAddrHigh") and hasattr(device_info, "nMacAddrLow"):
        high = device_info.nMacAddrHigh
        low = device_info.nMacAddrLow
        mac_int = (high << 32) | low
        return ":".join(f"{(mac_int >> (8 * i)) & 0xFF:02X}" for i in reversed(range(6)))
    gige_info = getattr(device_info.SpecialInfo, "stGigEInfo", None)
    if gige_info is not None and hasattr(gige_info, "chMacAddr"):
        return ":".join(f"{value:02X}" for value in gige_info.chMacAddr[:6])
    return None
