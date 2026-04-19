from __future__ import annotations

from ctypes import c_ubyte, memmove

import numpy as np

if __package__ in (None, ""):
    from mvsCamera.sdk.MvCameraControl_header import (
        PixelType_Gvsp_BayerBG8,
        PixelType_Gvsp_BayerBG10,
        PixelType_Gvsp_BayerBG10_Packed,
        PixelType_Gvsp_BayerBG12,
        PixelType_Gvsp_BayerBG12_Packed,
        PixelType_Gvsp_BayerGB8,
        PixelType_Gvsp_BayerGB10,
        PixelType_Gvsp_BayerGB10_Packed,
        PixelType_Gvsp_BayerGB12,
        PixelType_Gvsp_BayerGB12_Packed,
        PixelType_Gvsp_BayerGR8,
        PixelType_Gvsp_BayerGR10,
        PixelType_Gvsp_BayerGR10_Packed,
        PixelType_Gvsp_BayerGR12,
        PixelType_Gvsp_BayerGR12_Packed,
        PixelType_Gvsp_BayerRG8,
        PixelType_Gvsp_BayerRG10,
        PixelType_Gvsp_BayerRG10_Packed,
        PixelType_Gvsp_BayerRG12,
        PixelType_Gvsp_BayerRG12_Packed,
        PixelType_Gvsp_Mono8,
        PixelType_Gvsp_Mono10,
        PixelType_Gvsp_Mono10_Packed,
        PixelType_Gvsp_Mono12,
        PixelType_Gvsp_Mono12_Packed,
        PixelType_Gvsp_RGB8_Packed,
        PixelType_Gvsp_YUV422_Packed,
        PixelType_Gvsp_YUV422_YUYV_Packed,
    )
else:
    from .sdk.MvCameraControl_header import (
        PixelType_Gvsp_BayerBG8,
        PixelType_Gvsp_BayerBG10,
        PixelType_Gvsp_BayerBG10_Packed,
        PixelType_Gvsp_BayerBG12,
        PixelType_Gvsp_BayerBG12_Packed,
        PixelType_Gvsp_BayerGB8,
        PixelType_Gvsp_BayerGB10,
        PixelType_Gvsp_BayerGB10_Packed,
        PixelType_Gvsp_BayerGB12,
        PixelType_Gvsp_BayerGB12_Packed,
        PixelType_Gvsp_BayerGR8,
        PixelType_Gvsp_BayerGR10,
        PixelType_Gvsp_BayerGR10_Packed,
        PixelType_Gvsp_BayerGR12,
        PixelType_Gvsp_BayerGR12_Packed,
        PixelType_Gvsp_BayerRG8,
        PixelType_Gvsp_BayerRG10,
        PixelType_Gvsp_BayerRG10_Packed,
        PixelType_Gvsp_BayerRG12,
        PixelType_Gvsp_BayerRG12_Packed,
        PixelType_Gvsp_Mono8,
        PixelType_Gvsp_Mono10,
        PixelType_Gvsp_Mono10_Packed,
        PixelType_Gvsp_Mono12,
        PixelType_Gvsp_Mono12_Packed,
        PixelType_Gvsp_RGB8_Packed,
        PixelType_Gvsp_YUV422_Packed,
        PixelType_Gvsp_YUV422_YUYV_Packed,
    )


PIXEL_BYTE_DEPTH = {
    PixelType_Gvsp_Mono8: 1,
    PixelType_Gvsp_BayerGB8: 1,
    PixelType_Gvsp_RGB8_Packed: 3,
    PixelType_Gvsp_YUV422_Packed: 2,
}

MONO_PIXEL_TYPES = {
    PixelType_Gvsp_Mono8,
    PixelType_Gvsp_Mono10,
    PixelType_Gvsp_Mono10_Packed,
    PixelType_Gvsp_Mono12,
    PixelType_Gvsp_Mono12_Packed,
}

COLOR_PIXEL_TYPES = {
    PixelType_Gvsp_BayerGR8,
    PixelType_Gvsp_BayerRG8,
    PixelType_Gvsp_BayerGB8,
    PixelType_Gvsp_BayerBG8,
    PixelType_Gvsp_BayerGR10,
    PixelType_Gvsp_BayerRG10,
    PixelType_Gvsp_BayerGB10,
    PixelType_Gvsp_BayerBG10,
    PixelType_Gvsp_BayerGR12,
    PixelType_Gvsp_BayerRG12,
    PixelType_Gvsp_BayerGB12,
    PixelType_Gvsp_BayerBG12,
    PixelType_Gvsp_BayerGR10_Packed,
    PixelType_Gvsp_BayerRG10_Packed,
    PixelType_Gvsp_BayerGB10_Packed,
    PixelType_Gvsp_BayerBG10_Packed,
    PixelType_Gvsp_BayerGR12_Packed,
    PixelType_Gvsp_BayerRG12_Packed,
    PixelType_Gvsp_BayerGB12_Packed,
    PixelType_Gvsp_BayerBG12_Packed,
    PixelType_Gvsp_YUV422_Packed,
    PixelType_Gvsp_YUV422_YUYV_Packed,
}


def char_array_to_string(char_array) -> str:
    result = []
    for value in char_array:
        if value == 0:
            break
        result.append(chr(value))
    return "".join(result)


def int_to_ip(value: int) -> str:
    return ".".join(str((value >> shift) & 0xFF) for shift in (24, 16, 8, 0))


def copy_frame_buffer(buffer_pointer, data_size: int) -> np.ndarray:
    frame_buffer = (c_ubyte * data_size)()
    memmove(frame_buffer, buffer_pointer, data_size)
    return np.frombuffer(frame_buffer, count=data_size, dtype=np.uint8)


def frame_data_size(frame_info) -> int:
    bytes_per_pixel = PIXEL_BYTE_DEPTH.get(frame_info.enPixelType)
    if bytes_per_pixel is None:
        raise ValueError(f"unsupported pixel type: {frame_info.enPixelType}")
    return int(frame_info.nWidth * frame_info.nHeight * bytes_per_pixel)


def is_mono_pixel_type(pixel_type: int) -> bool:
    return pixel_type in MONO_PIXEL_TYPES


def is_color_pixel_type(pixel_type: int) -> bool:
    return pixel_type in COLOR_PIXEL_TYPES
