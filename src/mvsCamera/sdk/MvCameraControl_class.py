# -- coding: utf-8 --
"""ctypes 版 MVS SDK Python 包装层。

该文件主要负责把海康官方 DLL 暴露成 Python 可调用接口。
这里保留了与 SDK 原生函数名接近的形式，便于对照官方示例和文档排障。
"""

import ctypes
import os
import sys
from pathlib import Path

from ctypes import *

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from mvsCamera.sdk.CameraParams_const import *
    from mvsCamera.sdk.CameraParams_header import *
    from mvsCamera.sdk.MvCameraControl_header import *
    from mvsCamera.sdk.MvErrorDefine_const import *
    from mvsCamera.sdk.PixelType_header import *
    from mvsCamera.sdk.sdk_loader import MvsSdkLoadError, load_mvs_sdk_library
else:
    from .CameraParams_const import *
    from .CameraParams_header import *
    from .MvCameraControl_header import *
    from .MvErrorDefine_const import *
    from .PixelType_header import *
    from .sdk_loader import MvsSdkLoadError, load_mvs_sdk_library


class _MissingSdkProxy:
    """当 SDK 未成功加载时，延迟在首次调用 DLL 符号时给出明确报错。"""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def __getattr__(self, name: str):
        raise RuntimeError(
            "MvCamera SDK is unavailable. "
            "Please run on Windows with Hikrobot MVS installed. "
            "If MVS is already installed, make sure its runtime DLL directories are present "
            "and that Python bitness matches the SDK bitness.\n"
            f"Loader details: {self._error}",
        ) from self._error


def _load_sdk_library():
    """优先尝试项目本地 DLL，其次回退到系统安装的 MVS 环境。"""
    sdk_path = Path(__file__).resolve().parent.parent / "MvCameraControl.dll"
    try:
        return load_mvs_sdk_library(sdk_path)
    except MvsSdkLoadError as exc:
        return _MissingSdkProxy(exc)


def _encode_ascii(value):
    """SDK 节点名通常要求 ASCII 编码。"""
    return value.encode("ascii")


def _encode_path(value):
    """把路径参数转换为当前系统可接受的字节串。"""
    return os.fsencode(str(value))


MvCamCtrldll = _load_sdk_library()

# 用于回调函数传入相机实例
class _MV_PY_OBJECT_(Structure):
    pass
_MV_PY_OBJECT_._fields_ = [
    ('PyObject', py_object),
]
MV_PY_OBJECT = _MV_PY_OBJECT_


class _MV_FRAME_OUT_(Structure):
    pass


_MV_FRAME_OUT_._fields_ = [
    ("pBufAddr", POINTER(c_ubyte)),
    ("stFrameInfo", MV_FRAME_OUT_INFO_EX),
    ("nRes", c_uint * 16),
]
MV_FRAME_OUT = _MV_FRAME_OUT_

class MvCamera():
    """海康相机对象的 Python 封装。

    绝大多数方法都是对 DLL 原生接口的薄包装，命名与官方 SDK 保持一致，
    这样在 Windows 测试机上遇到问题时，能够直接对照海康文档与官方 Demo。
    """

    def __init__(self):
        self._handle = c_void_p()  # 当前连接设备句柄
        self.handle = pointer(self._handle)  # 传给 DLL 的句柄指针

    @staticmethod
    def MV_CC_Initialize():
        MvCamCtrldll.MV_CC_Initialize.restype = c_int
        return MvCamCtrldll.MV_CC_Initialize()

    @staticmethod
    def MV_CC_Finalize():
        MvCamCtrldll.MV_CC_Finalize.restype = c_int
        return MvCamCtrldll.MV_CC_Finalize()

    # 枚举设备
    @staticmethod
    def MV_CC_EnumDevices(nTLayerType, stDevList):
        MvCamCtrldll.MV_CC_EnumDevices.argtype = (c_uint, c_void_p)
        MvCamCtrldll.MV_CC_EnumDevices.restype = c_uint
        # C原型:int MV_CC_EnumDevices(unsigned int nTLayerType, MV_CC_DEVICE_INFO_LIST* pstDevList)
        return MvCamCtrldll.MV_CC_EnumDevices(c_uint(nTLayerType), byref(stDevList))

    @staticmethod
    def MV_CC_SetSDKLogPath(log_path):
        MvCamCtrldll.MV_CC_SetSDKLogPath.argtype = c_char_p
        MvCamCtrldll.MV_CC_SetSDKLogPath.restype = c_uint
        return MvCamCtrldll.MV_CC_SetSDKLogPath(_encode_path(log_path))

    # 创建句柄
    def MV_CC_CreateHandle(self, stDevInfo):
        MvCamCtrldll.MV_CC_DestroyHandle.argtype = c_void_p
        MvCamCtrldll.MV_CC_DestroyHandle.restype = c_uint
        MvCamCtrldll.MV_CC_DestroyHandle(self.handle)

        MvCamCtrldll.MV_CC_CreateHandle.argtype = (c_void_p, c_void_p)
        MvCamCtrldll.MV_CC_CreateHandle.restype = c_uint
        # C原型:int MV_CC_CreateHandle(void ** handle, MV_CC_DEVICE_INFO* pstDevInfo)
        return MvCamCtrldll.MV_CC_CreateHandle(byref(self.handle), byref(stDevInfo))

    # 创建句柄（不生成日志）
    def MV_CC_CreateHandleWithoutLog(self, stDevInfo):
        MvCamCtrldll.MV_CC_DestroyHandle.argtype = c_void_p
        MvCamCtrldll.MV_CC_DestroyHandle.restype = c_uint
        MvCamCtrldll.MV_CC_DestroyHandle(self.handle)

        MvCamCtrldll.MV_CC_CreateHandleWithoutLog.argtype = (c_void_p, c_void_p)
        MvCamCtrldll.MV_CC_CreateHandleWithoutLog.restype = c_uint
        # C原型:int MV_CC_CreateHandleWithoutLog(void ** handle, MV_CC_DEVICE_INFO* pstDevInfo)
        return MvCamCtrldll.MV_CC_CreateHandleWithoutLog(byref(self.handle), byref(stDevInfo))

    # 销毁句柄
    def MV_CC_DestroyHandle(self):
        MvCamCtrldll.MV_CC_DestroyHandle.argtype = c_void_p
        MvCamCtrldll.MV_CC_DestroyHandle.restype = c_uint
        return MvCamCtrldll.MV_CC_DestroyHandle(self.handle)

    # 打开设备
    def MV_CC_OpenDevice(self, nAccessMode=MV_ACCESS_Exclusive, nSwitchoverKey=0):
        MvCamCtrldll.MV_CC_OpenDevice.argtype = (c_void_p, c_uint32, c_uint16)
        MvCamCtrldll.MV_CC_OpenDevice.restype = c_uint
        # C原型:int MV_CC_OpenDevice(void* handle, unsigned int nAccessMode, unsigned short nSwitchoverKey)
        return MvCamCtrldll.MV_CC_OpenDevice(self.handle, nAccessMode, nSwitchoverKey)

    # 关闭设备
    def MV_CC_CloseDevice(self):
        MvCamCtrldll.MV_CC_CloseDevice.argtype = c_void_p
        MvCamCtrldll.MV_CC_CloseDevice.restype = c_uint
        return MvCamCtrldll.MV_CC_CloseDevice(self.handle)

    # 注册取流回调
    def MV_CC_RegisterImageCallBackEx(self, CallBackFun, pUser):
        MvCamCtrldll.MV_CC_RegisterImageCallBackEx.argtype = (c_void_p, c_void_p, c_void_p)
        MvCamCtrldll.MV_CC_RegisterImageCallBackEx.restype = c_uint
        # C原型:int MV_CC_RegisterImageCallBackEx(void* handle, void(* cbOutput)(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser),void* pUser);
        return MvCamCtrldll.MV_CC_RegisterImageCallBackEx(self.handle, CallBackFun, pUser)

    # 开始取流
    def MV_CC_StartGrabbing(self):
        MvCamCtrldll.MV_CC_StartGrabbing.argtype = c_void_p
        MvCamCtrldll.MV_CC_StartGrabbing.restype = c_uint
        return MvCamCtrldll.MV_CC_StartGrabbing(self.handle)

    # 停止取流
    def MV_CC_StopGrabbing(self):
        MvCamCtrldll.MV_CC_StopGrabbing.argtype = c_void_p
        MvCamCtrldll.MV_CC_StopGrabbing.restype = c_uint
        return MvCamCtrldll.MV_CC_StopGrabbing(self.handle)

    # 主动方式取流
    def MV_CC_GetOneFrameTimeout(self, pData, nDataSize, stFrameInfo, nMsec=1000):
        MvCamCtrldll.MV_CC_GetOneFrameTimeout.argtype = (c_void_p, c_void_p, c_uint, c_void_p, c_uint)
        MvCamCtrldll.MV_CC_GetOneFrameTimeout.restype = c_uint
        # C原型:int MV_CC_GetOneFrameTimeout(void* handle, unsigned char * pData , unsigned int nDataSize, MV_FRAME_OUT_INFO_EX* pFrameInfo, unsigned int nMsec)
        return MvCamCtrldll.MV_CC_GetOneFrameTimeout(self.handle, pData, nDataSize, byref(stFrameInfo), nMsec)

    def MV_CC_GetImageBuffer(self, stFrame, nMsec=1000):
        MvCamCtrldll.MV_CC_GetImageBuffer.argtype = (c_void_p, c_void_p, c_uint)
        MvCamCtrldll.MV_CC_GetImageBuffer.restype = c_uint
        return MvCamCtrldll.MV_CC_GetImageBuffer(self.handle, byref(stFrame), nMsec)

    def MV_CC_FreeImageBuffer(self, stFrame):
        MvCamCtrldll.MV_CC_FreeImageBuffer.argtype = (c_void_p, c_void_p)
        MvCamCtrldll.MV_CC_FreeImageBuffer.restype = c_uint
        return MvCamCtrldll.MV_CC_FreeImageBuffer(self.handle, byref(stFrame))

    # 设置SDK内部图像缓存节点个数，范围[1, 30]，在抓图前调用
    def MV_CC_SetImageNodeNum(self, nNum):
        MvCamCtrldll.MV_CC_SetImageNodeNum.argtype = (c_void_p, c_uint)
        MvCamCtrldll.MV_CC_SetImageNodeNum.restype = c_uint
        # C原型:int MV_CC_SetImageNodeNum(void* handle, unsigned int nNum)
        return MvCamCtrldll.MV_CC_SetImageNodeNum(self.handle, nNum)

    # 获取Integer型属性值
    def MV_CC_GetIntValue(self, strKey, stIntValue):
        MvCamCtrldll.MV_CC_GetIntValue.argtype = (c_void_p, c_char_p, c_void_p)
        MvCamCtrldll.MV_CC_GetIntValue.restype = c_uint
        # C原型:int MV_CC_GetIntValue(void* handle,char* strKey,MVCC_INTVALUE *pIntValue)
        return MvCamCtrldll.MV_CC_GetIntValue(self.handle, _encode_ascii(strKey), byref(stIntValue))

    def MV_CC_GetIntValueEx(self, strKey, stIntValue):
        MvCamCtrldll.MV_CC_GetIntValueEx.argtype = (c_void_p, c_char_p, c_void_p)
        MvCamCtrldll.MV_CC_GetIntValueEx.restype = c_uint
        return MvCamCtrldll.MV_CC_GetIntValueEx(self.handle, _encode_ascii(strKey), byref(stIntValue))
    
    # 设置Integer型属性值
    def MV_CC_SetIntValue(self, strKey, nValue):
        MvCamCtrldll.MV_CC_SetIntValue.argtype = (c_void_p, c_char_p, c_uint32)
        MvCamCtrldll.MV_CC_SetIntValue.restype = c_uint
        # C原型:int MV_CC_SetIntValue(void* handle,char* strKey,unsigned int nValue)
        return MvCamCtrldll.MV_CC_SetIntValue(self.handle, _encode_ascii(strKey), c_uint32(nValue))

    def MV_CC_SetIntValueEx(self, strKey, nValue):
        MvCamCtrldll.MV_CC_SetIntValueEx.argtype = (c_void_p, c_char_p, c_uint64)
        MvCamCtrldll.MV_CC_SetIntValueEx.restype = c_uint
        return MvCamCtrldll.MV_CC_SetIntValueEx(self.handle, _encode_ascii(strKey), c_uint64(nValue))

    # 获取Enum型属性值
    def MV_CC_GetEnumValue(self, strKey, stEnumValue):
        MvCamCtrldll.MV_CC_GetEnumValue.argtype = (c_void_p, c_char_p, c_void_p)
        MvCamCtrldll.MV_CC_GetEnumValue.restype = c_uint
        # C原型:int MV_CC_GetEnumValue(void* handle,char* strKey,MVCC_ENUMVALUE *pEnumValue)
        return MvCamCtrldll.MV_CC_GetEnumValue(self.handle, _encode_ascii(strKey), byref(stEnumValue))

    # 设置Enum型属性值
    def MV_CC_SetEnumValue(self, strKey, nValue):
        MvCamCtrldll.MV_CC_SetEnumValue.argtype = (c_void_p, c_char_p, c_uint32)
        MvCamCtrldll.MV_CC_SetEnumValue.restype = c_uint
        # C原型:int MV_CC_SetEnumValue(void* handle,char* strKey,unsigned int nValue)
        return MvCamCtrldll.MV_CC_SetEnumValue(self.handle, _encode_ascii(strKey), c_uint32(nValue))

    # 设置Enum型属性值
    def MV_CC_SetEnumValueByString(self, strKey, sValue):
        MvCamCtrldll.MV_CC_SetEnumValueByString.argtype = (c_void_p, c_char_p, c_char_p)
        MvCamCtrldll.MV_CC_SetEnumValueByString.restype = c_uint
        # C原型:int MV_CC_SetEnumValueByString(void* handle,char* strKey,char* sValue)
        return MvCamCtrldll.MV_CC_SetEnumValueByString(self.handle, _encode_ascii(strKey), _encode_ascii(sValue))

    # 获取Float型属性值
    def MV_CC_GetFloatValue(self, strKey, stFloatValue):
        MvCamCtrldll.MV_CC_GetFloatValue.argtype = (c_void_p, c_char_p, c_void_p)
        MvCamCtrldll.MV_CC_GetFloatValue.restype = c_uint
        # C原型:int MV_CC_GetFloatValue(void* handle,char* strKey,MVCC_FLOATVALUE *pFloatValue)
        return MvCamCtrldll.MV_CC_GetFloatValue(self.handle, _encode_ascii(strKey), byref(stFloatValue))

    # 设置Float型属性值
    def MV_CC_SetFloatValue(self, strKey, fValue):
        MvCamCtrldll.MV_CC_SetFloatValue.argtype = (c_void_p, c_char_p, c_float)
        MvCamCtrldll.MV_CC_SetFloatValue.restype = c_uint
        # C原型:int MV_CC_SetFloatValue(void* handle,char* strKey,float fValue)
        return MvCamCtrldll.MV_CC_SetFloatValue(self.handle, _encode_ascii(strKey), c_float(fValue))

    # 获取Boolean型属性值
    def MV_CC_GetBoolValue(self, strKey, BoolValue):
        MvCamCtrldll.MV_CC_GetBoolValue.argtype = (c_void_p, c_char_p, c_void_p)
        MvCamCtrldll.MV_CC_GetBoolValue.restype = c_uint
        # C原型:int MV_CC_GetBoolValue(void* handle,char* strKey,bool *pBoolValue)
        return MvCamCtrldll.MV_CC_GetBoolValue(self.handle, _encode_ascii(strKey), byref(BoolValue))

    # 设置Boolean型属性值
    def MV_CC_SetBoolValue(self, strKey, bValue):
        MvCamCtrldll.MV_CC_SetBoolValue.argtype = (c_void_p, c_char_p, c_bool)
        MvCamCtrldll.MV_CC_SetBoolValue.restype = c_uint
        # C原型:int MV_CC_SetBoolValue(void* handle,char* strKey,bool bValue)
        return MvCamCtrldll.MV_CC_SetBoolValue(self.handle, _encode_ascii(strKey), bValue)

    # 获取String型属性值
    def MV_CC_GetStringValue(self, strKey, StringValue):
        MvCamCtrldll.MV_CC_GetStringValue.argtype = (c_void_p, c_char_p, c_void_p)
        MvCamCtrldll.MV_CC_GetStringValue.restype = c_uint
        # C原型:int MV_CC_GetStringValue(void* handle,char* strKey,MVCC_STRINGVALUE *pStringValue)
        return MvCamCtrldll.MV_CC_GetStringValue(self.handle, _encode_ascii(strKey), byref(StringValue))
    
    # 设置String型属性值
    def MV_CC_SetStringValue(self, strKey, sValue):
        MvCamCtrldll.MV_CC_SetStringValue.argtype = (c_void_p, c_char_p, c_char_p)
        MvCamCtrldll.MV_CC_SetStringValue.restype = c_uint
        # C原型:int MV_CC_SetStringValue(void* handle,char* strKey,char * sValue)
        return MvCamCtrldll.MV_CC_SetStringValue(self.handle, _encode_ascii(strKey), _encode_path(sValue))
    
    # 设置Command型属性值
    def MV_CC_SetCommandValue(self, strKey):
        MvCamCtrldll.MV_CC_SetCommandValue.argtype = (c_void_p, c_char_p)
        MvCamCtrldll.MV_CC_SetCommandValue.restype = c_uint
        # C原型:int MV_CC_SetCommandValue(void* handle,char* strKey)
        return MvCamCtrldll.MV_CC_SetCommandValue(self.handle, _encode_ascii(strKey))

    def MV_CC_IsDeviceConnected(self):
        MvCamCtrldll.MV_CC_IsDeviceConnected.argtype = c_void_p
        MvCamCtrldll.MV_CC_IsDeviceConnected.restype = c_bool
        return bool(MvCamCtrldll.MV_CC_IsDeviceConnected(self.handle))

    def MV_CC_SetGrabStrategy(self, enGrabStrategy):
        MvCamCtrldll.MV_CC_SetGrabStrategy.argtype = (c_void_p, c_uint)
        MvCamCtrldll.MV_CC_SetGrabStrategy.restype = c_uint
        return MvCamCtrldll.MV_CC_SetGrabStrategy(self.handle, c_uint(enGrabStrategy))

    def MV_CC_SetOutputQueueSize(self, nOutputQueueSize):
        MvCamCtrldll.MV_CC_SetOutputQueueSize.argtype = (c_void_p, c_uint)
        MvCamCtrldll.MV_CC_SetOutputQueueSize.restype = c_uint
        return MvCamCtrldll.MV_CC_SetOutputQueueSize(self.handle, c_uint(nOutputQueueSize))

    # 注册异常消息回调
    def MV_CC_RegisterExceptionCallBack(self, ExceptionCallBackFun, pUser):
        MvCamCtrldll.MV_CC_RegisterExceptionCallBack.argtype = (c_void_p, c_void_p, c_void_p)
        MvCamCtrldll.MV_CC_RegisterExceptionCallBack.restype = c_uint
        # C原型:int MV_CC_RegisterExceptionCallBack(void* handle, void(* cbException)(unsigned int nMsgType, void* pUser),void* pUser)
        return MvCamCtrldll.MV_CC_RegisterExceptionCallBack(self.handle, ExceptionCallBackFun, pUser)

    # 注册事件回调
    def MV_CC_RegisterEventCallBackEx(self, pEventName, EventCallBackFun, pUser):
        MvCamCtrldll.MV_CC_RegisterEventCallBackEx.argtype = (c_void_p, c_char_p, c_void_p, c_void_p)
        MvCamCtrldll.MV_CC_RegisterEventCallBackEx.restype = c_uint
        # C原型:int MV_CC_RegisterEventCallBackEx(void* handle, char* pEventName,void(* cbEvent)(MV_EVENT_OUT_INFO * pEventInfo, void* pUser),void* pUser)
        return MvCamCtrldll.MV_CC_RegisterEventCallBackEx(self.handle, _encode_ascii(pEventName), EventCallBackFun, pUser)

    # 强制IP
    def MV_GIGE_ForceIpEx(self, nIP, nSubNetMask, nDefaultGateWay):
        MvCamCtrldll.MV_GIGE_ForceIpEx.argtype = (c_void_p, c_uint, c_uint, c_uint)
        MvCamCtrldll.MV_GIGE_ForceIpEx.restype = c_uint
        # C原型:int MV_GIGE_ForceIpEx(void* handle, unsigned int nIP, unsigned int nSubNetMask, unsigned int nDefaultGateWay)
        return MvCamCtrldll.MV_GIGE_ForceIpEx(self.handle, c_uint(nIP), c_uint(nSubNetMask), c_uint(nDefaultGateWay))
    
    # 配置IP方式
    def MV_GIGE_SetIpConfig(self, nType):
        MvCamCtrldll.MV_GIGE_SetIpConfig.argtype = (c_void_p, c_uint)
        MvCamCtrldll.MV_GIGE_SetIpConfig.restype = c_uint
        # C原型:int MV_GIGE_SetIpConfig(void* handle, unsigned int nType)
        return MvCamCtrldll.MV_GIGE_SetIpConfig(self.handle, c_uint(nType))

    # 设置传输模式，可以为单播模式、组播模式等
    def MV_GIGE_SetTransmissionType(self, stTransmissionType):
        MvCamCtrldll.MV_GIGE_SetTransmissionType.argtype = (c_void_p, c_void_p)
        MvCamCtrldll.MV_GIGE_SetTransmissionType.restype = c_uint
        # C原型:int MV_GIGE_SetTransmissionType(void* handle, MV_TRANSMISSION_TYPE * pstTransmissionType)
        return MvCamCtrldll.MV_GIGE_SetTransmissionType(self.handle, byref(stTransmissionType))

    # 保存一张图片
    def MV_CC_SaveImageEx2(self, stSaveParam):
        MvCamCtrldll.MV_CC_SaveImageEx2.argtype = (c_void_p, c_void_p)
        MvCamCtrldll.MV_CC_SaveImageEx2.restype = c_uint
        # C原型:int MV_CC_SaveImageEx2(void* handle, MV_SAVE_IMAGE_PARAM_EX* pSaveParam)
        return MvCamCtrldll.MV_CC_SaveImageEx2(self.handle, byref(stSaveParam))

    # 像素格式转换
    def MV_CC_ConvertPixelType(self, stConvertParam):
        MvCamCtrldll.MV_CC_ConvertPixelType.argtype = (c_void_p, c_void_p)
        MvCamCtrldll.MV_CC_ConvertPixelType.restype = c_uint
        # C原型:int MV_CC_ConvertPixelType(void* handle, MV_CC_PIXEL_CONVERT_PARAM* pstCvtParam)
        return MvCamCtrldll.MV_CC_ConvertPixelType(self.handle, byref(stConvertParam))

    # 保存属性节点
    def MV_CC_FeatureSave(self, pFileName):
        MvCamCtrldll.MV_CC_FeatureSave.argtype = (c_void_p, c_char_p)
        MvCamCtrldll.MV_CC_FeatureSave.restype = c_uint
        # C原型:int MV_CC_FeatureSave(void* handle, char* pFileName)
        return MvCamCtrldll.MV_CC_FeatureSave(self.handle, _encode_path(pFileName))
    
    # 加载属性节点
    def MV_CC_FeatureLoad(self, pFileName):
        MvCamCtrldll.MV_CC_FeatureLoad.argtype = (c_void_p, c_char_p)
        MvCamCtrldll.MV_CC_FeatureLoad.restype = c_uint
        # C原型:int MV_CC_FeatureLoad(void* handle, char* pFileName)
        return MvCamCtrldll.MV_CC_FeatureLoad(self.handle, _encode_path(pFileName))

    # fileaccess read
    def MV_CC_FileAccessRead(self, stFileAccess):
        MvCamCtrldll.MV_CC_FileAccessRead.argtype = (c_void_p, c_void_p)
        MvCamCtrldll.MV_CC_FileAccessRead.restype = c_uint
        # C原型:int MV_CC_FileAccessRead(void* handle, MV_CC_FILE_ACCESS * pstFileAccess)
        return MvCamCtrldll.MV_CC_FileAccessRead(self.handle, byref(stFileAccess))

    # fileaccess write
    def MV_CC_FileAccessWrite(self, stFileAccess):
        MvCamCtrldll.MV_CC_FileAccessWrite.argtype = (c_void_p, c_void_p)
        MvCamCtrldll.MV_CC_FileAccessWrite.restype = c_uint
        # C原型:int MV_CC_FileAccessWrite(void* handle, MV_CC_FILE_ACCESS * pstFileAccess)
        return MvCamCtrldll.MV_CC_FileAccessWrite(self.handle, byref(stFileAccess))

    # 获取文件存取进度
    def MV_CC_GetFileAccessProgress(self, stFileAccessProgress):
        MvCamCtrldll.MV_CC_GetFileAccessProgress.argtype = (c_void_p, c_void_p)
        MvCamCtrldll.MV_CC_GetFileAccessProgress.restype = c_uint
        # C原型:int MV_CC_GetFileAccessProgress(void* handle, MV_CC_FILE_ACCESS_PROGRESS * pstFileAccessProgress)
        return MvCamCtrldll.MV_CC_GetFileAccessProgress(self.handle, byref(stFileAccessProgress))

    # 获取网络最佳包大小
    def MV_CC_GetOptimalPacketSize(self):
        MvCamCtrldll.MV_CC_GetOptimalPacketSize.argtype = (c_void_p)
        MvCamCtrldll.MV_CC_GetOptimalPacketSize.restype = c_uint
        # C原型:int __stdcall MV_CC_GetOptimalPacketSize(void* handle);
        return MvCamCtrldll.MV_CC_GetOptimalPacketSize(self.handle)

    
