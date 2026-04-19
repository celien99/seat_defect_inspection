# Seat Defect Inspection 独立说明

`seat_defect_inspection` 是仓库内的独立子项目，已经和根项目 `seat_inspection` 解耦。

它面向汽车座椅缺陷检测，当前包含三条核心能力：

- 多机位采图
- 每机位独立 PatchCore 训练与推理
- YOLO 座椅定位模型训练

当前代码已经接入 `src/mvsCamera/` SDK，并支持 `mvs://...` 机位源。真机联调没有在这次改造里实测，后续可直接拿现场设备继续验证。

## 快速开始

```bash
cd seat_defect_inspection
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
seat-defect-inspection --help
```

也可以直接使用模块方式运行：

```bash
python -m seat_defect_inspection --help
```

## 主要命令

采集全部启用机位的一帧：

```bash
seat-defect-inspection capture \
  --config configs/seat_defect_inspection.mvs.json \
  --part-id seat_000001
```

把采集结果同时落到各机位 `train_good_dir`：

```bash
seat-defect-inspection capture \
  --config configs/seat_defect_inspection.mvs.json \
  --part-id seat_000001 \
  --save-to-train-good-dir
```

训练每个机位的 PatchCore：

```bash
seat-defect-inspection train-patchcore \
  --config configs/seat_defect_inspection.mvs.json
```

训练 YOLO：

```bash
seat-defect-inspection train-yolo \
  --config configs/seat_defect_inspection.mvs.json
```

执行一次完整检测：

```bash
seat-defect-inspection inspect \
  --config configs/seat_defect_inspection.mvs.json \
  --part-id seat_000001
```

## 推荐工作流

1. 先用 `capture` 采集正常样本。
2. 正常样本进入各机位 `train_good_dir`，再执行 `train-patchcore`。
3. 单独准备 YOLO 数据集并执行 `train-yolo`。
4. 把每个机位的 `patchcore_model_path` 和 YOLO `model_path` 配好后，执行 `inspect`。

如果现场还没有 YOLO 权重，可以先把 `detection.model_path` 设为 `null`，继续使用 `fallback_box` 走完整流程。

## 目录约定

- `data/seat_defect_inspection/<camera_id>/train/good`
  PatchCore 正常样本目录
- `models/seat_defect_inspection/<camera_id>_patchcore.npz`
  每个机位独立 PatchCore 模型
- `outputs/seat_defect_inspection/capture`
  `capture` 命令采图输出
- `outputs/seat_defect_inspection/debug`
  `inspect` 过程中的调试图输出
- `outputs/seat_defect_inspection/yolo_training`
  YOLO 训练输出

## 文件结构与模块职责

```text
seat_defect_inspection/
├── README.md
├── README_ZH.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── seat_defect_inspection.mvs.json
│   └── seat_defect_yolo.dataset.example.yaml
└── src/
    ├── media_inputs/
    │   ├── __init__.py
    │   └── core.py
    ├── mvsCamera/
    │   ├── __init__.py
    │   ├── frame_source.py
    │   ├── camera_controller.py
    │   ├── pixel_utils.py
    │   ├── MvCameraControl.dll
    │   └── sdk/
    ├── seat_defect_inspection/
    │   ├── __init__.py
    │   ├── __main__.py
    │   ├── cli.py
    │   ├── service.py
    │   ├── acquisition.py
    │   ├── quality.py
    │   ├── preprocess.py
    │   ├── detection.py
    │   ├── roi.py
    │   ├── patchcore.py
    │   ├── color_branch.py
    │   ├── fusion.py
    │   ├── reporting.py
    │   ├── runtime_config.py
    │   ├── config.py
    │   ├── schemas.py
    │   └── yolo_training.py
    └── seat_defect_inspection.egg-info/
```

- `README.md` / `README_ZH.md`
  项目英文/中文说明、命令示例、配置约定和落地边界。
- `pyproject.toml`
  打包配置、最小依赖定义，以及 `seat-defect-inspection` CLI 入口声明。
- `requirements.txt`
  轻量依赖清单，便于快速安装运行环境。
- `configs/seat_defect_inspection.mvs.json`
  当前 5 路 MVS 相机示例配置，定义机位源、PatchCore 路径、YOLO 配置和输出目录。
- `configs/seat_defect_yolo.dataset.example.yaml`
  YOLO 训练数据集 YAML 示例，定义 train/val/test 路径和类别名。
- `src/seat_defect_inspection/__main__.py`
  模块启动入口，支持 `python -m seat_defect_inspection`。
- `src/seat_defect_inspection/cli.py`
  命令行分发层，统一收口 `capture`、`inspect`、`train-patchcore`、`train-yolo`。
- `src/seat_defect_inspection/service.py`
  项目主编排层，负责把采图、质量判断、预处理、检测、ROI、PatchCore、颜色分支和多机位融合串成完整业务流程。
- `src/seat_defect_inspection/acquisition.py`
  单机位采图服务，把图片、视频、普通摄像头和 MVS 工业相机统一包装成 `FramePacket`。
- `src/seat_defect_inspection/quality.py`
  图像质量守卫，负责模糊、过暗、过曝、黑白帧过滤。
- `src/seat_defect_inspection/preprocess.py`
  OpenCV 预处理层，负责去噪、畸变矫正、亮度归一化、CLAHE 和锐化。
- `src/seat_defect_inspection/detection.py`
  YOLO 检测层，负责主座椅目标和忽略区域检测；无权重时退回到静态 `fallback_box`。
- `src/seat_defect_inspection/roi.py`
  ROI 精修层，负责扩框裁剪、GrabCut/分割掩膜、忽略区掩膜、对齐和有效区域掩膜生成。
- `src/seat_defect_inspection/patchcore.py`
  轻量 PatchCore 风格纹理异常检测实现，包含训练、记忆库压缩、模型保存/加载和热力图生成。
- `src/seat_defect_inspection/color_branch.py`
  颜色一致性分支，基于 LAB 统计量做正常颜色分布建模和异常评分。
- `src/seat_defect_inspection/fusion.py`
  多机位融合策略层，把各机位 `OK/NG/REJECT` 结果汇总成最终判定。
- `src/seat_defect_inspection/reporting.py`
  结果输出层，负责写出采图 manifest 和最终检测 JSON 报告。
- `src/seat_defect_inspection/runtime_config.py`
  JSON 配置加载层，负责把配置文件解析为程序内部 dataclass，并解析本地路径。
- `src/seat_defect_inspection/config.py`
  配置数据结构定义，描述质量、预处理、检测、ROI、PatchCore、颜色分支和 YOLO 训练参数。
- `src/seat_defect_inspection/schemas.py`
  流程中的核心数据结构定义，例如 `BoundingBox`、`DetectionResult`、`InspectionResult`。
- `src/seat_defect_inspection/yolo_training.py`
  YOLO 训练封装层，调用 Ultralytics 完成训练并落盘训练摘要。
- `src/seat_defect_inspection/__init__.py`
  对外导出的公共 API，便于外部脚本直接复用项目能力。
- `src/media_inputs/core.py`
  通用媒体输入中间层，统一图片、视频、普通摄像头和 MVS 工业相机的读取接口。
- `src/mvsCamera/frame_source.py`
  `mvs://` 源解析与 OpenCV 风格采图适配层，把海康相机封装成统一取流对象。
- `src/mvsCamera/camera_controller.py`
  海康 MVS 核心控制器，负责 SDK 初始化、设备枚举、按 SN/IP/MAC 选机、设置参数和取流。
- `src/mvsCamera/pixel_utils.py`
  工业相机像素格式与底层缓冲区辅助函数。
- `src/mvsCamera/sdk/`
  海康 MVS Python ctypes 封装和头文件映射，属于底层 SDK 绑定层。
- `src/mvsCamera/MvCameraControl.dll`
  海康 MVS 控制库，当前仓库放的是 Windows DLL。
- `src/seat_defect_inspection.egg-info/`
  安装或打包时生成的元数据目录，不属于核心业务逻辑。

## 配置文件

- `configs/seat_defect_inspection.mvs.json`
  当前 5 路 MVS 相机配置示例
- `configs/seat_defect_yolo.dataset.example.yaml`
  YOLO 数据集配置示例

## 配置重点

每个机位至少要关注这些字段：

- `source`
  输入源。可以是 `mvs://...`、本地图片、视频或普通摄像头编号
- `train_good_dir`
  PatchCore 正常样本目录
- `patchcore_model_path`
  当前机位 PatchCore 模型输出路径
- `detection.model_path`
  YOLO 权重路径。没有时可先设为 `null`
- `detection.fallback_box`
  YOLO 不可用时的兜底框

顶层最常用字段：

- `capture_dir`
  采图输出目录
- `output_json_path`
  最终检测结果 JSON
- `debug_dir`
  检测调试图目录

## 当前实现边界

- 已完成独立项目化、命令收口、配置整理、PatchCore/YOLO/采图闭环
- 已接通 `src/mvsCamera/` SDK 代码路径
- 未做真机 MVS 现场联调，本次不包含相机硬件实测
