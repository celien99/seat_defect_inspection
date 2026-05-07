# Seat Defect Inspection 架构说明

当前仓库采用“唯一 runtime + 工程工具层 + SDK 门面”的结构。`seat_defect_core` 是检测行为唯一真源；`seat_defect_inspection` 不再提供任何 runtime 兼容转发路径。

## 1. 架构边界

```text
seat_defect_core
  唯一检测 runtime 真源
  - config / schemas
  - preprocess
  - yolo detection
  - cvops / ROI / debug artifacts
  - patchcore / color branch
  - fusion / reporting
  - service core / single-camera inspection

seat_defect_inspection
  工程与现场工具层
  - CLI
  - runtime_config 解析工程字段
  - acquisition / media_inputs / mvsCamera
  - capture
  - inspect / inspect-folder 编排
  - train-patchcore
  - train-yolo / labelme conversion

seat_defect_sdk
  外部图片输入 SDK 门面
  - CameraFrame
  - SeatDefectInspector
  - inspect_once
```

`pyproject.toml` 发布包名为 `seat-defect-sdk`，只打包 `seat_defect_sdk` 和 `seat_defect_core`。CLI、采图、MVS 相机、YOLO 训练等工程能力不进入 SDK wheel。

## 2. 当前顶层结构

```text
src/
├── seat_defect_core/
│   ├── config.py
│   ├── schemas.py
│   ├── runtime_config.py
│   ├── runtime_config_parsers.py
│   ├── preprocess/
│   ├── cvops/
│   ├── yolo/
│   ├── patchcore/
│   ├── service/
│   ├── fusion.py
│   ├── reporting.py
│   └── util.py
├── seat_defect_sdk/
│   └── client.py
├── seat_defect_inspection/
│   ├── cli.py
│   ├── cli_commands/
│   ├── config.py
│   ├── schemas.py
│   ├── runtime_config.py
│   ├── runtime_config_parsers.py
│   ├── acquisition.py
│   ├── reporting.py
│   ├── service/
│   └── yolo/
│       ├── training.py
│       ├── dataset_validation.py
│       └── labelme_to_yolo.py
├── media_inputs/
└── mvsCamera/
```

## 3. Runtime 真源

检测链路只允许在 `seat_defect_core` 改：

| 能力 | 真源文件 |
| --- | --- |
| 配置模型 | `src/seat_defect_core/config.py` |
| 流程数据结构 | `src/seat_defect_core/schemas.py` |
| 配置解析 | `src/seat_defect_core/runtime_config.py`、`runtime_config_parsers.py` |
| 图像预处理 | `src/seat_defect_core/preprocess/engine.py` |
| 质量门控 | `src/seat_defect_core/cvops/quality.py` |
| ROI 与 mask 构造 | `src/seat_defect_core/cvops/roi.py`、`roi_geometry.py` |
| 调试图 | `src/seat_defect_core/cvops/debug_artifacts.py` |
| YOLO 推理 | `src/seat_defect_core/yolo/detection.py` |
| PatchCore | `src/seat_defect_core/patchcore/engine.py`、`features.py`、`scoring.py` |
| 颜色分支 | `src/seat_defect_core/patchcore/color_branch.py` |
| 单机位检测 | `src/seat_defect_core/service/inspection_camera.py` |
| runtime 缓存与模型加载 | `src/seat_defect_core/service/core.py` |
| 多机位融合 | `src/seat_defect_core/fusion.py` |
| 检测报告 | `src/seat_defect_core/reporting.py` |

`seat_defect_core.patchcore` 同时提供训练所需的 `fit/save/list_images`，因此 `train-patchcore` 也不会依赖工程层副本。

## 4. 工程层职责

`seat_defect_inspection` 只负责把 runtime 组织成现场工作流。

| 文件/目录 | 职责 |
| --- | --- |
| `cli.py`、`cli_commands/` | 命令入口和参数路由 |
| `config.py` | 继承 core 顶层配置，只扩展 `train_good_dir`、`capture_dir`、`capture_retries`、`YoloTrainingConfig` |
| `schemas.py` | 只定义 `CaptureRecord`、`CaptureSummary` |
| `runtime_config.py`、`runtime_config_parsers.py` | 解析工程配置和 YOLO 训练块 |
| `acquisition.py` | 把图片、视频、普通相机、MVS 相机统一成 `seat_defect_core.schemas.FramePacket` |
| `reporting.py` | 检测报告直接复用 core，只保留采图 manifest |
| `service/core.py` | 继承 `seat_defect_core.service.core.InspectionService`，只补充 `AcquisitionService` |
| `service/inspection.py` | 在线多机位采图、检测编排和 fail-fast |
| `service/capture.py` | 多机位采图与 manifest |
| `service/offline_inspection.py` | 离线目录样本发现，临时替换 camera source 后复用在线检测链 |
| `service/training.py` | PatchCore 训练编排，调用 core pipeline 与 core PatchCore |
| `yolo/training.py` | YOLO segmentation 训练 |
| `yolo/dataset_validation.py` | YOLO 数据集预检 |
| `yolo/labelme_to_yolo.py` | LabelMe 到 YOLO segmentation 转换 |

## 5. 主流程

### `inspect`

```text
seat_defect_inspection.cli
  -> cli_commands/inspect.py
  -> runtime_config.load_config
  -> service.run_inspection
  -> service/inspection.py
  -> seat_defect_core.service.inspection_camera.inspect_one_camera
  -> seat_defect_core fusion/reporting
```

在线检测仍由工程层负责采图；每个机位的预处理、YOLO、ROI、PatchCore、颜色分支和调试图全部走 core。

### `inspect-folder`

```text
service/offline_inspection.py
  -> 解析单样本/按样本分目录/按机位分目录
  -> 临时把图片路径写入 camera.source
  -> service/inspection.py:run_inspection
```

离线批测不维护独立检测逻辑，只切换输入源并复用在线主链。

### `train-patchcore`

```text
service/training.py
  -> core CameraPipeline.prepare_image
  -> core PatchCoreService.fit
  -> core ColorConsistencyService.fit
  -> core PatchCoreService.save
```

训练和推理共用同一套上游图像链路。模型包保存 `pipeline_signature`，线上加载时会校验签名，避免旧模型静默复用。

### `train-yolo`

YOLO 训练仍在工程层，因为它不是 SDK runtime 的一部分：

```text
cli_commands/train_yolo.py
  -> runtime_config.load_yolo_training_config
  -> yolo/training.py
  -> yolo/dataset_validation.py
  -> ultralytics.YOLO.train
```

## 6. Import 规则

不保留兼容导入。任何 runtime 能力都必须直接从 `seat_defect_core` 导入：

```python
from seat_defect_core.preprocess import PreprocessEngine
from seat_defect_core.cvops import RoiRefineEngine
from seat_defect_core.patchcore import PatchCoreService
from seat_defect_core.yolo import DetectionService
from seat_defect_core.fusion import fuse_camera_results
```

工程入口从 `seat_defect_inspection` 导入：

```python
from seat_defect_inspection import run_inspection, inspect_image_folder
from seat_defect_inspection.runtime_config import load_config
```

外部系统 SDK 从 `seat_defect_sdk` 导入：

```python
from seat_defect_sdk import CameraFrame, SeatDefectInspector
```

## 7. 维护规则

1. 改检测结果行为时，只改 `seat_defect_core`。
2. `seat_defect_inspection` 不新增预处理、YOLO 推理、ROI、PatchCore、融合、调试产物或检测报告副本。
3. `seat_defect_inspection` 不新增 runtime re-export 模块；旧 import 失效是预期行为。
4. 工程层新增能力时，应围绕输入、输出、训练、采图、CLI 编排展开。
5. 如果新增 runtime 配置字段，先加到 `seat_defect_core.config` 和 core 解析器；工程层只在需要 CLI/训练扩展字段时继承补充。
