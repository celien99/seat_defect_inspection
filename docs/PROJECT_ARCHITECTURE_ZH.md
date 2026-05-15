# Seat Defect Inspection 架构说明

当前项目的主检测架构只保留一层：`seat_defect_core`。

- `seat_defect_core`：唯一 inspect runtime 真源，包含完整主检测流程。
- `seat_defect_inspection`：工程工具层，负责 CLI、采图、训练、离线批测等辅助能力。
- `media_inputs` / `mvsCamera`：工程工具层依赖，不属于 core 主流程。

`capture`、`train-yolo`、`train-patchcore`、`inspect` CLI、`inspect-folder`、MVS 相机接入等可以继续存在于仓库中，但它们不能再承载检测判定逻辑。所有检测行为必须回到 `seat_defect_core`。

## 1. 总体结构

```text
external system
  -> seat_defect_core.api
     -> seat_defect_core.service.frames
     -> seat_defect_core.service.inspection
     -> seat_defect_core.service.response
     -> seat_defect_core.service.core
     -> seat_defect_core.service.inspection_camera
     -> yolo / roi / regions / patchcore / fusion / reporting

engineering tools
  -> seat_defect_inspection
     -> capture / offline discovery / training / CLI
     -> seat_defect_core.service.inspection
```

主流程只接受外部传入的图片，不负责打开相机、不负责采集、不负责训练、不负责遍历文件夹。

## 2. 目录边界

```text
src/
├── seat_defect_core/
│   ├── config.py                    # inspect runtime 配置模型
│   ├── api.py                       # 对外公开 inspect API
│   ├── types/
│   │   ├── __init__.py              # 类型聚合导出
│   │   ├── geometry.py              # 几何值对象
│   │   ├── input.py                 # 外部输入与内部帧
│   │   ├── pipeline.py              # pipeline 中间结果
│   │   └── results.py               # 检测结果
│   ├── runtime_config.py            # inspect 配置加载入口
│   ├── runtime_config_parsers.py    # inspect 配置解析与校验
│   ├── yolo/                        # YOLO 检测/分割推理
│   ├── cvops/                       # ROI、mask、regions、质量门控、调试图
│   ├── patchcore/                   # PatchCore 与颜色分支 runtime
│   ├── service/
│   │   ├── core.py                  # runtime 上下文、模型缓存、pipeline 缓存
│   │   ├── frames.py                # 外部 frame 标准化与 FramePacket 构造
│   │   ├── inspection.py            # 唯一 inspect 编排主流程
│   │   ├── response.py              # REJECT 结果、报告导出、响应封装
│   │   └── inspection_camera.py     # 单机位检测细节
│   ├── fusion.py                    # 多机位结果融合
│   ├── reporting.py                 # 检测报告输出
│   └── util.py
│
├── seat_defect_inspection/          # 工程工具层
├── media_inputs/                    # 工具层输入抽象
└── mvsCamera/                       # 工具层 MVS 适配
```

## 3. 主流程入口

公开入口：

```python
from seat_defect_core import InspectionFrame, SeatDefectInspector

inspector = SeatDefectInspector("configs/seat_defect_inspection.mvs.json")
response = inspector.inspect(
    frames=[
        InspectionFrame(camera_id="cam_0", image=cam_0_image),
        InspectionFrame(camera_id="cam_1", image=cam_1_image),
    ],
    part_id="seat_000001",
)
```

一次性调用：

```python
from seat_defect_core import inspect_once

response = inspect_once(
    "configs/seat_defect_inspection.mvs.json",
    frames=[
        {"camera_id": "cam_0", "image": cam_0_image},
        {"camera_id": "cam_1", "image": cam_1_image},
    ],
)
```

内部入口：

```python
from seat_defect_core.service.core import InspectionService
from seat_defect_core.service.inspection import inspect_frames

service = InspectionService(config)
result = inspect_frames(service, frames)
```

`seat_defect_core.api` 是公开 API；`service.*` 是内部工程化分层。`inspect_frames` 是唯一主检测编排入口，接收 `InspectionFrame` 列表并返回 `InspectionResult`。`SeatDefectInspector.inspect()` 会进一步封装成 `InspectionResponse`，包含报告路径和 artifact 路径。

## 4. inspect 主流程

```text
SeatDefectInspector.inspect
  -> normalize_inspection_frames
  -> inspect_frames
     -> resolve_context(seat_model_id)
     -> 校验 camera_id
     -> 为每个启用机位构造 FramePacket
     -> inspect_one_camera
        -> YOLO detection
        -> ROI refine + mask
        -> quality guard
        -> full-seat PatchCore 或 region PatchCore
        -> color branch（非 region 模式）
        -> debug artifacts
     -> early-stop 判断
     -> fuse_camera_results
     -> export_result
  -> build_inspection_response
```

工程层 `seat_defect_inspection inspect` 命令的流程变为：

```text
run_inspection
  -> acquisition.capture
  -> 转换成 InspectionFrame
  -> seat_defect_core.service.inspection.inspect_frames
```

因此融合、提前终止、缺帧、单机位异常兜底、报告输出都只有 core 一处实现。

## 5. 配置边界

core 的配置只保留 inspect 需要的字段：

- `part_id`
- `default_seat_model_id`
- `output_json_path`
- `debug_dir`
- `fusion`
- `cameras` / `seat_models`
- 单机位的 `camera_id`
- YOLO 检测参数
- ROI 参数
- PatchCore 参数
- color branch 参数
- regions 参数

以下字段不属于 core 主流程配置：

- `capture_dir`
- `capture_retries`
- `train_good_dir`
- `yolo_training`
- `ignore_classes`
- `save_debug_artifacts`
- `debug_artifact_mode`

如果这些字段出现在 core 配置里，应视为配置污染，而不是被静默忽略。

## 6. 数据结构

数据结构按用途拆分：

- `types/geometry.py`：`BoundingBox`
- `types/input.py`：`InspectionFrame`、`FramePacket`
- `types/pipeline.py`：图像质量、YOLO 检测、ROI 结果
- `types/results.py`：PatchCore/颜色/region/相机/整件检测结果

core 内部统一从 `seat_defect_core.types` 引用类型，不保留 `schemas.py` 兼容聚合入口。

核心输入：

```python
InspectionFrame(
    camera_id="cam_0",
    image=image,
    source="optional/path/or/url",
    frame_id="optional_frame_id",
    timestamp="optional_iso_timestamp",
    source_kind="external_image",
)
```

核心输出：

- `InspectionResult`：多机位融合后的纯检测结果。
- `InspectionResponse`：公开 API 响应，包含 `InspectionResult`、最新报告路径、历史报告路径、artifact 路径。

## 7. 单机位检测责任

`src/seat_defect_core/service/inspection_camera.py` 是单机位检测细节入口。

单机位职责：

1. 调用 `CameraPipeline.prepare_image()` 完成 YOLO、ROI、质量门控。
2. 如果未启用 regions，加载机位级 PatchCore 模型并检测完整 ROI。
3. 如果启用 regions，把标准 ROI 切为多个区域，每个区域加载自己的 PatchCore 模型。
4. 汇总单机位状态：`OK` / `NG` / `REJECT`。
5. 写出调试 artifacts。

## 8. regions 模式

regions 是 inspect runtime 的一部分，配置在单个 camera 下：

```json
{
  "region_id": "upper",
  "box": [0.05, 0.05, 0.95, 0.40],
  "patchcore_model_path": "../models/seat_defect_inspection/cam_0_upper_patchcore.npz"
}
```

规则：

- `box` 是标准 ROI 坐标系下的归一化矩形 `[x1, y1, x2, y2]`。
- 任一 enabled region 存在时，该机位进入 region PatchCore 模式。
- region 模式不使用 camera 级 `patchcore_model_path` 做纹理检测。
- 每个 region 需要自己的 PatchCore 模型。

## 9. 维护规则

1. 改 inspect 行为，只改 `seat_defect_core`。
2. 对外调用能力只从 `seat_defect_core` 暴露。
3. `seat_defect_core.api` 是公开 API，`service.*` 是内部实现分层。
4. `seat_defect_core` 包级入口保持轻量，重依赖 inspect runtime 通过懒加载进入。
5. `seat_defect_core` 不引入采图、训练、CLI、文件夹遍历职责。
6. `seat_defect_inspection` 可以做采图、训练和批测，但检测判定必须调用 core 主流程。
7. core 配置解析器不静默吞掉工程工具字段。
8. 新增主流程字段时，先更新 `seat_defect_core.config`、`runtime_config_parsers` 和 `types`。
