# Seat Defect Inspection 独立说明

当前主检测架构只保留 `seat_defect_core`。`seat_defect_core` 是唯一 inspect runtime 真源；CLI、采图、训练、离线批测属于工程工具层，只负责把图片交给 core 主流程。

更细的图像链路说明见 [IMAGE_PIPELINE_DETAILS_ZH.md](./IMAGE_PIPELINE_DETAILS_ZH.md)，架构边界见 [PROJECT_ARCHITECTURE_ZH.md](./PROJECT_ARCHITECTURE_ZH.md)。

## 快速开始

```bash
cd seat_defect_inspection
conda create -n seat-defect-inspection python=3.10 -y
conda activate seat-defect-inspection
pip install -e .
seat-defect-inspection --help
```

也可以直接用模块方式运行：

```bash
python -m seat_defect_inspection --help
```

## 工程工具命令

以下命令属于工程工具层，不是主检测架构入口。外部系统和产线集成应直接调用 `seat_defect_core`。

```bash
seat-defect-inspection capture --config configs/seat_defect_inspection.mvs.json --part-id seat_000001
seat-defect-inspection train-patchcore --config configs/seat_defect_inspection.mvs.json
seat-defect-inspection train-yolo --config configs/seat_defect_inspection.mvs.json
seat-defect-inspection inspect --config configs/seat_defect_inspection.mvs.json --part-id seat_000001
seat-defect-inspection inspect-folder --config configs/seat_defect_inspection.mvs.json --input-dir offline_samples
```

采图结果也可以直接写入各机位 `train_good_dir`：

```bash
seat-defect-inspection capture \
  --config configs/seat_defect_inspection.mvs.json \
  --part-id seat_000001 \
  --save-to-train-good-dir
```

## Python Runtime 调用

主流程包名是 `seat_defect_core`。core 不负责采图，调用方需要自己拿到图片，再按 `camera_id + image` 传入。

```python
import cv2
from seat_defect_core import InspectionFrame, SeatDefectInspector

inspector = SeatDefectInspector("configs/seat_defect_inspection.mvs.json")
response = inspector.inspect(
    frames=[
        InspectionFrame(camera_id="cam_0", image=cv2.imread("cam_0.png")),
        InspectionFrame(camera_id="cam_1", image=cv2.imread("cam_1.png")),
    ],
    part_id="seat_000001",
)

print(response.status, response.decision_reason)
print(response.report_path)
```

离线图片文件夹批测仍从工程包调用：

```python
from seat_defect_inspection import inspect_image_folder, load_config

config = load_config("configs/seat_defect_inspection.mvs.json")
summary = inspect_image_folder(
    config,
    input_dir="offline_samples",
    output_dir="outputs/offline_check",
)
print(summary["sample_count"], summary["ok_count"], summary["ng_count"])
```

## 工程工具参考工作流

1. 用 `capture` 采集正常样本。
2. 正常样本进入各机位 `train_good_dir`。
3. 执行 `train-patchcore`。
4. 准备 YOLO segmentation 数据集并执行 `train-yolo`。
5. 配好每个机位的 `patchcore_model_path` 和 YOLO `model_path` 后，线上跑 `inspect`，线下批测跑 `inspect-folder`。

`train_good_dir` 保存的是相机原图。训练 PatchCore 时仍会复用正式链路，先走 YOLO、ROI 和 mask 构造，再拟合 PatchCore。

如果修改了 YOLO 检测参数、ROI/mask、PatchCore 输入模式或 full 后端配置，必须重新执行 `train-patchcore`。模型包会保存 `pipeline_signature`，线上加载时会校验签名。

当前运行配置只允许完整版本 PatchCore：

- `patchcore.backend = full`
- 默认 backbone 为 `wide_resnet50_2`
- 默认特征层为 `layer2 / layer3`
- 需要可用的 `torch / torchvision`
- 如 `backbone_pretrained = false`，必须配置本地 `backbone_weights_path`

PatchCore 参数排查见 [PATCHCORE_TUNING_GUIDE_ZH.md](./PATCHCORE_TUNING_GUIDE_ZH.md)。

现场运行必须配置可用的 YOLO segmentation 权重；未检测到目标或缺少分割 mask 时，本次检测会返回 `REJECT`。

## 离线批测目录

单样本目录：

```text
offline_samples/
├── cam_0.jpg
└── cam_1.jpg
```

按样本分目录：

```text
offline_samples/
├── sample_001/
│   ├── cam_0.jpg
│   └── cam_1.jpg
└── sample_002/
    ├── cam_0.jpg
    └── cam_1.jpg
```

按机位分目录：

```text
offline_samples/
├── cam_0/
│   ├── sample_001.jpg
│   └── sample_002.jpg
└── cam_1/
    ├── sample_001.jpg
    └── sample_002.jpg
```

## 目录约定

- `data/seat_defect_inspection/<camera_id>/train/good`: PatchCore 正常样本目录
- `models/seat_defect_inspection/<camera_id>_patchcore.npz`: 每个机位的 PatchCore 模型
- `models/seat_defect_inspection/<camera_id>_patchcore.summary.json`: PatchCore 训练摘要
- `outputs/seat_defect_inspection/capture`: 采图输出
- `outputs/seat_defect_inspection/debug`: 检测调试图输出
- `outputs/seat_defect_inspection/yolo_training`: YOLO 训练输出

## 当前代码结构

```text
src/
├── seat_defect_core/        # inspect runtime 真源
├── seat_defect_inspection/  # 工程工具层：CLI、采图、训练、离线批测
├── media_inputs/            # 工具层输入抽象
└── mvsCamera/               # 工具层 MVS 适配
```

主检测行为只在 `seat_defect_core` 维护：外部帧标准化、YOLO 推理、ROI/mask、regions、PatchCore、颜色分支、融合、调试图和检测报告。

工程工具行为在 `seat_defect_inspection` 维护：CLI、配置扩展、采图、manifest、离线目录发现、PatchCore 训练编排、YOLO 训练和 LabelMe 转换。它的 inspect 命令只负责采图，然后调用 `seat_defect_core` 主流程。

不再保留旧 runtime 兼容导入路径。YOLO 推理、ROI、PatchCore、融合、调试产物等能力请直接从 `seat_defect_core` 导入。
