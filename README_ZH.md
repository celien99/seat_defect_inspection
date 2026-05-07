# Seat Defect Inspection 独立说明

`seat_defect_inspection` 是汽车座椅缺陷检测的工程 CLI 项目。当前检测 runtime 已统一到 `seat_defect_core`，外部图片输入 SDK 门面是 `seat_defect_sdk`。

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

## 主要命令

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

## Python SDK 调用

SDK 包名是 `seat_defect_sdk`。SDK 不负责采图，调用方需要自己拿到图片，再按 `camera_id + image` 传入。

```python
import cv2
from seat_defect_sdk import CameraFrame, SeatDefectInspector

inspector = SeatDefectInspector("configs/seat_defect_inspection.mvs.json")
response = inspector.inspect(
    frames=[
        CameraFrame(camera_id="cam_0", image=cv2.imread("cam_0.png")),
        CameraFrame(camera_id="cam_1", image=cv2.imread("cam_1.png")),
    ],
    part_id="seat_000001",
)

print(response.status, response.decision_reason)
print(response.report_path)
print(response.archive_report_path)
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

## 推荐工作流

1. 用 `capture` 采集正常样本。
2. 正常样本进入各机位 `train_good_dir`。
3. 执行 `train-patchcore`。
4. 准备 YOLO segmentation 数据集并执行 `train-yolo`。
5. 配好每个机位的 `patchcore_model_path` 和 YOLO `model_path` 后，线上跑 `inspect`，线下批测跑 `inspect-folder`。

`train_good_dir` 保存的是相机原图。训练 PatchCore 时仍会复用正式链路，先走预处理、YOLO、ROI 和 mask 构造，再拟合 PatchCore。

如果修改了 `preprocess`、YOLO 检测参数、ROI/mask、PatchCore 输入模式或 PatchCore 后端配置，必须重新执行 `train-patchcore`。模型包会保存 `pipeline_signature`，线上加载时会校验签名。

当前运行配置支持两类 torch 后端：

- `patchcore.backend = full`：默认 CNN PatchCore，backbone 为 `wide_resnet50_2`，默认特征层为 `layer2 / layer3`。
- `patchcore.backend = transformer`：ViT token PatchCore，当前支持 `vit_b_16`、`vit_b_32`、`vit_l_16`、`vit_l_32`。
- 两类后端都需要可用的 `torch / torchvision`。
- 如 `backbone_pretrained = false`，必须配置本地 `backbone_weights_path`。

Transformer 后端示例配置见 `configs/seat_defect_inspection.transformer_patchcore.example.json`。该后端只替代 PatchCore 特征提取与异常检测步骤，不替代 YOLO 定位、ROI/mask、质量门控或多机位融合。

PatchCore 参数排查见 [PATCHCORE_TUNING_GUIDE_ZH.md](./PATCHCORE_TUNING_GUIDE_ZH.md)。

如果现场还没有 YOLO 权重，可以先把 `detection.model_path` 设为 `null`，继续使用 `fallback_box` 跑完整流程。

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
- `<output_json_path 同目录>/<output_json_path.stem>_history`: 检测历史报告归档目录
- `outputs/seat_defect_inspection/yolo_training`: YOLO 训练输出

## 当前代码结构

```text
src/
├── seat_defect_core/        # 唯一检测 runtime 真源
├── seat_defect_sdk/         # 外部图片输入 SDK 门面
├── seat_defect_inspection/  # CLI、采图、训练、离线批测
├── media_inputs/            # 图片/视频/相机输入抽象
└── mvsCamera/               # 海康 MVS 适配
```

runtime 行为只在 `seat_defect_core` 维护：预处理、YOLO 推理、ROI/mask、PatchCore、颜色分支、融合、调试图和检测报告。

工程行为在 `seat_defect_inspection` 维护：CLI、配置扩展、采图、manifest、离线目录发现、PatchCore 训练编排、YOLO 训练和 LabelMe 转换。

不再保留旧 runtime 兼容导入路径。预处理、YOLO 推理、ROI、PatchCore、融合、调试产物等能力请直接从 `seat_defect_core` 导入。
