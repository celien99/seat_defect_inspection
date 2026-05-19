# Seat Defect Inspection 独立说明

当前主检测架构只保留 `seat_defect_core`。`seat_defect_core` 是唯一 inspect runtime 真源；CLI、采图、训练、离线批测属于工程工具层，只负责把图片交给 core 主流程。

更细的图像链路说明见 [IMAGE_PIPELINE_DETAILS_ZH.md](./IMAGE_PIPELINE_DETAILS_ZH.md)，架构边界见 [PROJECT_ARCHITECTURE_ZH.md](./PROJECT_ARCHITECTURE_ZH.md)。

## 快速开始

工程工具层可以继续使用现有 Python 3.10 开发环境：

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

## LabVIEW Python 3.8.5 Core 运行环境

`seat_defect_core` 可作为兼容 Python 3.8.5 的检测运行时交付给 LabVIEW 或现场工具。这个运行时只覆盖 core 检测链路：加载已有 YOLO/PatchCore 模型，接收外部传入的图片或图片路径，在 CPU 上执行检测，并返回可 JSON 序列化的结果。

公共机推荐环境：

```bash
conda create -n seat-defect-core-py38 python=3.8.5 -y
conda activate seat-defect-core-py38
pip install -r requirements-core-py38-cpu.txt
pip install --no-build-isolation .
```

离线安装时，先在可联网的 Python 3.8.5 机器上准备 wheel 缓存，再拷贝 `wheelhouse` 到 LabVIEW 公共机：

```bash
python -m pip download --only-binary=:all: -r requirements-core-py38-cpu.txt -d wheelhouse
python -m pip wheel --no-deps --no-build-isolation . -w wheelhouse
python -m pip install --no-index --find-links wheelhouse -r requirements-core-py38-cpu.txt
python -m pip install --no-index --find-links wheelhouse seat-defect-core
```

现场配置中，模型路径和报告路径必须能被 LabVIEW 进程访问。生产环境建议设置 `backbone_device = cpu`、`debug_artifacts_enabled = false`，并把 `output_json_path` / `debug_dir` 放到可写目录。

## 工程工具命令

以下命令属于工程工具层，不是主检测架构入口。外部系统和产线集成应直接调用 `seat_defect_core`。

```bash
seat-defect-inspection capture --config configs/seat_defect_inspection.mvs.json --part-id seat_000001
seat-defect-inspection train-patchcore --config configs/seat_defect_inspection.mvs.json
seat-defect-inspection train-yolo --config configs/seat_defect_inspection.mvs.json
seat-defect-inspection inspect --config configs/seat_defect_inspection.mvs.json --part-id seat_000001
seat-defect-inspection inspect-folder --config configs/seat_defect_inspection.mvs.json --input-dir offline_samples
seat-defect-inspection benchmark --config configs/seat_defect_inspection.mvs.json
```

## Benchmark — 检测流程量化评估

`benchmark` 命令通过三轮标准化测试，对当前检测流程和模型产出可量化的性能指标。

### 三轮测试设计

| 轮次 | 目录 | 样本真值 | 考察指标 |
|------|------|----------|----------|
| Good | `benchmark_data/good/` | 全部为 OK（无缺陷） | 误报率（False Positive Rate） |
| Defect | `benchmark_data/defect/` | 全部为 NG（有缺陷） | 漏检率（Miss Rate）、检出率（Detection Rate） |
| Mixed | `benchmark_data/mixed/` | OK / NG 随机混合 | 真实分布下的 OK/NG 分布 |

Good 轮和 Defect 轮的结果会合并计算 **精准率（Precision）、召回率（Recall）、F1 值、准确率（Accuracy）**，作为流程的综合评价指标。

### 准备数据

在项目根目录创建 `benchmark_data/`，按机位分目录存放图片。每个机位目录下的图片数量必须一致（命名不限，按文件名排序后一一配对）：

```text
benchmark_data/
├── good/                # 全部正确样本
│   ├── cam_0/           # 机位 cam_0 的图片
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── cam_1/           # 机位 cam_1 的图片（数量与 cam_0 一致）
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── ...
├── defect/              # 全部缺陷样本
│   ├── cam_0/
│   │   └── ...
│   ├── cam_1/
│   │   └── ...
│   └── ...
└── mixed/               # OK/NG 杂糅随机样本
    ├── cam_0/
    │   └── ...
    ├── cam_1/
    │   └── ...
    └── ...
```

每个机位目录下的图片**按文件名排序**后按索引一一配对，不同机位之间**不要求文件名一致**。例如 `good/cam_0/a.jpg` 会与 `good/cam_1/b.jpg` 配对，只要两者在各机位目录中的排序位置相同。

### 运行评估

```bash
# 使用默认配置
seat-defect-inspection benchmark

# 指定配置文件
seat-defect-inspection benchmark --config configs/my_custom_config.json
```

命令会遍历三轮数据集，每个样本逐一检测并实时输出进度和判定结果。三轮跑完后打印汇总报告。

### 量化指标说明

输出示例：

```text
============================================================
  BENCHMARK SUMMARY
============================================================
  Cameras: cam_0, cam_1

  [Good (all OK)]
    Samples: 100
    OK: 95  |  NG: 3  |  REJECT: 2
    False positive rate: 5.0%

  [Defect (all NG)]
    Samples: 100
    OK: 8  |  NG: 90  |  REJECT: 2
    Miss rate: 8.0%  |  Detection rate: 90.0%

  [Mixed]
    Samples: 200
    OK: 104  |  NG: 93  |  REJECT: 3
    OK rate: 52.0%  |  NG rate: 46.5%

  [Combined Metrics (Good + Defect)]
    TP=90  TN=95  FP=5  FN=8
    Precision (精准率): 94.7%
    Recall    (召回率): 91.8%
    F1 Score  (F1 值):  93.2%
    Accuracy  (准确率): 92.5%
============================================================
```

#### 各轮次指标

| 指标 | 含义 | 计算方式 | 来源 |
|------|------|----------|------|
| **False positive rate（误报率）** | 无缺陷样本被误判为 NG/REJECT 的比例 | `(NG + REJECT) / total × 100%` | Good 轮 |
| **Miss rate（漏检率）** | 有缺陷样本被漏判为 OK 的比例 | `OK / total × 100%` | Defect 轮 |
| **Detection rate（检出率）** | 有缺陷样本被正确判定为 NG 的比例 | `NG / total × 100%` | Defect 轮 |

#### 综合指标（Good + Defect 合并计算）

将 Good 轮作为负样本集（真值 = OK）、Defect 轮作为正样本集（真值 = NG），构建混淆矩阵：

```
               预测 OK    预测 NG/REJECT
真值 OK         TN           FP
真值 NG         FN           TP
```

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| **Precision（精准率）** | 被判为 NG 的样本中有多少确实是 NG | `TP / (TP + FP) × 100%` |
| **Recall（召回率）** | 真实 NG 样本中有多少被正确检出 | `TP / (TP + FN) × 100%` |
| **F1 Score（F1 值）** | 精准率与召回率的调和平均，综合评价模型 | `2 × P × R / (P + R)` |
| **Accuracy（准确率）** | 所有样本中判定正确的比例 | `(TP + TN) / (TP + TN + FP + FN) × 100%` |

#### 判读指南

- **Good 轮误报率高** → 模型对正常纹理变化过于敏感，考虑适当放宽 PatchCore 阈值或补充更多正常样本参与训练。
- **Defect 轮漏检率高 / 召回率低** → 模型对当前缺陷类型不敏感，需要补充对应类型的缺陷样本重新训练。
- **精准率低** → 误报多，正常件被频繁打回，影响生产效率。
- **召回率低** → 漏检多，缺陷件可能流出，影响出货质量。
- **F1 值**是精准率和召回率的综合指标，迭代模型时以 F1 提升为主要优化方向。
- **Mixed 轮**反映接近真实产线分布下的 OK/NG 比例，辅助判断产线直通率。
- **REJECT** 表示图片质量不达标（模糊、过曝、欠曝等），流程拒绝检测，不计入分类判定。若 REJECT 比例异常高，应检查采图质量或调整质量门禁阈值。

也可以给 LabVIEW 或现场工具使用 INI 配置，JSON 主格式不变：

```bash
seat-defect-inspection inspect --config configs/seat_defect_inspection.labview.example.ini --part-id seat_000001
```

INI section 约定：

- `[seat_defect_inspection]`：顶层路径、开关、默认工件等字段
- `[fusion]`：整件融合策略
- `[camera.<camera_id>]`：顶层单机位
- `[camera.<camera_id>.detection]`、`roi`、`roi.alignment`、`patchcore`、`color_branch`
- `[camera.<camera_id>.region.<region_id>]`：单机位局部区域
- 多型号时使用 `[seat_model.<seat_model_id>]` 和 `[seat_model.<seat_model_id>.camera.<camera_id>]`

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
