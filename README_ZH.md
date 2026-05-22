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

`benchmark` 命令通过标准化测试集对当前检测流程和模型产出可量化的性能指标。设计原则：**精简**——只保留机位训练评估、结果图像导出和 Markdown 报告三类核心能力。

### 核心能力

| 能力 | 说明 |
|------|------|
| 整件 Precision / Recall / F1 / Accuracy | ✅ 融合后整体评估 |
| **漏检率 (Miss Rate)** | ✅ FN/(TP+FN) |
| **错检率 (False Alarm Rate)** | ✅ FP/(FP+TN) |
| Mixed 轮量化评估 | ✅ 有 `ground_truth.json` 后与 Good / Defect 等同 |
| **单机位拆解指标** | ✅ 每个 camera 独立混淆矩阵 + P/R/F1/Accuracy/Miss/False Alarm |
| 逐机位标注支持 | ✅ `camera_results` 字段，机位级 GT 优先于整体 GT |
| REJECT 独立计数 | ✅ 质量不合格样本单独统计 |
| **结果图像导出** | ✅ overlay 缺陷热力叠加图，自动开启 debug artifacts |
| **Markdown 报告** | ✅ 纯文本报告，含融合指标 + 逐机位指标 + 失败案例 |

### 准备数据

在项目根目录（或通过 `--data-dir` 指定路径）创建基准数据集目录，按机位分目录存放图片：

```text
benchmark_data/
├── good/
│   ├── cam_0/           # 机位 cam_0 的图片
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── cam_1/
│   │   └── ...
│   └── ground_truth.json    # 可选：显式标注
├── defect/
│   ├── cam_0/
│   ├── cam_1/
│   └── ground_truth.json    # 可选
└── mixed/
    ├── cam_0/
    ├── cam_1/
    └── ground_truth.json    # 强烈建议 mixed 轮提供标注
```

每个机位目录下的图片数量必须一致（命名不限，按文件名排序后一一配对）。

### Ground Truth 标注文件（可选，推荐）

在每个 round 目录下放置 `ground_truth.json`，按图片索引显式标注真实标签。支持逐机位标注：

```json
{
  "version": 1,
  "samples": [
    {"index": 0, "label": "OK"},
    {"index": 1, "label": "NG", "defect_type": "scratch", "severity": "high"},
    {"index": 2, "label": "NG", "defect_type": "dent", "severity": "medium",
     "camera_results": {
       "cam_0": {"label": "NG", "defect_type": "dent"},
       "cam_1": {"label": "OK"}
     }
    },
    {"index": 3, "label": "OK"}
  ]
}
```

字段说明：

| 字段 | 必填 | 说明 |
|------|------|------|
| `index` | ✅ | 图片在机位目录中的排序位置（0-based） |
| `label` | ✅ | `"OK"` 或 `"NG"` |
| `defect_type` | 否 | 缺陷类型，如 `"scratch"`、`"dent"`、`"stain"`、`"crack"` |
| `severity` | 否 | 严重程度：`"low"` / `"medium"` / `"high"` |
| `camera_results` | 否 | 逐机位标注，key 为 `camera_id`，value 含 `label`。逐机位指标计算时优先使用此处标注 |

没有 `ground_truth.json` 时，按目录名自动推断：`good` → 全部 OK，`defect` → 全部 NG，`mixed` → 无标签（不计算指标）。

### 运行评估

```bash
# 基础用法（使用默认 ./benchmark_data）
seat-defect-inspection benchmark

# 指定数据集目录和配置
seat-defect-inspection benchmark --data-dir /path/to/dataset --config configs/my_config.json

# 只跑特定轮次
seat-defect-inspection benchmark --round good
seat-defect-inspection benchmark --round defect

# 筛选特定机位（只评估指定机位，跳过其他）
seat-defect-inspection benchmark --cameras cam_0,cam_1

# 多型号配置
seat-defect-inspection benchmark --seat-model-id seat_v2

# 自定义输出路径
seat-defect-inspection benchmark \
  --artifacts-dir outputs/my_artifacts \
  --report-output outputs/my_report.md

# 完整示例
seat-defect-inspection benchmark \
  --data-dir benchmark_data \
  --round all \
  --cameras cam_0,cam_1 \
  --artifacts-dir outputs/benchmark_artifacts \
  --report-output outputs/benchmark_report.md
```

### CLI 参数

| 参数 | 说明 |
|------|------|
| `--data-dir PATH` | 基准数据集根目录（默认 `./benchmark_data`） |
| `--config PATH` | 检测配置文件路径 |
| `--round {good,defect,mixed,all}` | 运行轮次（默认 `all`） |
| `--cameras cam_0,cam_1` | 逗号分隔的机位 ID（默认全部启用的机位） |
| `--seat-model-id ID` | 多型号配置时指定座椅型号 |
| `--artifacts-dir PATH` | 结果图输出目录（默认 `outputs/benchmark/artifacts_<timestamp>`） |
| `--report-output PATH` | Markdown 报告路径（默认 `outputs/benchmark/benchmark_report_<timestamp>.md`） |

### 执行逻辑

1. **图片收集**：对每个机位目录，按文件名排序后得到有序列表。
2. **样本配对**：按索引依次取各机位的同序号图片组合成一个样本。
3. **图片加载**：通过 `cv2.imread` 读取图片构造 `InspectionFrame`，不经过采图层。
4. **开启 debug artifacts**：benchmark 自动启用 `debug_artifacts_enabled` 和 `"overlay"`，结束后恢复原设置。
5. **检测流程**：调用 `inspect_frames()` 执行 YOLO → ROI → PatchCore → 颜色 → 融合的完整流程。
6. **结果图导出**：从每个机位的 `CameraInspectionResult.overlay_image` 导出 PNG 到 `--artifacts-dir`。
7. **Ground Truth 匹配**：检查 `ground_truth.json` → 按索引匹配标签 → 不存在则按目录名推断（good=OK, defect=NG, mixed=无标签）。
8. **指标计算**：融合后混淆矩阵 + 二元指标 → 逐机位混淆矩阵 + 指标（优先使用 `camera_results` 标注）。
9. **报告生成**：输出 Markdown 报告。

### Markdown 报告结构

生成的 Markdown 报告包含以下章节：

1. **报告头** — 生成时间、配置文件、机位列表、轮次列表
2. **Combined Metrics** — 所有标注轮次的融合后综合指标（Precision/Recall/F1/Accuracy/Miss Rate/False Alarm）
3. **逐轮次结果** — 每轮包含：
   - 样本统计（OK / NG / REJECT 计数）
   - 融合后混淆矩阵和二元指标
   - **逐机位指标表** — 每个 camera 独立的 TP/TN/FP/FN + P/R/F1/Accuracy/Miss/False Alarm
   - 失败案例列表（漏检 + 误报）
   - 结果图路径列表

### 输出示例

#### 终端输出

```text
============================================================
  Benchmark round: defect
============================================================
  [0001/0100] x NG  part_id=defect_0000
  [0002/0100] x NG  part_id=defect_0001
  ...

  Results: Total=100
  OK=5  NG=93  REJECT=2
  TP=93  TN=0  FP=0  FN=5
  Precision: 100.0%  Recall: 94.9%  F1: 97.4%  Accuracy: 94.9%
  Miss Rate (漏检率): 5.1%  False Alarm (错检率): 0.0%

  Per-camera metrics:
  Camera       TP    TN    FP    FN   Prec    Rec     F1
  cam_0        95     0     0     5  100.0%  95.0%  97.4%
  cam_1        93     0     0     7  100.0%  93.0%  96.4%

============================================================
  BENCHMARK SUMMARY
============================================================
  Cameras: cam_0, cam_1
  Rounds: good, defect
  Precision (精准率): 100.0%
  Recall    (召回率): 94.9%
  F1 Score  (F1 值):  97.4%
  Accuracy  (准确率): 97.5%
  Miss Rate (漏检率): 5.1%
  False Alarm (错检率): 0.0%
============================================================
```

### 量化指标说明

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| **Precision（精准率）** | 被判为 NG 的样本中有多少确实是 NG | `TP / (TP + FP)` |
| **Recall（召回率）** | 真实 NG 样本中有多少被正确检出 | `TP / (TP + FN)` |
| **F1 Score（F1 值）** | 精准率与召回率的调和平均 | `2 × P × R / (P + R)` |
| **Accuracy（准确率）** | 所有样本中判定正确的比例 | `(TP + TN) / Total` |
| **Miss Rate（漏检率）** | 缺陷样本漏判比例 | `FN / (TP + FN)` |
| **False Alarm Rate（错检率）** | 正常样本误报比例 | `FP / (FP + TN)` |

> **融合指标 vs 逐机位指标**：融合指标反映多机位综合判定效果（受 `FusionConfig.ng_strategy` 控制），逐机位指标反映单机位模型独立表现。逐机位指标优先使用 `camera_results` 标注，无逐机位标注时回退到样本整体 `label`。

### 判读指南

- **Good 轮错检率高** → 模型对正常纹理变化过于敏感，考虑补充类似正常纹理参与训练。
- **Defect 轮漏检率高 / 召回率低** → 查看逐机位指标，定位漏检集中在哪个机位，针对性补充该机位的缺陷样本。
- **精准率低** → 误报多，正常件被频繁打回，影响生产效率。
- **召回率低** → 漏检多，缺陷件可能流出。查看逐机位指标，可能只有一个机位识别不到缺陷。
- **单机位指标差异大** → 说明某些视角的模型较弱，针对该机位补充训练样本或调整阈值。
- **F1 值**是精准率和召回率的综合指标，迭代模型时以 F1 提升为主要优化方向。
- **Mixed 轮**反映接近真实产线分布下的 OK/NG 比例。有 `ground_truth.json` 时与 Good/Defect 等同评估。
- **REJECT** 表示图片质量不达标（模糊、过曝、欠曝等），流程拒绝检测。REJECT 在混淆矩阵中计入 positive call（安全侧），但独立计数可辅助判断采图质量是否正常。

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
