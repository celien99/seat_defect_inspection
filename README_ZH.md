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

`benchmark` 命令通过标准化测试集对当前检测流程和模型产出可量化的性能指标，支持两项核心能力：

1. **隐式标注**（向后兼容）：按目录名推断真值（good=OK, defect=NG），无需额外标注即可快速跑分
2. **显式标注**（推荐）：通过 `ground_truth.json` 为每张图片标注真实标签, 解锁按机位 / 按缺陷类型 / 置信区间等完整评估能力

### 新版核心收益（v.s 旧版隐式标注）

| 能力 | 旧版（隐式） | 新版（显式 ground_truth.json） |
|------|-------------|-------------------------------|
| 整件 Precision / Recall / F1 / Accuracy | ✅ | ✅ |
| 误报率 / 漏检率 / 检出率 | ✅ | ✅ |
| Mixed 轮量化评估 | ❌ 仅能统计 OK/NG 比例 | ✅ 有真实标签后与 Good / Defect 等同 |
| 95% 威尔逊置信区间 | ❌ | ✅ 所有率指标附带 CI |
| 单机位拆解指标 | ❌ | ✅ 每个 camera 独立 P/R/F1 |
| 异常分数分布 | ❌ | ✅ OK vs NG 的 min/max/mean/median/std/p5/p95 |
| 缺陷类型召回率 | ❌ | ✅ scratch / dent / stain 等分别统计 |
| ROC/PR 数据导出 | ❌ | ✅ `--export-curves` 导出 CSV |

### 准备数据

`benchmark` 默认还会为每个样本导出图像产物：

- `overlay.png`：与原检测程序一致的缺陷热力叠加图

默认目录是 `outputs/seat_defect_inspection/benchmark_artifacts/`。需要关闭时加 `--no-artifacts`，需要改路径时加 `--artifacts-dir <path>`.

在项目根目录创建 `benchmark_data/`，按机位分目录存放图片。每个机位目录下的图片数量必须一致（命名不限，按文件名排序后一一配对）：

```text
benchmark_data/
├── good/
│   ├── cam_0/           # 机位 cam_0 的图片
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── cam_1/
│   │   └── ...
│   └── ground_truth.json    # 可选：显式标注（见下方）
├── defect/
│   ├── cam_0/
│   ├── cam_1/
│   └── ground_truth.json    # 可选
└── mixed/
    ├── cam_0/
    ├── cam_1/
    └── ground_truth.json    # 强烈建议 mixed 轮提供标注
```

### Ground Truth 标注文件（可选，推荐）

在每个 round 目录下放置 `ground_truth.json`，按图片索引显式标注真实标签：

```json
{
  "version": 1,
  "samples": [
    {"index": 0, "label": "OK"},
    {"index": 1, "label": "NG", "defect_type": "scratch", "severity": "high"},
    {"index": 2, "label": "NG", "defect_type": "dent", "severity": "medium"},
    {"index": 3, "label": "OK"}
  ]
}
```

字段说明：

| 字段 | 必填 | 说明 |
|------|------|------|
| `index` | ✅ | 图片在机位目录中的排序位置（0-based） |
| `label` | ✅ | `"OK"` 或 `"NG"` |
| `defect_type` | 否 | 缺陷类型，如 `"scratch"`、`"dent"`、`"stain"`、`"crack"`。标注后启用缺陷类型召回率统计 |
| `severity` | 否 | 严重程度：`"low"` / `"medium"` / `"high"`（供后续按严重度筛选） |

各轮次的标注策略：

| 轮次 | 推荐做法 |
|------|----------|
| Good | 图片本身就全是 OK，可不加 `ground_truth.json`（自动推断 label=OK），无需手动建 100 条 `{"index":0,"label":"OK"}` |
| Defect | **建议添加**：为 NG 样本标注 `defect_type` 和 `severity`，即可获得按缺陷类型的召回率分析 |
| Mixed | **强烈建议添加**：mixed 轮无标注完全无法评估，添加后即可与 Good/Defect 轮同等参与所有指标计算 |

### 运行评估

```bash
# 基础用法（兼容旧版，隐式标注）
seat-defect-inspection benchmark

# 指定配置
seat-defect-inspection benchmark --config configs/my_custom_config.json

# 只跑特定轮次
seat-defect-inspection benchmark --round good
seat-defect-inspection benchmark --round defect

# 筛选特定机位
seat-defect-inspection benchmark --cameras cam_0,cam_1

# 导出 ROC/PR 曲线数据（需 ground_truth.json）
seat-defect-inspection benchmark --export-curves outputs/benchmark_curves

# 保存完整结果为 JSON
seat-defect-inspection benchmark --output outputs/benchmark_results.json

# 组合使用
seat-defect-inspection benchmark --round all --cameras cam_0 --export-curves outputs/curves --output outputs/results.json
```

### 执行逻辑

1. **图片收集**：对每个机位目录，按文件名 `sorted()` 排序后得到有序列表。
2. **样本配对**：按索引 `idx = 0 → N-1` 依次取各机位的第 `idx` 张图片组合成一个样本。
3. **图片加载**：直接通过 `cv2.imread` 读取图片构造 `InspectionFrame`，不再经过采图层 `AcquisitionService.capture`，简洁高效。
4. **检测流程**：调用 `inspect_frames()` 执行完整 YOLO → ROI → PatchCore → 颜色 → 融合流程。
5. **Ground Truth 匹配**：检查 `ground_truth.json` 是否存在 → 存在则按索引匹配标签 → 不存在则按目录名自动推断（good=OK, defect=NG, mixed=无标签）。
6. **跨轮次隔离**：`good` / `defect` / `mixed` 三个目录完全独立，互不干扰。

### 输出示例

#### 无标注时（与旧版兼容的输出格式）

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

#### 有 `ground_truth.json` 时（完整输出）

```text
============================================================
  BENCHMARK SUMMARY
============================================================
  Cameras: cam_0, cam_1

  [Defect (all NG)]
    Samples: 100
    OK: 8  |  NG: 90  |  REJECT: 2

    Ground truth:
    TP=90  TN=0  FP=0  FN=8
    Precision: 100.0%  [95% CI: 95.8–100.0%]
    Recall:    91.8%  [95% CI: 84.7–96.0%]
    F1 Score:  95.7%
    Accuracy:  91.8%  [95% CI: 84.7–96.0%]
    FPR:       N/A    [95% CI: N/A]

    Per-camera metrics:
    Camera       TP    TN    FP    FN   Prec    Rec     F1
    cam_0        88     0     0    12  100.0%  88.0%  93.6%
    cam_1        85     0     0    15  100.0%  85.0%  91.9%

    [OK] anomaly scores (n=0): (no OK samples in this round)
    [NG] anomaly scores (n=90):
      min=0.5234  max=2.8912  mean=1.2456  median=1.1023  std=0.4821  p5=0.6123  p95=2.3401

    Defect-type recall:
      dent            15/15   recall=100.0%
      scratch         60/65   recall=92.3%
      stain           10/10   recall=100.0%
      tear             5/8     recall=62.5%

  [Combined Metrics (all labeled rounds)]
    TP=90  TN=95  FP=5  FN=8
    Precision (精准率): 94.7%  [95% CI: 88.0–97.8%]
    Recall    (召回率): 91.8%  [95% CI: 84.7–96.0%]
    F1 Score  (F1 值):  93.2%
    Accuracy  (准确率): 92.5%  [95% CI: 87.8–95.5%]
    FPR       (误报率):  5.0%  [95% CI:  2.2–10.0%]
============================================================
```

### 量化指标说明

#### 各轮次基础指标

| 指标 | 含义 | 计算方式 | 适用场景 |
|------|------|----------|----------|
| **False positive rate（误报率）** | 无缺陷样本被误判为 NG/REJECT 的比例 | `(NG + REJECT) / total × 100%` | Good 轮评估模型过杀程度 |
| **Miss rate（漏检率）** | 有缺陷样本被漏判为 OK 的比例 | `OK / total × 100%` | Defect 轮评估模型漏放风险 |
| **Detection rate（检出率）** | 有缺陷样本被正确判定为 NG 的比例 | `NG / total × 100%` | Defect 轮评估模型检出能力 |

#### 综合指标（所有标注轮次合并计算）

| 指标 | 含义 | 计算方式 | 含 95% CI |
|------|------|----------|-----------|
| **Precision（精准率）** | 被判为 NG 的样本中有多少确实是 NG | `TP / (TP + FP)` | ✅ (Wilson) |
| **Recall（召回率）** | 真实 NG 样本中有多少被正确检出 | `TP / (TP + FN)` | ✅ (Wilson) |
| **F1 Score（F1 值）** | 精准率与召回率的调和平均 | `2 × P × R / (P + R)` | ❌ |
| **Accuracy（准确率）** | 所有样本中判定正确的比例 | `(TP + TN) / (TP + TN + FP + FN)` | ✅ (Wilson) |
| **FPR（误报率）** | 无缺陷样本被判为 NG/REJECT 的比例 | `FP / (FP + TN)` | ✅ (Wilson) |

#### 新增评估维度（需 ground_truth.json）

| 维度 | 含义 | 数据来源 |
|------|------|----------|
| **95% 置信区间** | 指标在 95% 置信水平下的取值范围，样本量越小越宽 | Wilson score interval（无外部依赖） |
| **单机位指标** | 每个 camera_id 独立计算的 P/R/F1，用于定位哪个视角是检测短板 | 按 `camera_results[].status` vs `ground_truth_label` 逐机位构建混淆矩阵 |
| **异常分数分布** | OK / NG 样本的 anomaly score 分布统计（min/max/mean/median/std/p5/p95），用于判断阈值分离度 | 从 `texture_result.score` 或 `region_results[].texture_result.score` 提取 |
| **缺陷类型召回** | 按 `defect_type` 分别统计检出率，识别模型对哪些缺陷类型不敏感 | `ground_truth.json` 中 `defect_type` 字段 |
| **ROC/PR 数据** | 每条记录包含 sample_index + gt_label + camera_id + anomaly_score + threshold | `--export-curves` 导出为 CSV，用于外部画 ROC/PR 曲线 |

### ROC/PR 曲线数据导出

```bash
seat-defect-inspection benchmark --export-curves outputs/curves
```

导出文件结构：

```text
outputs/curves/
├── good_scores.csv
├── defect_scores.csv
└── mixed_scores.csv
```

CSV 格式：

| sample_index | part_id | ground_truth_label | camera_id | anomaly_score | is_anomaly | threshold |
|-------------|---------|-------------------|-----------|---------------|------------|-----------|
| 0 | defect_0000 | NG | cam_0 | 1.234 | True | 0.500 |
| 0 | defect_0000 | NG | cam_1 | 0.876 | False | 0.500 |

每行 = 一个样本 × 一个机位。`ground_truth_label` 为样本级标签，`anomaly_score` 为该机位的异常分数。可直接导入 Python/R 中画 ROC 曲线。

### 完整结果 JSON 导出

```bash
seat-defect-inspection benchmark --output outputs/benchmark_results.json
```

JSON 包含所有轮次的完整 records 列表（每个样本的 label、status、各机位 score 等），以及所有聚合指标、CI、异常分数分布、缺陷类型召回等统计结果，可用于自动化报告或后续分析。

### 判读指南

- **Good 轮误报率高** → 模型对正常纹理变化过于敏感，可查看 OK 样本的异常分数分布，若 p95 远低于阈值说明部分样本异常偏高，考虑补充类似正常纹理参与训练。
- **Defect 轮漏检率高 / 召回率低** → 查看**缺陷类型召回率**，定位漏检集中在哪种缺陷类型，针对性补充该类缺陷样本重新训练。
- **精准率低** → 误报多，正常件被频繁打回，影响生产效率。查看 FPR 和 Precision CI 确认问题是否统计显著。
- **召回率低** → 漏检多，缺陷件可能流出。查看**单机位指标**，可能只有一个机位识别不到缺陷（如缺陷仅在特定角度可见）。
- **单机位指标差异大** → 说明某些视角的模型较弱，针对该机位补充训练样本或调整阈值。
- **F1 值**是精准率和召回率的综合指标，迭代模型时以 F1 提升为主要优化方向。
- **异常分数分布重叠** → OK 和 NG 的 score 分布如果高度重叠（p95(OK) 接近甚至超过 p5(NG)），说明当前阈值无法有效分离正负样本，需重新训练或调整特征提取器。
- **Mixed 轮**反映接近真实产线分布下的 OK/NG 比例。有 `ground_truth.json` 时与 Good/Defect 等同评估；无标注时仅辅助判断产线直通率。
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
