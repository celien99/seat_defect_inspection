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

`benchmark` 命令通过标准化测试集对当前检测流程和模型产出可量化的性能指标，支持三项核心能力：

1. **隐式标注**（向后兼容）：按目录名推断真值（good=OK, defect=NG），无需额外标注即可快速跑分
2. **显式标注**（推荐）：通过 `ground_truth.json` 为每张图片标注真实标签，解锁按机位 / 按缺陷类型 / 置信区间等完整评估能力
3. **报告生成**：支持 HTML（可打印为 PDF）和 PPTX 两种格式，生成包含图表、指标、失败案例的完整汇报文档

### 新版核心收益

| 能力 | 说明 |
|------|------|
| 整件 Precision / Recall / F1 / Accuracy | ✅ 95% Wilson CI |
| **漏检率 (Miss Rate)** | ✅ FN/(TP+FN)，附带 95% CI |
| **错检率 (False Alarm Rate)** | ✅ FP/(FP+TN)，附带 95% CI |
| Mixed 轮量化评估 | ✅ 有 `ground_truth.json` 后与 Good / Defect 等同 |
| 单机位拆解指标 | ✅ 每个 camera 独立 P/R/F1 |
| 异常分数分布 | ✅ OK vs NG 的 min/max/mean/median/std/p5/p95 |
| 缺陷类型召回率 + 精准率 + F1 | ✅ 逐缺陷类型统计 |
| **ROC 曲线 + AUC** | ✅ `--threshold-sweep` 启用，梯形积分计算 |
| **PR 曲线 + AUC** | ✅ 同上 |
| **推理耗时分析** | ✅ mean/std/P50/P95/P99 |
| **HTML 报告** | ✅ 自包含单文件，图表 base64 嵌入，浏览器打开 |
| **PPTX 报告** | ✅ python-pptx 生成，图表嵌入幻灯片 |
| JSON 结果导出 | ✅ `--output` 保存完整结果 |

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
| `defect_type` | 否 | 缺陷类型，如 `"scratch"`、`"dent"`、`"stain"`、`"crack"`。标注后启用缺陷类型召回率统计 |
| `severity` | 否 | 严重程度：`"low"` / `"medium"` / `"high"` |
| `camera_results` | 否 | 逐机位标注，key 为 `camera_id`，value 含 `label` 和可选 `defect_type`。用于定位缺陷出现的具体视角 |

### 运行评估

```bash
# 基础用法（兼容旧版，使用默认 ./benchmark_data）
seat-defect-inspection benchmark

# 指定数据集目录
seat-defect-inspection benchmark --data-dir /path/to/dataset

# 指定配置
seat-defect-inspection benchmark --config configs/my_custom_config.json

# 只跑特定轮次
seat-defect-inspection benchmark --round good
seat-defect-inspection benchmark --round defect

# 筛选特定机位
seat-defect-inspection benchmark --cameras cam_0,cam_1

# 生成 HTML 报告（默认格式，可打印为 PDF）
seat-defect-inspection benchmark --report-format html --report-output outputs/report.html

# 生成 PPTX 报告
seat-defect-inspection benchmark --report-format pptx --report-output outputs/report.pptx

# 启用 ROC/PR 曲线计算（含 AUC）
seat-defect-inspection benchmark --threshold-sweep --sweep-steps 50

# 导出 ROC/PR 曲线 CSV 数据
seat-defect-inspection benchmark --export-curves outputs/benchmark_curves

# 保存完整结果为 JSON
seat-defect-inspection benchmark --output outputs/benchmark_results.json

# 组合使用
seat-defect-inspection benchmark \
  --data-dir benchmark_data \
  --round all \
  --cameras cam_0,cam_1 \
  --threshold-sweep \
  --report-format html \
  --report-output outputs/benchmark_report.html \
  --export-curves outputs/curves \
  --output outputs/results.json
```

### 新增 CLI 参数

| 参数 | 说明 |
|------|------|
| `--data-dir PATH` | 基准数据集根目录（默认 `./benchmark_data`） |
| `--report-format {html,pptx,json}` | 报告输出格式（默认 `html`） |
| `--report-output PATH` | 报告输出文件路径（自动生成默认路径） |
| `--threshold-sweep` | 启用阈值扫描，计算 ROC/PR 曲线及 AUC |
| `--sweep-steps N` | 阈值扫描步数（默认 50） |
| `--seat-model-id ID` | 多型号配置时指定座椅型号 |

### 执行逻辑

1. **图片收集**：对每个机位目录，按文件名 `sorted()` 排序后得到有序列表。
2. **样本配对**：按索引 `idx = 0 → N-1` 依次取各机位的第 `idx` 张图片组合成一个样本。
3. **图片加载**：通过 `cv2.imread` 读取图片构造 `InspectionFrame`，不经过采图层。
4. **检测流程**：调用 `inspect_frames()` 执行 YOLO → ROI → PatchCore → 颜色 → 融合的完整流程，同时记录推理耗时。
5. **Ground Truth 匹配**：检查 `ground_truth.json` → 按索引匹配标签 → 不存在则按目录名推断（good=OK, defect=NG, mixed=无标签）。
6. **指标计算**：混淆矩阵 → 二元指标 + Wilson CI → 单机位拆解 → 缺陷类型分析 → 分值分布 → 时序统计 → 可选 ROC/PR 曲线。
7. **报告生成**：根据 `--report-format` 生成对应格式的汇报文档。

### HTML 报告结构

生成的 HTML 报告包含 11 个章节，所有图表以 base64 嵌入，单文件即可在任何浏览器中打开并打印为 PDF：

1. **执行摘要** — 核心指标评分卡（精准率/召回率/F1/准确率/漏检率/错检率）
2. **测试场景** — 检测流程描述、测试配置参数
3. **数据构成** — 各轮次样本量、OK/NG 占比、标注来源
4. **总体指标** — 每轮混淆矩阵热力图 + Precision/Recall/F1/Accuracy/Miss Rate/False Alarm
5. **按机位分析** — 各机位独立指标表 + 分组柱状图
6. **按缺陷类型分析** — 逐缺陷类型召回率/精准率表 + 柱状图
7. **ROC & PR 曲线** — 含 AUC 标注（需 `--threshold-sweep`）
8. **分值分布** — OK vs NG 异常分值的重叠直方图 + 统计摘要表
9. **推理耗时** — mean/std/P50/P95/P99 耗时统计
10. **失败案例分析** — 漏检样本（GT=NG, Pred=OK）和误报样本（GT=OK, Pred=NG）的详细信息，含图像路径和各机位异常分值
11. **优化建议** — 基于指标的自动建议（漏检率高 → 检查决策阈值；错检率高 → 调整融合策略）

### 输出示例

#### 终端输出（基础指标）

```text
============================================================
  Benchmark round: defect
============================================================
  [0001/100] x NG  part_id=defect_0001
  [0002/100] ✓ OK  part_id=defect_0002
  ...

  Results: Total=100
  TP=90  TN=0  FP=0  FN=8
  Precision: 100.0%  Recall: 90.0%  F1: 94.7%  Accuracy: 90.0%
  Miss Rate (漏检率): 8.0%  False Alarm (错检率): 0.0%
  ROC AUC: 0.9634  PR AUC: 0.8921
  Timing: mean=152ms  p50=148ms  p95=201ms

============================================================
  BENCHMARK SUMMARY
============================================================
  Cameras: cam_0, cam_1
  Rounds: good, defect, mixed
  Precision (精准率): 97.8%
  Recall    (召回率): 91.8%
  F1 Score  (F1 值):  94.7%
  Accuracy  (准确率): 93.5%
  Miss Rate (漏检率): 8.2%
  False Alarm (错检率): 2.1%
============================================================
```

### 量化指标说明

#### 各轮次基础指标

| 指标 | 含义 | 计算方式 | 适用场景 |
|------|------|----------|----------|
| **Miss Rate（漏检率）** | 有缺陷样本被漏判为 OK 的比例 | `FN / (TP + FN) × 100%` | Defect 轮评估模型漏放风险 |
| **False Alarm Rate（错检率）** | 无缺陷样本被误判为 NG 的比例 | `FP / (FP + TN) × 100%` | Good 轮评估模型过杀程度 |
| **Detection Rate（检出率）** | 有缺陷样本被正确判定为 NG 的比例 | `TP / (TP + FN) × 100%` | Defect 轮评估模型检出能力 |

#### 综合指标（所有标注轮次合并计算）

| 指标 | 含义 | 计算方式 | 含 95% CI |
|------|------|----------|-----------|
| **Precision（精准率）** | 被判为 NG 的样本中有多少确实是 NG | `TP / (TP + FP)` | ✅ (Wilson) |
| **Recall（召回率）** | 真实 NG 样本中有多少被正确检出 | `TP / (TP + FN)` | ✅ (Wilson) |
| **F1 Score（F1 值）** | 精准率与召回率的调和平均 | `2 × P × R / (P + R)` | ❌ |
| **Accuracy（准确率）** | 所有样本中判定正确的比例 | `(TP + TN) / Total` | ✅ (Wilson) |
| **Miss Rate（漏检率）** | 缺陷样本漏判比例 | `FN / (TP + FN)` | ✅ (Wilson) |
| **False Alarm Rate（错检率）** | 正常样本误报比例 | `FP / (FP + TN)` | ✅ (Wilson) |

#### 新增评估维度

| 维度 | 含义 | 数据来源 |
|------|------|----------|
| **ROC 曲线 + AUC** | 遍历决策阈值绘制的 TPR-FPR 曲线，AUC 越高模型区分能力越强（需 `--threshold-sweep`） | 逐机位 anomaly_score vs ground_truth |
| **PR 曲线 + AUC** | 遍历决策阈值绘制的 Precision-Recall 曲线，适合不平衡数据集评估 | 同上 |
| **推理耗时统计** | 每个样本从读图到融合判定的总耗时分布 | `perf_counter()` 计时 |
| **单机位指标** | 每个 camera_id 独立计算的 P/R/F1，用于定位哪个视角是检测短板 | 按 `camera_results[].status` vs `ground_truth_label` |
| **异常分数分布** | OK / NG 样本的 anomaly score 分布统计（min/max/mean/median/std/p5/p95） | 从 `texture_result.score` 提取 |
| **缺陷类型分析** | 按 `defect_type` 分别统计检出率、精准率、F1 | `ground_truth.json` 中 `defect_type` 字段 |
| **逐机位标注** | 每个样本的每个机位可独立标注标签和缺陷类型 | `camera_results` 字段 |

### ROC/PR 曲线

两种方式获取曲线：

**方式一：代码内计算（推荐）**
```bash
seat-defect-inspection benchmark --threshold-sweep --sweep-steps 100
```
使用每张图片的 anomaly score 进行阈值扫描，通过梯形积分计算 ROC AUC 和 PR AUC，结果直接呈现在 HTML/PPTX 报告中。

**方式二：导出 CSV 外部分析**
```bash
seat-defect-inspection benchmark --export-curves outputs/curves
```

### 完整结果 JSON 导出

```bash
seat-defect-inspection benchmark --output outputs/benchmark_results.json
```

JSON 包含所有轮次的完整 records 列表（每个样本的 label、status、各机位 score、inference_timing_ms 等），以及所有聚合指标、CI、异常分数分布、缺陷类型分析等统计结果。

### 判读指南

- **Good 轮错检率高** → 模型对正常纹理变化过于敏感，可查看 OK 样本的异常分数分布，若 p95 远低于阈值说明部分样本异常偏高，考虑补充类似正常纹理参与训练。
- **Defect 轮漏检率高 / 召回率低** → 查看**缺陷类型召回率**，定位漏检集中在哪种缺陷类型，针对性补充该类缺陷样本重新训练。
- **精准率低** → 误报多，正常件被频繁打回，影响生产效率。查看 FPR 和 Precision CI 确认问题是否统计显著。
- **召回率低** → 漏检多，缺陷件可能流出。查看**单机位指标**，可能只有一个机位识别不到缺陷（如缺陷仅在特定角度可见）。
- **单机位指标差异大** → 说明某些视角的模型较弱，针对该机位补充训练样本或调整阈值。
- **F1 值**是精准率和召回率的综合指标，迭代模型时以 F1 提升为主要优化方向。
- **异常分数分布重叠** → OK 和 NG 的 score 分布如果高度重叠（p95(OK) 接近甚至超过 p5(NG)），说明当前阈值无法有效分离正负样本。
- **ROC AUC 低** → 模型对不同阈值的鲁棒性差，AUC < 0.8 建议检查特征提取器或训练数据质量。
- **推理耗时过高** → 查看 P95/P99 耗时，如超过产线节拍要求，考虑优化 backbone、降低 image_size 或 patch_size。
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
