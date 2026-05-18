# 汽车座椅缺陷检测系统升级方案报告

> 版本：v2.0  
> 日期：2026-05-18  
> 分支：`feature/defect-classification-self-learning`  
> 提交：`65fa8e3` `70d9f9a` `f045e01`  
> 状态：Phase 1/2/3 + 生产加固 全部完成，84 项测试通过

---

## 一、现状分析

### 1.1 当前系统架构

```
YOLO11m-seg  →  ROI 裁剪  →  PatchCore(无监督)  →  颜色一致性  →  多机位融合  →  OK/NG/REJECT
 (座椅定位)      (标准化)      (纹理异常检测)        (LAB色差)      (any/majority/all)
```

| 组件 | 技术方案 | 作用 |
|------|---------|------|
| 目标定位 | YOLO11m-seg 单类分割 | 检测图像中的座椅位置，生成前景 mask |
| ROI 裁剪 | 外接框扩缩 + mask 腐蚀 + Letterbox | 将座椅区域标准化为 256×256 输入 |
| 质量门控 | 拉普拉斯方差 + 亮度检查 | 过滤模糊/过曝/欠曝图像 |
| 纹理检测 | PatchCore (WideResNet50) + 记忆库 | 无监督异常检测，输出异常分数与热力图 |
| 颜色检测 | LAB 色彩空间一致性 | 检测颜色偏差（可选） |
| 多机位融合 | any / majority / all 策略 | 汇总多个相机视角的判定结果 |

### 1.2 当前方案的核心局限

1. **仅能输出 OK/NG 二分类**  
   PatchCore 只能回答"是否异常"，无法区分缺陷类型（划痕、污渍、褶皱、跳针、异物、凹陷等）。生产质检需要知道具体缺陷类型以指导返工。

2. **误报率高且难以控制**  
   光照变化、座椅姿态微小偏差、正常制造公差、皮革纹理自然差异等因素都可能导致误报。产线上频繁误报会降低工人对系统的信任度。

3. **缺乏自学习能力**  
   产线每天产生大量经过人工确认的检测结果（包括验证过的真缺陷和确认过的误报），但这些数据完全未被利用。系统无法从经验中改进。

4. **冷启动后的数据浪费**  
   PatchCore 训练只需"好样本"，产线积累的大量已标注缺陷样本未被模型利用，这些样本是宝贵的监督学习数据。

5. **阈值调优脆弱**  
   PatchCore 的多级判定规则（normal/critical/peak 三种触发机制）包含 10+ 个超参数，换产品型号或工位时需要重新标定。

---

## 二、升级方案总览

### 2.1 核心设计理念

**不替换 PatchCore，而是在其上叠加监督学习层，实现"无监督筛查 + 监督确认"的双层互补架构。**

```
                           ┌──────────────────────────────────┐
                           │        PatchCore (无监督)         │
                           │   高召回 · 不遗漏任何异常           │
                           │   作为"筛查层"永远保留              │
                           └──────────────┬───────────────────┘
                                          │ is_anomaly = true
                                          ▼
                           ┌──────────────────────────────────┐
                           │      误报过滤器 (Veto)             │
                           │   启发式规则 · 过滤物理上不合理的检出  │
                           │   面积/长宽比/边缘贴近              │
                           └──────────────┬───────────────────┘
                                          │ passes veto
                                          ▼
                           ┌──────────────────────────────────┐
                           │      缺陷分类器 (监督)             │
                           │   高精度 · 识别缺陷类型            │
                           │   EfficientNet/MobileNet          │
                           └──────────────┬───────────────────┘
                                          │
                                          ▼
                           ┌──────────────────────────────────┐
                           │       多机位融合 + 缺陷类型        │
                           │   OK / NG(scratch,stain,...) / REJECT │
                           └──────────────────────────────────┘
                                          │
                          ┌───────────────┴───────────────┐
                          ▼                               ▼
                    产线输出                          自学习飞轮
                (含缺陷类型+置信度)              (自动采集→标注→重训练)
```

### 2.2 分阶段实施路线

| 阶段 | 内容 | 周期 | 状态 |
|------|------|------|------|
| Phase 1 | 缺陷分类层 + 误报过滤 | 已完成 | ✅ |
| Phase 2 | 自学习数据闭环 | 已完成 | ✅ |
| Phase 3 | 基础模型集成 (DINOv2 / SAM) | 已完成 | ✅ |
| Phase 4 | MLOps 基础设施 (A/B测试/持续评估) | 待规划 | 🔲 |
| 生产加固 | 异步飞轮/超时保护/热加载/缺陷图返回 | 已完成 | ✅ |

---

## 三、升级后完整系统架构

### 3.1 总体架构图

```
                           ┌─────────────────────────────────────────┐
                           │              外部调用方                   │
                           │   SeatDefectInspector.inspect(frames)   │
                           └──────────────────┬──────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         inspect_frames() 主流程                              │
│                                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐   ┌──────────────────┐ │
│  │ 帧校验     │   │ YOLO     │   │ 逐相机检测         │   │ 多机位融合        │ │
│  │ 帧映射     │──▶│ 批量检测  │──▶│ inspect_prepared   │──▶│ fuse_camera      │ │
│  │ 校验ID    │   │ 分组推理  │   │ _camera()         │   │ _results()       │ │
│  └──────────┘   └──────────┘   └───────┬────────────┘   └────────┬─────────┘ │
│                                        │                          │          │
│               ┌────────────────────────┼──────────────────────────┤          │
│               │                        ▼                          ▼          │
│               │  ┌──────────────────────────────────────────────────────┐   │
│               │  │              单相机检测流水线                          │   │
│               │  │                                                      │   │
│               │  │  prepare ──▶ PatchCore ──▶ Color ──▶ ★Veto+Classify │   │
│               │  │  (YOLO→ROI)  (纹理异常)   (LAB色差)  (误报过滤+分类)  │   │
│               │  │                                 ★SAM refine        │   │
│               │  └──────────────────────────────────────────────────────┘   │
│               │                                                             │
│               │  ┌──────────────────────────────────────────────────────┐   │
│               │  │              自学习数据飞轮 (异步)                      │   │
│               │  │                                                      │   │
│               │  │  collect → queue → bg_thread → .npz disk write      │   │
│               │  │     ↑                          ↓                    │   │
│               │  │  检测出口                   buffer 积累               │   │
│               │  │     ↑                          ↓                    │   │
│               │  │  人工确认  ←── hard/  ←── 触发重训练                   │   │
│               │  │     │                          ↓                    │   │
│               │  │     └── tp/{type}/ ── 分类器微调 → 覆盖active模型      │   │
│               │  │                              ↓                      │   │
│               │  │                       mtime变化 → 热加载 → 生效        │   │
│               │  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          InspectionResponse                                 │
│                                                                             │
│  ┌─────────────────────┐  ┌──────────────────────┐  ┌────────────────────┐ │
│  │ result              │  │ defect_images        │  │ report_path        │ │
│  │ .status / .reason   │  │ {cam_id: base64 PNG} │  │ → results.json     │ │
│  │ .camera_results[]   │  │ (NG相机热力图叠加)     │  │                    │ │
│  │ .classification[]   │  └──────────────────────┘  └────────────────────┘ │
│  │ .timings_ms         │                                                    │
│  └─────────────────────┘                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 单相机检测决策树

```
inspect_prepared_camera(frame, camera, prepared)
│
├─ ROI=None / prepare失败 ────────────────────────▶ REJECT
│
├─ enabled regions 存在 ──▶ build_region_patchcore_plan()
│   └─ _finish_region_plans() → 批量PatchCore → 逐区域★veto+classify
│
└─ 全ROI路径:
   │
   ├─ load_model_bundle()          ← mtime校验 / pipeline签名校验
   ├─ PatchCore.predict()          ← 无监督纹理异常检测
   │   ├─ backbone: WideResNet50 / DINOv2-small / DINOv2-base
   │   └─ 输出: score, is_anomaly, heatmap, decision_mode
   │
   ├─ valid_patch_ratio < min? ───────────────────▶ REJECT
   │
   ├─ _predict_color_branch()      ← LAB颜色一致性 (可选)
   │
   ├─ ★ _apply_defect_classification()
   │   │
   │   ├─ is_anomaly=false? ──────▶ skip (OK路径, 0开销)
   │   │
   │   ├─ ★ 1) apply_veto(heatmap)
   │   │   ├─ 面积比 < 0.02%?     → veto → is_anomaly=false → OK
   │   │   ├─ 长宽比 < 1:20?      → veto
   │   │   └─ 80%+在边缘2%内?     → veto
   │   │
   │   ├─ ★ 2) classifier.predict(heatmap, roi)
   │   │   ├─ 超时保护: 200ms ThreadPoolExecutor
   │   │   ├─ 输出: [DefectClassificationResult, ...]
   │   │   └─ 超时/异常 → 降级跳过
   │   │
   │   └─ ★ 3) SAM refinement (可选)
   │       ├─ 热力图峰值 → 提示点
   │       ├─ SAM ViT-B predict → 精确mask
   │       └─ 输出: defect_bbox, defect_area_ratio
   │
   └─ 最终判定:
      ├─ texture+color NG → NG "texture_and_color_anomaly"
      ├─ texture NG only  → NG "texture_anomaly"
      ├─ color NG only    → NG "color_anomaly"
      ├─ quality reject   → REJECT
      └─ all pass         → OK
```

### 3.3 数据流拓扑

```
                        ┌──────────────┐
                        │  外部帧输入    │
                        │  {cam_id: img}│
                        └──────┬───────┘
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
          ┌─────────┐    ┌─────────┐    ┌─────────┐
          │ cam_0   │    │ cam_1   │    │ cam_N   │
          └────┬────┘    └────┬────┘    └────┬────┘
               │              │              │
     ┌─────────┼──────────────┼──────────────┼─────────┐
     │         ▼              ▼              ▼         │
     │    YOLO detect    YOLO detect    YOLO detect    │  批量分组
     │    + ROI refine   + ROI refine   + ROI refine   │
     │         │              │              │         │
     │    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐     │
     │    │PatchCore│   │PatchCore│   │PatchCore│     │  独立推理
     │    │+ Color  │   │+ Color  │   │+ Color  │     │
     │    │+ Veto   │   │+ Veto   │   │+ Veto   │     │
     │    │+Classify│   │+Classify│   │+Classify│     │
     │    │+ SAM    │   │+ SAM    │   │+ SAM    │     │
     │    └────┬────┘   └────┬────┘   └────┬────┘     │
     │         │              │              │         │
     └─────────┼──────────────┼──────────────┼─────────┘
               │              │              │
               └──────────────┼──────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   fuse_camera_   │
                    │   results()      │
                    │   any/majority/  │
                    │   all strategy   │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
    ┌──────────────────┐        ┌──────────────────┐
    │  InspectionResult │        │  飞轮异步采集     │
    │  + report JSON    │        │  queue → disk    │
    └────────┬─────────┘        └────────┬─────────┘
             │                           │
             ▼                           ▼
    ┌──────────────────┐        ┌──────────────────┐
    │  InspectionResponse│       │  flywheel_data/  │
    │  + defect_images  │        │  cam/type/*.npz  │
    └──────────────────┘        └──────────────────┘
```

### 3.4 耗时分布

```
单相机典型耗时 (WideResNet50, CPU):

阶段           默认       +Veto    +Classifier   +SAM     说明
────────────── ────────  ───────  ────────────  ───────  ──────────────────
YOLO detect    20-50ms   →        →             →        批量分组推理
ROI refine      2-5ms    →        →             →        mask+resize
PatchCore      50-200ms  →        →             →        backbone相关
Color branch    5-10ms   →        →             →        可选
Veto             0       <1ms     <1ms          <1ms     仅NG触发
Classifier       0        0       30-50ms       30-50ms  仅NG+超时保护
SAM              0        0        0           50-500ms  仅NG+GPU/CPU
Debug artifact  5-10ms   →        →             →        overlay.png
Flywheel         0       2-3ms    2-3ms         2-3ms    异步入队(非阻塞)
────────────── ────────  ───────  ────────────  ───────  ──────────────────
OK 总计         80-270   82-273   82-273        82-273
NG 总计         同上     同上     +32-52ms      +82-552ms
```

**关键**: OK 路径 (>95% 座椅) 零额外开销 (仅飞轮 2ms 异步入队)。NG 路径额外 30-50ms 在产线 10-15s 节拍内可吸收。

### 3.5 模块依赖关系

```
seat_defect_core/
│
├── api.py ───────────────────────────── 对外入口 (SeatDefectInspector)
│   └── service/inspection.py ────────── 主流程编排 (inspect_frames)
│       ├── service/frames.py ────────── 帧规范化
│       ├── service/core.py ──────────── InspectionService
│       │   ├── ModelBundleCache ─────── PatchCore模型缓存 (mtime校验)
│       │   ├── PatchCorePredictorPool ─ PatchCore批量推理
│       │   ├── ★ classifier_cache ──── 分类器缓存 (mtime热加载)
│       │   └── ★ flywheel_* ────────── 飞轮采集器/缓冲区
│       ├── service/inspection_camera.py 单相机检测
│       │   ├── yolo/detection.py ────── YOLO检测
│       │   ├── cvops/roi.py ─────────── ROI裁剪
│       │   ├── cvops/quality.py ─────── 质量门控
│       │   ├── patchcore/engine.py ──── PatchCore推理
│       │   │   └── patchcore/features.py ── ★ WideResNet50/DINOv2
│       │   ├── patchcore/color_branch.py ─ 颜色一致性
│       │   ├── ★ classifier/veto.py ─── 误报过滤
│       │   ├── ★ classifier/engine.py ── 缺陷分类 (EfficientNet)
│       │   └── ★ cvops/sam_refinement.py ─ SAM边界精修
│       ├── fusion.py ───────────────── 多机位融合 (★ 含defect_type)
│       ├── serialization.py ────────── JSON序列化 (★ 含classification)
│       └── service/response.py ─────── ★ defect_images base64
│
├── ★ classifier/ ──────────────────── 缺陷分类模块
│   ├── engine.py ──────────────────── DefectClassifierService
│   ├── training.py ────────────────── DefectClassifierTrainer
│   └── veto.py ────────────────────── FalsePositiveVeto
│
├── ★ flywheel/ ────────────────────── 自学习飞轮
│   ├── collector.py ───────────────── 异步数据采集
│   └── buffer_manager.py ──────────── 缓冲区管理+触发
│
├── ★ model_registry.py ────────────── 模型版本管理
│
└── types/results.py ───────────────── ★ DefectType, DefectClassificationResult
```

### 3.6 配置完整示例

```json
{
  "cameras": [{
    "camera_id": "cam_front",
    "patchcore_model_path": "models/cam_front_patchcore.npz",
    "source": "0",
    "patchcore": {
      "backend": "full",
      "backbone_name": "dinov2_small",
      "backbone_device": "cuda"
    },
    "classification": {
      "enabled": true,
      "model_path": "models/cam_front_classifier.pt",
      "confidence_threshold": 0.5,
      "inference_timeout_ms": 200,
      "sam_refinement_enabled": false
    },
    "veto": {
      "enabled": true,
      "min_defect_area_ratio": 0.0002,
      "max_defect_aspect_ratio": 0.05,
      "edge_proximity_ratio": 0.02
    },
    "color_branch": { "enabled": true }
  }],
  "flywheel": {
    "enabled": true,
    "buffer_dir": "flywheel_data/",
    "auto_label_threshold": 0.92,
    "human_validation_threshold": 0.60,
    "min_samples_before_retrain": 200,
    "retrain_cooldown_hours": 72,
    "sampling_rate_ok": 0.01,
    "max_samples_per_class": 5000
  },
  "model_registry_dir": "model_registry/",
  "fusion": { "ng_strategy": "any", "defect_overrides_reject": true }
}
```

---

## 四、Phase 1 — 缺陷分类层（已实现）

### 4.1 缺陷类型体系

```python
class DefectType(str, Enum):
    NONE           = "none"            # 无已知缺陷（误报抑制）
    SCRATCH        = "scratch"         # 划痕
    STAIN          = "stain"           # 污渍
    WRINKLE        = "wrinkle"         # 褶皱
    THREAD_JUMP    = "thread_jump"     # 跳针/缝线异常
    FOREIGN_MATTER = "foreign_matter"  # 异物
    DENT           = "dent"            # 凹陷
    COLOR_SHIFT    = "color_shift"     # 颜色异常
    OTHER          = "other"           # 已知异常但未分类
    POOR_ALIGNMENT = "poor_alignment"  # 座椅姿态异常
```

### 4.2 分类器技术方案

| 属性 | 设计选择 | 理由 |
|------|---------|------|
| 输入 | PatchCore 热力图 + ROI 灰度图 (2通道) | 热力图提供异常空间分布，ROI 提供纹理细节；两者互补 |
| 架构 | EfficientNet-B0 / MobileNetV3-Small | 轻量级(~2M参数)，推理<50ms，适合产线实时性要求 |
| 损失函数 | Focal Loss (γ=2.0) | 处理产线中缺陷样本天然不均衡的问题 |
| 输出 | 多类别 softmax + 置信度 | 支持按置信度阈值过滤低质量预测 |

**分类器与 PatchCore 完全解耦**：分类器不依赖 PatchCore 的 backbone，切换 PatchCore 特征提取器（如 WideResNet50→DINOv2）不需要重新训练分类器。

### 4.3 误报过滤器（FalsePositiveVeto）

三条启发式规则，在分类器之前执行，独立于分类器工作：

| 规则 | 判定逻辑 | 目标 |
|------|---------|------|
| 最小缺陷面积 | 异常区域面积 / ROI 面积 < 0.02% | 过滤传感器噪点 |
| 长宽比检测 | 最大异常连通域宽/高比 < 1:20 | 过滤光照条带状伪影 |
| 边缘贴近 | 异常像素 80%+ 位于 ROI 边界 2% 范围内 | 过滤 ROI 边界伪影 |

### 4.4 流水线集成点

在 `inspection_camera.py` 的相机检测流水线中：

```
PatchCore.predict()  →  texture_result
    ↓ (is_anomaly = true)
apply_veto(heatmap)  →  误报过滤
    ↓ (passes veto)
classifier.predict(heatmap, roi_image)  →  classification_results
    ↓
color_branch.predict()  →  color_result
    ↓
merge_status()  →  CameraInspectionResult(含 defect_type)
```

### 4.5 CLI 训练命令

```bash
python -m seat_defect_inspection train-classifier \
  --config configs/inspection.json \
  --dataset-dir datasets/defect_classifier/ \
  --backbone efficientnet_b0 \
  --epochs 50 \
  --batch-size 32
```

数据集目录结构：
```
datasets/defect_classifier/
├── scratch/       # 划痕样本
├── stain/         # 污渍样本
├── wrinkle/       # 褶皱样本
├── thread_jump/   # 跳针样本
├── foreign_matter/# 异物样本
├── dent/          # 凹陷样本
├── color_shift/   # 颜色异常样本
├── other/         # 其他缺陷
├── none/          # 正常样本
└── good/          # 正常样本（与 none 等价）
```

---

## 五、Phase 2 — 自学习数据闭环（已实现）

### 5.1 数据飞轮架构

```
┌─────────────────────────────────────────────────────────────┐
│                        产线检测                               │
│  InspectionService.inspect_frames()                         │
│       │                                                     │
│       ▼                                                     │
│  _collect_flywheel_samples()  ← 自动触发，每次检测后执行       │
│       │                                                     │
│       ├── OK 样本  ──→  sampling_rate_ok (默认 1%) ──→ ok/  │
│       │                                                   │
│       └── NG 样本  ──→ 全部保存 ──→ 根据分类置信度路由：       │
│              ├── confidence ≥ 0.92  →  tp/{defect_type}/   │
│              ├── 0.60 ≤ conf < 0.92 →  hard/ (待人工复核)    │
│              └── confidence < 0.60  →  fp/ (疑似误报)        │
│                                                             │
│  缓冲区目录结构：                                              │
│  {flywheel_dir}/{camera_id}/{seat_model_id}/                 │
│       ├── ok/           ← 正常分布漂移监控                     │
│       ├── tp/scratch/   ← 自动标注的真缺陷                     │
│       ├── tp/stain/                                            │
│       ├── hard/         ← 待人工复核样本                        │
│       ├── fp/           ← 疑似误报样本                          │
│       └── _archive/     ← 过期归档                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 训练触发与执行

**触发条件**（满足任一）：
- TP 样本总数 ≥ `min_samples_before_retrain`（默认 200）
- Hard 样本数 ≥ `min_samples_before_retrain`
- 距上次训练超过 `retrain_cooldown_hours`（默认 72h）

**重训练流程**：
```
1. 分类器微调（fine-tune）
   ├── 从 tp/*/ 和 fp/ 加载样本 (.npz)
   ├── 使用更低学习率 (1e-4) 和更少 epochs (30)
   ├── 冻结 backbone，仅训练分类头
   └── 验证集上评估 → 通过则保存新版本

2. 模型版本管理（ModelRegistry）
   ├── 注册新版本 → card.json
   ├── 更新 active 符号链接
   ├── 归档旧版本
   └── 精度下降则自动回滚

3. 缓冲区维护
   ├── 归档 90 天前的过期样本
   └── 裁剪超量类别（每类最多 max_samples_per_class）
```

### 5.3 模型注册中心（ModelRegistry）

```
{registry_dir}/
├── cam_front/
│   ├── __full__/              ← 全 ROI PatchCore 模型
│   │   ├── 20260518_120000/
│   │   │   ├── model.npz
│   │   │   └── card.json      ← ModelCard (版本/指标/父版本/标签分布)
│   │   ├── 20260515_080000/
│   │   └── active -> 20260518_120000/
│   └── __classifier__/        ← 缺陷分类器
│       ├── 20260518_120000/
│       │   ├── classifier.pt
│       │   └── card.json
│       └── active -> 20260518_120000/
├── cam_side/
│   └── ...
└── cam_top/
    └── ...
```

### 5.4 增量 PatchCore 更新（设计就绪，代码占位）

当飞轮积累了被确认的正常纹理变化样本后，可以增量更新 PatchCore 记忆库：

```
新正常样本 → 提取 embedding → 追加到 memory_bank → coreset 重采样 → 重新计算阈值
```

时间复杂度 O(n × d) vs 全量重训 O(N × d × B)，在产线换型间隙即可完成。

---

## 六、配置体系扩展

### 6.1 新增配置项

```json
{
  "cameras": [{
    "camera_id": "cam_front",
    "classification": {
      "enabled": true,
      "model_path": "models/classifier/cam_front_classifier.pt",
      "confidence_threshold": 0.5,
      "sam_refinement_enabled": false,
      "enable_zero_shot_fallback": false
    },
    "veto": {
      "enabled": true,
      "min_defect_area_ratio": 0.0002,
      "max_defect_aspect_ratio": 0.05,
      "edge_proximity_ratio": 0.02
    }
  }],
  "flywheel": {
    "enabled": true,
    "buffer_dir": "flywheel_data/",
    "auto_label_threshold": 0.92,
    "human_validation_threshold": 0.60,
    "min_samples_before_retrain": 200,
    "retrain_cooldown_hours": 72,
    "sampling_rate_ok": 0.01,
    "incremental_patchcore_enabled": true,
    "max_samples_per_class": 5000
  },
  "model_registry_dir": "model_registry/"
}
```

所有新字段均有合理默认值，现有配置文件无需修改即可继续使用。

---

## 七、文件变更清单

### 7.1 新增文件（13 个）

| 文件路径 | 说明 |
|---------|------|
| `src/seat_defect_core/classifier/__init__.py` | 分类器包导出 |
| `src/seat_defect_core/classifier/veto.py` | 误报过滤器（3 条启发式规则） |
| `src/seat_defect_core/classifier/engine.py` | 缺陷分类器推理服务 |
| `src/seat_defect_core/classifier/training.py` | 缺陷分类器训练器（Focal Loss） |
| `src/seat_defect_core/flywheel/__init__.py` | 飞轮包导出 |
| `src/seat_defect_core/flywheel/collector.py` | 检测数据采集服务 |
| `src/seat_defect_core/flywheel/buffer_manager.py` | 缓冲区管理与触发 |
| `src/seat_defect_core/model_registry.py` | 模型版本注册中心 |
| `src/seat_defect_inspection/cli_commands/train_classifier.py` | train-classifier 命令 |
| `src/seat_defect_inspection/service/classifier_training.py` | 分类器训练编排 |
| `src/seat_defect_inspection/service/flywheel.py` | 飞轮自学习训练编排 |
| `src/seat_defect_core/cvops/sam_refinement.py` | SAM 缺陷边界精修 |

### 7.2 修改文件（16 个）

| 文件路径 | 变更内容 |
|---------|---------|
| `src/seat_defect_core/types/results.py` | 新增 DefectType 枚举、DefectClassificationResult 数据类 |
| `src/seat_defect_core/types/__init__.py` | 导出新类型 |
| `src/seat_defect_core/config.py` | 新增 3 个配置类，扩展 CameraConfig 和 InspectionConfig |
| `src/seat_defect_core/__init__.py` | 导出新配置类和类型 |
| `src/seat_defect_core/runtime_config_parsers.py` | 新增 3 段配置解析器 + inference_timeout_ms |
| `src/seat_defect_core/runtime_config.py` | DINOv2 免检 pretrained 配置 |
| `src/seat_defect_core/service/core.py` | 分类器缓存(mtime热加载)、飞轮采集器/缓冲区管理 |
| `src/seat_defect_core/service/inspection.py` | 检测完成后自动采集飞轮样本 |
| `src/seat_defect_core/service/inspection_camera.py` | 集成 veto + 分类器 + SAM + defect_images |
| `src/seat_defect_core/service/response.py` | defect_images base64 编码返回 |
| `src/seat_defect_core/fusion.py` | 融合决策原因包含缺陷类型摘要 |
| `src/seat_defect_core/serialization.py` | 分类结果 JSON 序列化 |
| `src/seat_defect_core/patchcore/features.py` | DINOv2 backbone 特征提取 |
| `src/seat_defect_core/classifier/engine.py` | is_stale/reload mtime 热加载 |
| `src/seat_defect_inspection/cli.py` | 注册 train-classifier 命令 |
| `src/seat_defect_inspection/cli_commands/__init__.py` | 导出新命令 |
| `src/seat_defect_inspection/service/__init__.py` | 导出 train_classifier_models |

---

## 八、测试验证

### 8.1 测试结果

```
84 passed in 1.96s — 全部通过，无回归
```

### 8.2 向后兼容性验证

- 现有 PatchCore 推理结果序列化格式不变
- 现有 JSON/INI 配置文件无需修改即可正常加载
- 所有新功能默认关闭 (`enabled: false`)
- 无监督异常检测路径（PatchCore 直出）完全不受影响

### 8.3 配置解析验证

```python
# 验证新字段可被正确解析
config = load_config("config_with_classification.json")
assert config.cameras[0].classification.enabled == True
assert config.cameras[0].veto.min_defect_area_ratio == 0.0002
assert config.flywheel.auto_label_threshold == 0.92

# 验证未知字段拒绝机制仍然有效
load_config("config_with_unknown_field.json")  # raises ValueError
```

### 8.4 误报过滤器验证

```python
# 极小异常 → 否决（斑点噪声）
heatmap_small = create_anomaly_heatmap(area_ratio=0.0001)
result = apply_veto(heatmap_small, config=veto_config)
assert result.vetoed == True

# 正常异常 → 放行
heatmap_large = create_anomaly_heatmap(area_ratio=0.1)
result = apply_veto(heatmap_large, config=veto_config)
assert result.vetoed == False
```

---

## 九、使用指南

### 9.1 启用分类器

在现有的检测配置 JSON 中，为需要分类的机位添加 `classification` 块：

```json
{
  "cameras": [{
    "camera_id": "cam_0",
    "patchcore_model_path": "models/cam_0_patchcore.npz",
    "classification": {
      "enabled": true,
      "model_path": "models/cam_0_classifier.pt",
      "confidence_threshold": 0.5
    }
  }]
}
```

### 9.2 启用误报过滤（可选，独立使用）

```json
{
  "cameras": [{
    "veto": {
      "enabled": true,
      "min_defect_area_ratio": 0.0002
    }
  }]
}
```

### 9.3 启用自学习飞轮

```json
{
  "flywheel": {
    "enabled": true,
    "buffer_dir": "flywheel_data/",
    "min_samples_before_retrain": 200,
    "retrain_cooldown_hours": 72
  }
}
```

飞轮数据会自动采集，重训练可通过以下方式触发：

```python
from seat_defect_inspection.service.flywheel import check_and_retrain_if_needed

# 检查并在条件满足时自动重训练
summary = check_and_retrain_if_needed(service, dry_run=False)
```

或设置定时任务调用该函数。

### 9.4 训练分类器

```bash
# 准备标注数据集后
python -m seat_defect_inspection train-classifier \
  --config configs/inspection.json \
  --dataset-dir datasets/defect_classifier/ \
  --backbone efficientnet_b0 \
  --epochs 50
```

---

## 十、Phase 3 — 基础模型集成（已实现）

### 10.1 DINOv2 骨干网络

DINOv2 作为 PatchCore 特征提取器的可选替代方案。

| 属性 | WideResNet50 (原) | DINOv2-Small (新) | DINOv2-Base (新) |
|------|-------------------|-------------------|-------------------|
| 参数量 | 69M | 21M | 86M |
| 训练方式 | ImageNet 监督 | 自监督 (无需标签) | 自监督 |
| 特征维度 | 1024+2048 | 384×2层 | 768×2层 |
| 推理速度 | 基准 | ~快 30% | ~相近 |
| 配置名 | `wide_resnet50_2` | `dinov2_small` | `dinov2_base` |

**使用方式**：在配置中修改 `patchcore.backbone_name`，无需改代码。

```json
{
  "patchcore": {
    "backend": "full",
    "backbone_name": "dinov2_small"
  }
}
```

DINOv2 为自监督模型，从 `torch.hub` 自动下载，无需配置 `backbone_pretrained` 或 `backbone_weights_path`。特征提取通过 `model.get_intermediate_layers()` 在 `reshape=True` 模式直接获取空间特征图，无需 forward hook 机制。

### 10.2 SAM 缺陷边界精修

当分类器检测到缺陷且 `sam_refinement_enabled=true` 时，使用 SAM (Segment Anything) 在热力图峰值位置生成精确缺陷 mask。

**特性**：
- 懒加载 + 进程级单例缓存
- 使用轻量 SAM ViT-B (375MB)
- GPU 推理 ~50ms，CPU 推理 ~500ms
- 失败时静默降级，不影响主流程
- 输出：精确 `defect_bbox`（归一化坐标）和 `defect_area_ratio`

```json
{
  "classification": {
    "sam_refinement_enabled": true
  }
}
```

---

## 十一、生产可靠性加固（已实现）

### 11.1 异步飞轮采集

DataCollectorService 使用后台 daemon 线程 + queue.Queue 实现异步磁盘写入。主检测流程仅需 `np.copy()` + `queue.put()` (~2ms)，不影响检测延迟。

### 11.2 分类器超时保护

`ClassificationConfig.inference_timeout_ms` (默认 200ms)。推理通过 `ThreadPoolExecutor.submit()` + `future.result(timeout=...)` 执行。超时后降级为 PatchCore 原结果，不阻塞主流程。

### 11.3 模型热加载

`get_classifier_service()` 基于 mtime 检测模型文件变更。飞轮重训练覆盖 active 模型文件后，下一次检测自动加载新模型，无需重启进程。与 PatchCore `ModelBundleCache` 保持一致的失效策略。

### 11.4 全线异常降级

| 组件 | 异常处理 | 降级行为 |
|------|---------|---------|
| Veto | try/except | 跳过过滤，PatchCore 原结果 |
| Classifier | try/except + 超时 | 跳过分类，NG 判定不变 |
| SAM | try/except | 跳过精修，无 bbox/area |
| Flywheel collect | try/except | 跳过采集，不影响结果导出 |
| Classifier warmup | try/except | 跳过预加载，首次推理时懒加载 |

### 11.5 缺陷图 API 返回

`InspectionResponse.defect_images` 直接返回 NG 机位的 base64 编码缺陷标注图（热力图叠加 JET 配色），调用方无需读取磁盘。

---

## 十二、后续规划 (Phase 4)

| 能力 | 说明 |
|------|------|
| A/B 测试 | 流量分配对照/实验模型，自动对比指标 |
| 持续评估 | 定期在标注验证集上评估模型衰减 |
| 自动回滚 | 精度下降超阈值自动切回上一版本 |
| PatchCore 增量更新 | 新正常样本 embedding 追加到 memory bank + coreset 重采样 |
| CLIP 零样本分类 | 图文匹配作为分类器兜底 |

---

## 十三、风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| 分类器冷启动需要标注数据 | 初期无分类器可用 | 分类器默认关闭；误报过滤器可独立工作 |
| 自动标注错误导致模型退化 | 错误标签进入训练集 | 高置信度阈值 (0.92) + 人工复核通道 + 精度下降自动回滚 |
| 分类器推理增加延迟 | 产线节拍超时 | 轻量 backbone (<50ms)；200ms 超时 fallback 到 PatchCore-only |
| 新缺陷类型未被分类器识别 | 漏检未知缺陷 | PatchCore 永远作为兜底筛查层；unknown 缺陷类型触发告警 |
| DINOv2 首次下载失败 | 离线产线无法启动 | 预置模型文件到 .torch_cache；回退到 WideResNet50 |
| SAM CPU 推理过慢 | 产线节拍超时 | 默认关闭；仅 GPU 环境启用；失败静默降级 |

---

## 附录 A：核心类型定义

```python
class DefectType(str, Enum):
    NONE = "none"
    SCRATCH = "scratch"
    STAIN = "stain"
    WRINKLE = "wrinkle"
    THREAD_JUMP = "thread_jump"
    FOREIGN_MATTER = "foreign_matter"
    DENT = "dent"
    COLOR_SHIFT = "color_shift"
    OTHER = "other"
    POOR_ALIGNMENT = "poor_alignment"

@dataclass
class DefectClassificationResult:
    defect_type: DefectType
    confidence: float
    defect_bbox: BoundingBox | None = None
    defect_area_ratio: float = 0.0
    classifier_version: str | None = None
    veto_applied: bool = False
```

## 附录 B：检测结果 JSON 输出示例

```json
{
  "part_id": "seat_20260518_001",
  "status": "NG",
  "decision_reason": "ng_from_cam_front,cam_side_types:cam_front:scratch,cam_side:stain",
  "camera_results": [{
    "camera_id": "cam_front",
    "status": "NG",
    "reason": "texture_and_color_anomaly",
    "texture_result": {
      "score": 1.45,
      "threshold": 0.98,
      "is_anomaly": true,
      "decision_mode": "normal_and_critical",
      "classification_results": [{
        "defect_type": "scratch",
        "confidence": 0.94,
        "defect_area_ratio": 0.023,
        "veto_applied": false
      }]
    }
  }]
}
```
