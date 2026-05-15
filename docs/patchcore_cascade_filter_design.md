# PatchCore 级联过滤方案设计文档

> 版本: v1.0 | 日期: 2026-05-14 | 状态: 方案评估

---

## 1. 背景与动机

### 1.1 当前 Pipeline 的局限

当前 `seat_defect_core` 的缺陷检测完全依赖 PatchCore 无监督异常检测模型。其判定逻辑位于 `patchcore/scoring.py:_decide_patchcore_anomaly`，通过三条规则（normal_rule / critical_rule / peak_rule）对 patch 级异常热力图做空间统计后给出最终 OK/NG 判定。

该架构的优点是不需要标注缺陷样本即可上线。但在实际座椅面料检测中面临三个问题：

| 问题 | 表现 | 根因 |
|------|------|------|
| **误报偏高** | 面料纹理褶皱、缝线、光照渐变常被判定为 NG | PatchCore 只度量"与正常分布的偏差"，无法区分缺陷和正常纹理变异 |
| **阈值难以单点最优** | 调高阈值漏检小缺陷，调低阈值误报增加 | 单阶段判定，没有独立的召回/精度调节点 |
| **规则不透明** | `decision_score_margin` / `strong_patch_ratio` 等参数调整缺乏直观的业务含义 | 特征空间距离 → 空间统计的映射是非线性的 |

### 1.2 已知有利条件

用户确认已有 **100+ 标注缺陷样本**。这为训练一个监督型分类器提供了基础数据量。

---

## 2. 方案概述

### 2.1 核心思路

将当前的单阶段 PatchCore 判定拆分为**两阶段级联**：

```
Stage 0 (不变)   YOLO 检测 → ROI 精修 → Quality Guard → PatchCore（宽松阈值）
Stage 1 (新增)   候选区域提取 → 缺陷分类器（过滤误报）
Stage 2 (新增)   传统斑点特征卡控（确定性规则，可选聚合层）
```

不推荐三阶段独立串联，而是将传统斑点分析的特征作为**分类器的输入**和 Stage 2 的**可选安全网**。

### 2.2 级联数据流

```
PatchCore heatmap（宽松阈值）
    │
    ▼
候选区域提取（连通分量分析，contour-based）
    │  输出: List[CandidateRegion]  (box, contour, patch_score)
    │
    ▼
缺陷分类器（轻量 CNN）
    │  每个候选区域 → patch 图像 → 分类器 → real_defect / false_alarm
    │  输出: List[ClassifiedRegion]  (box, class, confidence, features)
    │
    ▼
传统斑点卡控（可选，确定性规则）
    │  面积 / 长宽比 / solidity / 对比度 等阈值
    │  输出: List[ConfirmedDefect]
    │
    ▼
判定聚合（替代当前 _decide_patchcore_anomaly）
    → 任一候选通过全部阶段 → NG
    → 无候选通过 → OK
```

### 2.3 与现有流程的对照

| 维度 | 当前 | 方案后 |
|------|------|--------|
| PatchCore 建模 | 正常样本 → memory bank + threshold | **不变** |
| PatchCore 推理阈值 | `decision_score_margin = 1.08` 等生产阈值 | **下调至 0.92-0.95**，追求 >98% 召回 |
| 误报控制 | PatchCore 内置 `normal_rule` 的空间统计 | 移至下游分类器 + 斑点规则 |
| 判定逻辑 | `_decide_patchcore_anomaly` 三条规则 | `CascadeDecisionEngine` 多级 AND/OR 可配置 |
| 模型文件 | 每个机位 1 个 `.npz` | 每个机位 1 个 `.npz` + 可选 1 个 `.pt`（分类器） |

---

## 3. 详细设计

### 3.1 Stage 1: 候选区域提取

**模块**: `seat_defect_core/cvops/candidate_extraction.py`

#### 3.1.1 输入与输出

```
输入:
  - heatmap: np.ndarray          # PatchCore 产出的异常热力图 (aligned_roi 尺寸)
  - roi_image: np.ndarray        # 原始 ROI 图像 (用于裁剪)
  - target_mask: np.ndarray      # 前景 mask
  - config: CandidateExtractionConfig

输出:
  - List[CandidateRegion]:
      box: BoundingBox           # ROI 坐标系下的外接矩形
      contour: np.ndarray        # 候选区域轮廓点
      heatmap_patch: np.ndarray  # 区域内 heatmap 值
      max_heatmap_score: float   # 区域内最高异常分数
      area_ratio: float          # 面积占 ROI 的比例
```

#### 3.1.2 提取算法

```python
def extract_candidates(heatmap, roi_image, target_mask, config) -> List[CandidateRegion]:
    # 1. 二值化: heatmap > decision_threshold * floor_ratio
    #    (宽松阈值，floor_ratio 默认 0.7，确保不漏)
    binary = heatmap > config.candidate_floor_ratio * decision_threshold

    # 2. 形态学闭运算: 连接邻近碎片 (kernel 3x3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 3. 连通分量分析 + 轮廓提取
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. 过滤: 面积 < min_area_pixels 或面积 > max_area_ratio * roi_area 的丢弃
    # 5. 按 max_heatmap_score 降序排列，取 topK
    # 6. 对每个候选，在原图 ROI 上裁出对应区域
```

#### 3.1.3 为什么用 contour 而非 patch grid

当前 `_analyze_patch_evidence` 在 **patch 坐标系**（如 8×8 grid）上做连通分析，分辨率受限于 patch_size/stride。候选区域提取直接在 pixel 级 heatmap 上操作，可以获得像素级轮廓，为分类器提供更精确的裁剪区域。

#### 3.1.4 复用现有能力

- `cv2.connectedComponentsWithStats` 和 `cv2.findContours` 的调用模式与当前 `_measure_patch_components` 一致
- heatmap 归一化逻辑复用 `normalize_map_against_threshold`
- ROI 切分逻辑复用 `cvops/regions.py` 中的 `build_region_roi_sample_from_box`

---

### 3.2 Stage 2: 缺陷分类器

**模块**: `seat_defect_core/classifier/`（新增包）

#### 3.2.1 模型选型

| 方案 | 参数量 | 推理延迟 (CPU) | 训练数据需求 | 推荐 |
|------|--------|---------------|-------------|------|
| MobileNetV3-Small | 2.5M | ~3ms/patch | 500+ | 标注充足时首选 |
| **EfficientNet-B0** | 5.3M | ~6ms/patch | 300+ | 精度更优 |
| MobileNetV2 (1.0) | 3.5M | ~4ms/patch | 300+ | 社区成熟度高 |
| 传统 HOG+LBP+SVM | 0 | ~2ms/patch | 100+ | **当前阶段首选** |

**当前阶段推荐 HOG+LBP+SVM**：100+ 标注样本对深度模型偏少（容易过拟合），但对手工特征 + SVM 是可行区间。后续标注积累到 500+ 时可切换为 MobileNetV3。

实现上应定义抽象接口 `DefectClassifier`，支持两种后端：

```python
class DefectClassifier(ABC):
    @abstractmethod
    def predict(self, image_patch: np.ndarray) -> ClassifierPrediction: ...

    @abstractmethod
    def predict_batch(self, patches: List[np.ndarray]) -> List[ClassifierPrediction]: ...
```

#### 3.2.2 分类器输入构造

分类器的输入 patch 需要统一尺寸（如 128×128），构造方式取决于候选区域形状：

- **方形候选**：直接 resize 至 128×128
- **非方形候选**：按外接矩形裁剪，letterbox 补零至 128×128，保持长宽比
- **极小候选**：如果候选区域 < 16×16（在 aligned ROI 坐标系中），放大至 128×128 后送入

同时支持**多通道输入**：
- `rgb`: 原始 RGB patch
- `rgb+heatmap`: RGB + heatmap 单通道拼接（4 通道），利用 PatchCore 的先验信息

#### 3.2.3 训练数据构造

```
训练正样本（real_defect）:
  - 来源: 标注的缺陷样本
  - 构造: 在缺陷区域取外接矩形 + 轻微随机扰动（±10% 位置偏移、±5% 尺度缩放）
  - 数量: 每张缺陷图取 1-3 个候选区域，总共 100-300 个正样本

训练负样本（false_alarm）:
  - 来源: 在正常样本上运行宽松 PatchCore → 所有候选区域标注为 false_alarm
  - 构造: 对正常样本做数据增强（亮度变化 ±15%、轻微旋转 ±5°、高斯模糊 σ=1）后再跑 PatchCore，增加多样性
  - 数量: 目标是正样本的 2-3 倍
```

#### 3.2.4 分类器配置模型

```python
@dataclass
class ClassifierFilterConfig:
    enabled: bool = False
    model_path: str | None = None               # 分类器模型文件 (.pkl for SVM, .pt for CNN)
    model_type: str = "hog_lbp_svm"             # "hog_lbp_svm" | "mobilenet_v3" | "efficientnet_b0"
    input_size: int = 128                        # 输入 patch 归一化尺寸
    input_channels: str = "rgb"                  # "rgb" | "rgb+heatmap"
    confidence_threshold: float = 0.5            # 分类置信度阈值
    device: str = "cpu"
    # 特征提取参数 (仅 hog_lbp_svm 模式)
    hog_orientations: int = 9
    hog_pixels_per_cell: int = 8
    hog_cells_per_block: int = 2
    lbp_radius: int = 1
    lbp_points: int = 8
```

该配置挂载到 `CameraConfig` 上：
```python
@dataclass
class CameraConfig:
    # ... 现有字段保持不变 ...
    classifier_filter: ClassifierFilterConfig = field(
        default_factory=ClassifierFilterConfig
    )
```

**向后兼容**：`classifier_filter.enabled = False` 时所有新增代码路径不执行，行为与当前完全一致。

---

### 3.3 Stage 2 (可选): 传统斑点特征卡控

**模块**: `seat_defect_core/cvops/blob_rules.py`

#### 3.3.1 定位

传统斑点卡控不是独立阶段，而是**判定聚合层**的一部分。它提供一套可配置的确定性规则，作用于分类器确认的候选区域（或直接作用于候选区域，如果分类器未启用）。

```
候选区域 → [分类器] → 斑点规则检查 → 最终判定
          (可选)      (可选但推荐)
```

#### 3.3.2 斑点特征

对每个候选区域的轮廓，计算以下特征：

| 特征 | 计算方式 | 业务含义 |
|------|---------|---------|
| `area_mm2` | contour 面积 × pixel_to_mm 比例 | 缺陷物理尺寸 |
| `extent` | contour 面积 / 外接矩形面积 | 区分点状 vs 线状缺陷 |
| `aspect_ratio` | 外接矩形宽高比 | 缝线偏长条形，污渍偏圆形 |
| `solidity` | contour 面积 / 凸包面积 | 缺陷边缘粗糙度 |
| `mean_heatmap` | 区域内 heatmap 均值 | PatchCore 异常程度 |
| `contrast` | 区域内原图对比度 (std) | 缺陷与背景的视觉差异 |
| `circularity` | 4π × area / perimeter² | 形状规则度 |

#### 3.3.3 规则配置

```python
@dataclass
class BlobRuleConfig:
    enabled: bool = True
    min_area_mm2: float = 0.5                  # 最小缺陷面积 (mm²)
    max_area_mm2: float = 500.0                # 最大缺陷面积 (mm²)
    min_solidity: float = 0.3                  # 最小 solidity
    min_contrast: float = 8.0                  # 最小对比度 (灰度 std)
    pixel_to_mm_ratio: float = 0.15            # 像素到毫米的转换比 (默认值需标定)
    # 高级规则（可选，默认关闭）
    max_aspect_ratio: float = 0.0              # 0 表示不限制
    min_extent: float = 0.0                    # 0 表示不限制
```

挂载方式：在 `CameraConfig` 上增加可选字段 `blob_rules: BlobRuleConfig | None = None`。

#### 3.3.4 规则组合模式

```
blob_rule_mode: "all"  → 所有规则必须满足
blob_rule_mode: "any"  → 任一规则满足即可（默认）
blob_rule_mode: "vote" → 满足半数以上规则
```

---

### 3.4 判定聚合层

**模块**: `seat_defect_core/cvops/cascade_decision.py`

替代当前的 `_decide_patchcore_anomaly`，新增 `CascadeDecisionEngine`：

```python
class CascadeDecisionEngine:
    """级联判定引擎：组合分类器和斑点规则做出最终 OK/NG 判定。"""

    def __init__(
        self,
        classifier: DefectClassifier | None,
        blob_rules: BlobRuleConfig | None,
        cascade_config: CascadeConfig,
    ):
        ...

    def decide(
        self,
        candidates: List[CandidateRegion],
        roi_image: np.ndarray,
        patchcore_result: TextureAnomalyResult,
    ) -> CascadeDecision:
        """
        返回:
          CascadeDecision:
            is_anomaly: bool
            reason: str
            confirmed_defects: List[ConfirmedDefect]
            stage_results: dict  # 每个阶段的明细
        """
```

#### 判定流程

```
for candidate in candidates:
    # Stage 1: 分类器（如果启用）
    if classifier is not None:
        pred = classifier.predict(candidate.image_patch)
        if pred.class != "real_defect" or pred.confidence < threshold:
            continue  # 跳过此候选
        candidate.classifier_result = pred

    # Stage 2: 斑点规则（如果启用）
    if blob_rules is not None:
        blob_features = extract_blob_features(candidate)
        if not check_blob_rules(blob_features, blob_rules):
            continue  # 跳过此候选
        candidate.blob_features = blob_features

    # 通过全部阶段 → 确认缺陷
    confirmed.append(ConfirmedDefect(candidate))

if len(confirmed) > 0:
    return CascadeDecision(is_anomaly=True, reason="cascade_confirmed", ...)
else:
    return CascadeDecision(is_anomaly=False, reason="cascade_all_filtered", ...)
```

#### 兜底逻辑：保留 PatchCore 原始判断作为 safety net

当分类器和斑点规则都开启时，存在一个风险：两者都漏掉的真实缺陷。需要保留 PatchCore 原始 `critical_rule` 作为**不可绕过的安全网**：

```python
# 如果 PatchCore 本身给出的是高置信度 critical 判定，直接放行
if patchcore_result.decision_mode in ("critical_rule", "normal_and_critical"):
    return CascadeDecision(is_anomaly=True, reason="patchcore_critical_override", ...)
```

---

### 3.5 与现有 Pipeline 的集成点

#### 3.5.1 修改 `inspection_camera.py:inspect_prepared_camera`

改动范围：在 PatchCore 推理完成后、判定结果构造之前，插入级联过滤。

```python
def inspect_prepared_camera(...):
    # ... 现有代码保持不变，执行 PatchCore 推理 ...
    texture_result = model_bundle.patchcore.predict(...)

    # === 新增：级联过滤 ===
    cascade_decision = None
    cascade_config = camera.classifier_filter  # 或其他挂载点
    if cascade_config and cascade_config.enabled:
        candidates = extract_candidates(
            texture_result.heatmap,
            prepared.roi.aligned_roi_image,
            prepared.roi.target_mask,
            cascade_config.candidate_extraction,
        )
        classifier = service.load_classifier(camera, seat_model_id)
        cascade_decision = service.cascade_engine.decide(
            candidates,
            prepared.roi.aligned_roi_image,
            texture_result,
        )
        # 用 cascade_decision 覆盖 texture_result.is_anomaly
        texture_result = texture_result_with_cascade_override(
            texture_result, cascade_decision
        )
    # === 新增结束 ===

    # ... 后续判定逻辑不变 ...
    if texture_result.is_anomaly and color_result is not None and ...:
        status = "NG"
    ...
```

#### 3.5.2 修改 `TextureAnomalyResult`

新增可选字段，承载级联过滤明细：

```python
@dataclass
class TextureAnomalyResult:
    # ... 现有字段不变 ...

    # 级联过滤结果（可选，向后兼容）
    cascade: CascadeDecision | None = None
    candidate_count: int = 0
    confirmed_defect_count: int = 0
```

#### 3.5.3 Region 模式的兼容

对于 Region 模式（`build_region_patchcore_plan`），每个 Region 的 `texture_result` 独立进入级联过滤。Region 级别的分类器可以与 Camera 级别共享（同一个模型文件），也可以独立配置。建议**第一阶段共享分类器**，降低管理和训练成本。

#### 3.5.4 多机位融合不变

`fusion.py:fuse_camera_results` 无需修改。单机位判定结果的状态（OK/NG/REJECT）语义不变，只是 NG 的原因从 `"texture_anomaly"` 变为 `"cascade_confirmed"` 或 `"patchcore_critical_override"`。

---

## 4. 训练管线

### 4.1 分类器训练命令

在 `seat_defect_inspection` 层新增 `train-classifier` 子命令：

```bash
python -m seat_defect_inspection train-classifier \
    --config configs/seat_defect_inspection.json \
    --defect-dir data/defects/ \
    --normal-dir data/normal/ \
    --output models/classifier/hog_lbp_svm.pkl \
    --model-type hog_lbp_svm
```

### 4.2 训练流程

```
Step 1: 对 normal-dir 中的正常样本运行完整 inspect pipeline
        → 获取宽松 PatchCore 推理结果 → 提取所有候选区域 → 标注为 false_alarm

Step 2: 对 defect-dir 中的缺陷样本运行完整 inspect pipeline
        → 获取宽松 PatchCore 推理结果
        → 提取候选区域 → 与标注框做 IoU 匹配
        → IoU > 0.3 的候选标注为 real_defect

Step 3: 特征提取 + SVM 训练 (或 CNN fine-tune)
        → 输出模型文件 + 验证指标

Step 4: 在留出验证集上评估
        → 报告: recall / precision / F1 / 误报减少率
```

### 4.3 模型文件格式

- **SVM**: `joblib` 格式 (`.pkl`)，包含特征提取器参数 + 分类器权重
- **CNN**: PyTorch `state_dict` (`.pt`)，包含模型定义类名

---

## 5. 配置完整示例

```json
{
  "seat_defect_inspection": {
    "cameras": [
      {
        "camera_id": "cam_0",
        "patchcore_model_path": "models/cam_0_patchcore.npz",
        "classifier_filter": {
          "enabled": true,
          "model_path": "models/classifier_svm.pkl",
          "model_type": "hog_lbp_svm",
          "input_size": 128,
          "input_channels": "rgb",
          "confidence_threshold": 0.6,
          "candidate_extraction": {
            "candidate_floor_ratio": 0.7,
            "min_area_pixels": 16,
            "max_area_ratio": 0.15,
            "max_candidates": 10,
            "morph_close_kernel": 3
          }
        },
        "blob_rules": {
          "enabled": true,
          "min_area_mm2": 0.5,
          "max_area_mm2": 500.0,
          "min_solidity": 0.3,
          "min_contrast": 8.0,
          "pixel_to_mm_ratio": 0.15,
          "rule_mode": "all"
        },
        "patchcore": {
          "decision_score_margin": 0.95,
          "min_strong_patch_count": 1,
          "min_strong_component_count": 1
        }
      }
    ]
  }
}
```

注意 `patchcore.decision_score_margin` 从 1.08 下调至 0.95，`min_strong_patch_count` 从 3 下调至 1 —— 这些是"宽松阈值"的体现。

---

## 6. 文件改动清单

```
src/seat_defect_core/
├── config.py                          # [改] 新增 ClassifierFilterConfig, BlobRuleConfig, CandidateExtractionConfig
├── types/
│   ├── pipeline.py                    # [改] 新增 CascadeDecision, CandidateRegion, ClassifierPrediction, ConfirmedDefect
│   └── results.py                     # [改] TextureAnomalyResult 新增 cascade 等可选字段
├── cvops/
│   ├── candidate_extraction.py        # [新] 候选区域提取
│   ├── blob_features.py               # [新] 斑点特征计算
│   ├── blob_rules.py                  # [新] 斑点规则检查
│   └── cascade_decision.py            # [新] 级联判定引擎
├── classifier/                        # [新] 分类器包
│   ├── __init__.py
│   ├── base.py                        # DefectClassifier 抽象接口
│   ├── hog_lbp_svm.py                 # HOG+LBP+SVM 实现
│   ├── mobilenet.py                   # MobileNetV3 实现（可选）
│   └── factory.py                     # 分类器工厂函数
├── service/
│   ├── core.py                        # [改] InspectionService 新增 load_classifier, cascade_engine
│   └── inspection_camera.py           # [改] 插入级联过滤调用点
└── api.py                             # [改] SeatDefectInspector 暴露级联过滤相关参数

src/seat_defect_inspection/
├── cli_commands/
│   └── train_classifier.py            # [新] 分类器训练命令
└── classifier/
    └── training.py                    # [新] 训练管线实现
```

**总计**: 约 7 个新文件，5 个修改文件。所有新增在 `seat_defect_core` 内，训练命令在 `seat_defect_inspection` 内，遵守两层架构规则。

---

## 7. 风险评估

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| **100+ 样本不足以训练泛化好的分类器** | 中 | 先用传统特征+SVM（低数据需求），标注积累后切换 CNN；数据增强构造负样本多样性 |
| **宽松 PatchCore 导致候选过多，延迟增加** | 低 | `max_candidates=10` 硬限制，候选数超限按 heatmap 峰值截断 |
| **分类器引入新类别的漏检** | 中 | PatchCore `critical_rule` 作为不可绕过的 safety net；定期在验证集上评估端到端召回 |
| **多机位/多型号场景下的分类器泛化** | 中 | 初期每个 seat_model 训练独立分类器，验证跨型号泛化能力后再考虑共享 |
| **配置复杂度上升** | 低 | 所有新增配置都有合理默认值；`enabled=false` 时行为完全不变 |

---

## 8. 评估方案

### 8.1 离线评估

在留出测试集（建议至少 30 张缺陷 + 100 张正常）上对比：

| 指标 | 当前 PatchCore (标准阈值) | 方案后 (宽松 PatchCore + 分类器 + 斑点) |
|------|--------------------------|---------------------------------------|
| 缺陷召回率 (Recall) | ? | 目标 ≥ 95% |
| 正常样本通过率 | ? | 目标 ≥ 90% |
| 精确率 (Precision) | ? | 目标 ≥ 75% |
| F1 Score | ? | — |
| 平均推理延迟 (ms) | ? | 增量 ≤ 15ms |
| 候选区域数量 (avg) | — | ≤ 3 个/图 |

### 8.2 在线验证

1. **影子模式**: 新方案在后台运行，不控制实际判定，收集 1-2 周的对比数据
2. **A/B 测试**: 确认影子模式结果后，分时段交替使用新旧方案
3. **全量切换**: A/B 测试通过后全量上线

---

## 9. 迭代路线

```
Phase 1 (当前 → 2 周)
├── 实现 CandidateExtraction + BlobRules (无分类器)
├── 仅用传统斑点分析做过滤，验证"宽松 PatchCore + 斑点"的效果
└── 交付: 减少误报率 30-50%，不需要标注数据

Phase 2 (2 → 4 周)
├── 收集/整理标注数据，训练 HOG+LBP+SVM 分类器
├── 实现分类器接口 + 级联判定引擎
├── 离线评估 + 影子模式验证
└── 交付: 完整级联方案，精确率提升

Phase 3 (可选，数据积累后)
├── 切换至 MobileNetV3/EfficientNet 分类器
├── 探索跨型号共享分类器
└── 交付: 更高精度的深度学习分类器
```

---

## 10. 待定事项

- [ ] 确认 `pixel_to_mm_ratio` 的标定方法和各机位的实际值
- [ ] 确认缺陷标注格式和与现有数据管道的对接方式
- [ ] 评估 PatchCore `critical_rule` 在宽松阈值下的实际表现（是否会产生大量候选）
- [ ] 确认是否需要对不同缺陷类型（划痕、污渍、破洞、缝线异常）分别训练分类器
