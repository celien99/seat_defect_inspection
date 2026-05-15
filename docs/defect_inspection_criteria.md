# 缺陷检测评判标准与 PatchCore 参数说明

## 检测流水线全景

```
图片进入 → [1.图像质量] → [2.YOLO检测] → [3.ROI裁剪] → [4.PatchCore纹理] → [5.颜色一致性] → [6.多机位融合] → 最终结论
              ↓ 不合格             ↓ 不合格                           ↓ 异常
            REJECT              REJECT                              NG
```

一张图片进入后经过 **6 道关卡**，最终判定为 OK（合格）、NG（有缺陷）或 REJECT（拒检）。

---

## 第1关：图像质量检查 (`QualityGuardConfig`)

**目的**：过滤掉拍糊了、太暗、太亮的废图，避免在废图上浪费检测算力。

| 参数 | 默认值 | 通俗解释 |
|------|--------|----------|
| `min_laplacian_variance` | 80.0 | 图像清晰度下限。用拉普拉斯算子计算，值越高要求越清晰，低于此值 = 拍糊了 |
| `min_brightness_mean` | 30.0 | 平均亮度下限（0-255），低于 30 = 太暗看不清 |
| `max_brightness_mean` | 225.0 | 平均亮度上限，高于 225 = 过曝一片白 |
| `max_overexposed_ratio` | 0.25 | 过曝像素占比上限，超过 25% 就算过曝 |
| `max_underexposed_ratio` | 0.35 | 欠曝像素占比上限，超过 35% 就算太暗 |

**判定逻辑**：任一条件不满足 → `REJECT`（拒检，不继续后续检测）。

额外硬编码规则：平均亮度 ≤ 3.0 判为黑帧，平均亮度 ≥ 252.0 判为白帧，直接 REJECT。

---

## 第2关：YOLO 目标检测 + 实例分割 (`DetectionConfig`)

**目的**：在图片中找到座椅，画出它的轮廓（segmentation mask）。

| 参数 | 默认值 | 通俗解释 |
|------|--------|----------|
| `model_path` | - | YOLO 模型文件路径（.pt） |
| `target_class` | `"seat"` | 要检测的目标类别名称 |
| `confidence` | 0.25 | 置信度阈值，低于此值的检测框丢弃 |
| `iou` | 0.45 | NMS 去重阈值，越高越严格（保留更少的重叠框） |
| `device` | `"cpu"` | 推理设备，cpu / cuda |
| `imgsz` | 960 | 输入 YOLO 的图片尺寸，越大越精细但越慢 |
| `fill_segmentation_holes` | true | 是否填充分割 mask 内部的空洞 |
| `segmentation_hole_fill_max_area_ratio` | 0.08 | 填充空洞的最大面积占比（相对于总面积） |

**判定逻辑**：找不到目标（无检测框或 mask 为空）→ `REJECT`。

---

## 第3关：ROI 裁剪与精修 (`RoiRefineConfig`)

**目的**：根据 YOLO 找到的轮廓，把座椅区域裁出来并整理干净，为 PatchCore 准备输入。

| 参数 | 默认值 | 通俗解释 |
|------|--------|----------|
| `crop_expand_ratio` | 0.05 | 裁剪框外扩 5%，避免裁得太紧切掉边缘 |
| `crop_shrink_ratio` | 0.0 | 裁剪框内缩比例，0 = 不缩 |
| `mask_erode_pixels` | 1 | 向内腐蚀 1 像素，去掉轮廓边缘的毛刺 |
| `edge_ignore_pixels` | 6 | 忽略边缘 6 像素，边界区域容易混入背景噪声 |
| `alignment.output_width` | 256 | 统一缩放到 256 宽度 |
| `alignment.output_height` | 256 | 统一缩放到 256 高度 |

**判定逻辑**：裁剪出的区域为空 → `REJECT`。

---

## 第4关：PatchCore 纹理异常检测（核心判定）

这是整个系统**最关键的判定环节**。它的核心思想是：

> "我见过几百张好座椅长什么样，你这张跟它们像不像？不像的地方在哪、多大、多严重？"

### 4.1 PatchCore 工作原理

#### 训练阶段（只用合格样本）

```
若干张好座椅图片（如 300 张）
  → 每张切成很多小块（patch），比如 32×32 像素
  → 用预训练神经网络提取每个小块的特征向量（一串数字，描述纹理）
  → 用 coreset 算法精选最有代表性的小块存起来（memory bank）
  → 用这些图自己算自己的分数分布，确定"正常范围"的阈值
```

#### 推理阶段（检测新图片）

```
新图片
  → 同样切成小块，提取特征向量
  → 每个小块去 memory bank 里找最相似的邻居
  → 距离越远 = 越不像正常样本 = 越异常
  → 所有小块的异常距离构成一张"异常热力图"
  → 热力图 + 决策规则 → 判定 OK / NG
```

#### 图片分数的计算

```
每张图的所有 patch 到 memory bank 的最短距离 → 取第 99 百分位作为这张图的分数
```

也就是说，一张图的分数反映的是它"最异常的那 1% patch"的异常程度。

### 4.2 Patch 提取参数 (`PatchCoreConfig`)

| 参数 | 默认值 | 通俗解释 |
|------|--------|----------|
| `backend` | `"full"` | 特征提取后端，目前只支持 `"full"`（神经网络） |
| `image_size` | 256 | ROI 输入尺寸 |
| `patch_size` | 32 | 每个 patch 的像素大小。越小 = 检测越细粒度，但计算量越大 |
| `stride` | 16 | patch 之间的步长。16 = 相邻 patch 重叠一半，覆盖更密集 |
| `max_memory` | 1024 | memory bank 最多存多少特征向量，越大越全但越慢。推荐 500 样本时设为 1024 |
| `texture_input` | `"lab_l"` | 输入 PatchCore 的颜色通道：`lab_l` = 仅 L 通道（亮度） |
| `coreset_sampling_ratio` | 0.1 | memory bank 采样比例，0.1 = 取 10% 的 patch |
| `backbone_name` | `"wide_resnet50_2"` | 特征提取的骨干网络 |
| `feature_layers` | `["layer2","layer3"]` | 从网络的哪些层提取特征 |
| `backbone_device` | `"cpu"` | 骨干网络推理设备 |

### 4.3 有效 Patch 过滤参数

每个 patch 必须满足一定条件才算"有效"，无效 patch 不参与异常判定。

| 参数 | 默认值 | 通俗解释 |
|------|--------|----------|
| `min_target_coverage` | 0.8 | patch 内座椅像素占比 ≥ 80% 才算有效 |
| `max_ignore_overlap` | 0.1 | patch 内忽略区域（边缘）占比 ≤ 10% |
| `min_valid_patch_ratio` | 0.65 | 有效 patch 占总 patch 的比例 ≥ 65%，否则 REJECT |

### 4.4 训练阈值公式

训练时用以下公式计算阈值（即"正常线"画在哪）：

```
threshold = max(
    quantile(训练图片分数, threshold_quantile),          // ① 分位数
    训练图片分数的均值 + 3 × 标准差,                      // ② 统计上界
    quantile(训练图片分数, training_threshold_upper_quantile) // ③ 鲁棒上界
)
```

三个值取最大，作为最终阈值。

**为什么取最大？** 这是"取最保守的那个"。如果训练数据质量高、分布紧致，①②③ 都会很低；如果训练数据本身有波动，②（3σ 上界）会成为主导项，防止阈值过低导致误检。

**③ 的设计意图**：曾用 `最大分数 × 1.1`，但这会被"单个离群样本"绑架——300 张图里只要有一张分数异常高，阈值就被拉高。改用 `quantile(scores, training_threshold_upper_quantile)` 后，通过调整分位数灵活控制离群排除力度。

> **推荐值**：300 样本时设为 `0.995`（排除最高约 1-2 张），500 样本时也建议 `0.995`（排除最高约 2-3 张）。`0.999` 在样本量不到 1000 时实际上约等于 `max`，无法有效排除离群值。

| 参数 | 默认值 | 通俗解释 |
|------|--------|----------|
| `threshold_quantile` | 0.99 | ① 的分位数。0.99 = 第 99 百分位 |
| `training_threshold_upper_quantile` | 0.995 | ③ 的分位数。推荐 0.995，在 300-500 样本量下有效排除 1-3 个离群样本 |

**阈值越低 → 检测越敏感，但误检风险也越高。**

### 4.5 异常判定：4 条规则

推理时，一张图被判为 NG 需要满足**至少一条**规则：

#### 规则1：常规规则 (`normal_rule`) — 检测大面积可见缺陷

```
图片分数 > 训练阈值 × decision_score_margin（1.08）
AND 强异常 patch 数量 ≥ min_strong_patch_count（3）
AND 最大异常连通域大小 ≥ min_strong_component_count（2）
AND 强异常 patch 占比 ≥ min_strong_patch_ratio（1.5%）
AND 最大连通域占比 ≥ min_strong_component_ratio（1%）
```

通俗版：**整体分够高，且异常区域有一定规模**。

#### 规则2：临界规则 (`critical_rule`) — 快速放行小面积高亮缺陷

```
图片分数 > 训练阈值 × critical_score_margin（1.35）
AND 最热 patch 分值 > 训练阈值 × critical_peak_score_margin（1.45）
AND max(强异常连通域, 决策线连通域) ≥ critical_min_component_patch_count（2）
```

通俗版：**哪怕面积很小，但异常强度极高（明显缺陷），直接判 NG**。

> 连通域检查取 `max(强异常, 决策线)` 是为了防止"孤立超热 patch + 周围中等热度邻居"的场景漏检：中心超热 patch 在强异常 mask 中可能孤立（邻居未达 `score × 0.9`），但在决策线 mask 中与邻居连通。取两者的较大值确保不被漏掉。

#### 规则3：峰值规则 (`peak_rule`) — 兜底，确保不遗漏

```
最热 patch 分值 > 训练阈值 × decision_score_margin（1.08）
AND 最大决策连通域 patch 数 ≥ min_peak_component_patch_count（1）
```

通俗版：**哪怕面积很小，但热度已经到了决策线，也要抓出来**。与"跨过决策线的总 patch 数"不同，这里要求这些 patch 互相连通，天然过滤了分散噪声。默认只需 1 个连通 patch 即触发，可通过 `min_peak_component_patch_count` 提高到 2-3 来进一步过滤噪声。

#### 规则4：放行 (`none`)

以上都不满足 → OK。

#### 规则判定参数一览

> **注意**：`decision_score_margin`、`critical_score_margin`、`critical_peak_score_margin` 三个乘数都经过 `_threshold_margin(x) = max(1.0, x)` 处理，即**永远不会低于 1.0**。这意味着决策线永远 ≥ 训练阈值，确保推理阶段不会比训练时更宽松。

| 参数 | 默认值 | 通俗解释 |
|------|--------|----------|
| `decision_score_margin` | 1.08 | 决策阈值乘数。训练阈值 × 1.08 = 常规/峰值规则的分数线 |
| `strong_patch_score_ratio` | 0.9 | 强异常 patch 的定义：分值 ≥ max(阈值, 图片分) × 0.9 |
| `min_strong_patch_count` | 3 | 常规规则：强异常 patch 至少 3 个 |
| `min_strong_component_count` | 2 | 常规规则：最大强异常连通域至少包含 2 个 patch |
| `min_strong_patch_ratio` | 0.015 | 常规规则：强异常 patch 占比至少 1.5% |
| `min_strong_component_ratio` | 0.01 | 常规规则：最大强异常连通域占比至少 1% |
| `critical_score_margin` | 1.35 | 临界规则：图片分数超过阈值 35% |
| `critical_peak_score_margin` | 1.45 | 临界规则：最热 patch 超过阈值 45% |
| `critical_min_component_patch_count` | 2 | 临界规则：最小连通域 patch 数（取 max(强异常, 决策线) 连通域中的较大值），下限 1 |
| `min_peak_component_patch_count` | 1 | 峰值规则：最小决策连通域 patch 数，提高可过滤单 patch 噪声 |

### 4.6 热力图归一化

热力图的显示采用**相对于决策阈值的归一化**（`normalize_map_against_threshold`）：

- `阈值 × 0.5` 以下 → 冷色（蓝），正常区域
- `阈值 × 0.5 ~ 阈值 × 1.08` → 暖色（黄），接近决策线的区域
- `阈值 × 1.08` 以上 → 红色，超过决策线的异常区域

这样可以一眼看到哪些区域跨过了判定线。

---

## 第5关：颜色一致性分支 (`ColorBranchConfig`)

**目的**：检查座椅颜色是否跟训练样本一致（色差、染色不均等）。

**注意**：这个分支默认是**关闭的**（`enabled: false`）。

| 参数 | 默认值 | 通俗解释 |
|------|--------|----------|
| `enabled` | false | 是否开启颜色检测 |
| `threshold_quantile` | 0.99 | 颜色异常阈值分位数 |
| `threshold` | null | 手动指定颜色阈值（不为 null 时覆盖分位数计算） |
| `min_valid_pixel_ratio` | 0.4 | 有效像素占比至少 40%，低于此值不计算颜色特征 |
| `training_threshold_upper_quantile` | 0.999 | 训练阈值上限分位数，排除最高 0.1% 离群值 |

**颜色检测逻辑**：将 ROI 转到 LAB 色彩空间，取有效像素的 L/A/B 均值+标准差（6 维特征），计算与训练正常分布的标准化距离，超过阈值 = 颜色异常。

颜色分支的训练阈值公式与 PatchCore 一致（取 `quantile(0.99)`、`mean+3σ`、`quantile(training_threshold_upper_quantile)` 三者的最大），同样具备统计鲁棒性。

### Full-ROI 模式联合判定

```
PatchCore 异常 AND 颜色异常 → NG（texture_and_color_anomaly，双重确认）
PatchCore 异常               → NG（texture_anomaly）
颜色异常                      → NG（color_anomaly）
都正常                        → OK
质量不通过                    → REJECT
```

### Region 模式下的判定

当相机配置了 region 时，每个 region 独立跑 PatchCore（**不走颜色分支交叉验证**），然后合并：

```
任意 region 异常               → NG（列出异常 region ID，若同时有 REJECT region 会附加 `_with_reject:<region_ids>`）
存在 REJECT region（无 NG）    → REJECT（列出首个 REJECT region 及原因）
颜色异常                        → NG（仅在全 ROI 颜色分支启用时）
所有 region 正常                → OK
```

> **注意**：当同时存在 NG 和 REJECT region 时，最终判定为 NG，但 reason 中会附加 REJECT region 的 ID（如 `region_texture_anomaly:r1_with_reject:r3`），确保不丢失 REJECT 信息。

---

## 第6关：多机位融合 (`FusionConfig`)

**目的**：将多个摄像头的检测结果合并成最终结论。

| 参数 | 默认值 | 通俗解释 |
|------|--------|----------|
| `ng_strategy` | `"any"` | NG 策略：`any` = 任一机位 NG 即 NG，`all` = 全 NG 才 NG，`majority` = 多数 NG 才 NG |
| `reject_on_any_reject` | true | 任一机位 REJECT 即 REJECT |
| `defect_overrides_reject` | true | 有缺陷时覆盖 REJECT（缺陷优先级高于质量问题） |

---

## 从训练到推理：关键数字链路示例

假设训练阈值得出 `threshold = 0.2`：

```
推理时一张新图片：
  图片分数 = 0.25
  
  决策线 = 0.2 × 1.08 = 0.216     ← 常规规则和峰值规则的分数线（_threshold_margin 保证 ≥ 0.2）
  临界分数线 = 0.2 × 1.35 = 0.270  ← 临界规则的分数线
  临界峰值线 = 0.2 × 1.45 = 0.290  ← 临界规则的峰值线
  
  0.25 > 0.216 ✓  → 常规规则可能触发（还需满足 patch 数量和面积条件）
  0.25 > 0.216 ✓  → 峰值规则可能触发（还需满足 min_peak_component_patch_count）
  0.25 < 0.270 ✗  → 临界规则不触发
```

**训练阈值 `threshold` 是整个系统的根基**。阈值越低，所有判定线都越低，检测越敏感。由于 `_threshold_margin` 的 ≥1.0 保证，`decision_score_margin` 设为 0.95 时仍然按 1.0 生效。要降低决策线，唯一途径是降低训练阈值本身（如调低 `training_threshold_upper_quantile`）。

---

## 收紧检测的调节旋钮

按影响从大到小：

1. **`training_threshold_upper_quantile`** ↓（如设为 0.99，让训练阈值更低。这是最直接、影响最大的方式）
2. **`decision_score_margin`** ↓（如设为 1.05，决策线更接近训练阈值。注意：会被 `_threshold_margin` 截断到 ≥1.0，设 0.95 仍按 1.0 生效）
3. **`min_strong_patch_count`** ↓（如设为 2，更少的异常 patch 就能触发常规规则）
4. **`strong_patch_score_ratio`** ↓（如设为 0.85，更多 patch 被归类为"强异常"）
5. **`min_strong_patch_ratio`** / **`min_strong_component_ratio`** ↓（降低占比门槛）
6. **`critical_min_component_patch_count`** ↓（如设为 1，单个连通 patch 的超热缺陷即可触发临界规则）
7. **`min_peak_component_patch_count`** ↑（提高峰值规则连通域要求，仅过滤噪声，非收紧检测。注意：调高后极热的孤立单 patch 缺陷仍会被临界规则兜底捕获，因为临界规则取 `max(强异常, 决策线)` 连通域）
