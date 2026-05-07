# PatchCore 调参新手指南

本文面向工业视觉初学者，目标是让现场人员快速理解当前项目里 PatchCore 参数的作用，并能根据检测现象做有方向的调整。

当前项目的 PatchCore 不是直接拿整张原图训练，而是先走完整图像链路：

```text
原图 -> preprocess -> YOLO / fallback_box -> ROI -> target_mask / valid_mask -> 透明 BGRA PatchCore 输入 -> full PatchCore
```

因此调参时不要只盯着 `patchcore` 小节。上游预处理、YOLO 定位、ROI 裁剪和 mask 质量都会直接影响 PatchCore 学到的正常分布。

## 1. 先记住三条原则

1. PatchCore 只学习正常样本。
   `train_good_dir` 里混入 NG 图，会把缺陷学成正常，后续会漏检。

2. 训练和推理看到的图像必须一致。
   修改 `preprocess`、YOLO、ROI、mask、`patchcore.image_size`、backbone 或透明 BGRA 规则后，都要重新执行 `train-patchcore`。

3. 先查输入，再调阈值。
   如果 ROI 裁偏、mask 错、黑边进入 PatchCore、正常样本不足，单纯调高/调低阈值只会把问题藏起来。

## 2. 推荐调参顺序

按下面顺序排查，效率最高：

1. 看调试图。
   重点看 `patchcore_input.png`、`roi.png`、`target_mask.png`、`valid_mask.png`、`heatmap.png`。确认 PatchCore 看到的是座椅有效区域，不是背景、边缘、黑底或夹具。

2. 检查训练数据。
   每个机位都要有足够的正常样本，覆盖正常光照、正常材质批次、正常姿态和现场允许的轻微波动。

3. 固定一批验证样本。
   至少包含正常样本、真实 NG、小缺陷、边缘样本、容易误报样本。每次改参数都用 `inspect-folder` 跑同一批样本。

4. 先调输入质量参数。
   优先调整 ROI、mask、`image_size`、`min_target_coverage`、`max_ignore_overlap`。

5. 再调模型容量参数。
   调整 `max_memory`、`coreset_sampling_ratio`、backbone 和特征层。

6. 最后调判定参数。
   调整 `threshold_quantile`、`decision_score_margin`、`critical_*`、`min_strong_*`。

## 3. 当前项目中的核心参数

示例配置位置：

```text
configs/seat_defect_inspection.mvs.json
```

每个机位都有自己的：

```json
"patchcore": {
  "backend": "full",
  "image_size": 320,
  "patch_size": 16,
  "stride": 8,
  "max_memory": 512,
  "threshold_quantile": 0.95,
  "coreset_sampling_ratio": 0.1,
  "texture_input": "lab_l",
  "backbone_name": "wide_resnet50_2",
  "backbone_pretrained": true,
  "backbone_device": "cuda",
  "min_target_coverage": 0.65,
  "max_ignore_overlap": 0.1,
  "min_valid_patch_ratio": 0.35,
  "decision_score_margin": 1.05,
  "strong_patch_score_ratio": 0.85,
  "min_strong_patch_count": 2,
  "min_strong_component_count": 2,
  "min_strong_patch_ratio": 0.006,
  "min_strong_component_ratio": 0.004,
  "critical_score_margin": 1.1,
  "critical_peak_score_margin": 1.15,
  "critical_min_component_patch_count": 2
}
```

## 4. 模型输入和特征参数

这些参数决定 PatchCore 看到什么、用什么特征表达正常样本。

| 参数 | 当前值 | 作用 | 调大/调强的效果 | 调小/调弱的效果 |
| --- | --- | --- | --- | --- |
| `backend` | `full` | 选择 PatchCore 后端 | `full` 使用 CNN feature map；`transformer` 使用 ViT patch token | 不应改成 handcrafted |
| `image_size` | `320` | PatchCore ROI 输入尺寸 | 保留更多细节，小缺陷更容易被看见，但更慢 | 更快，但小划痕、小脏污可能被压掉 |
| `texture_input` | `lab_l` | CNN 输入前的纹理口径 | `lab_l` 更关注亮度和纹理，降低颜色波动影响 | 若改 RGB，可增强颜色缺陷敏感度，但也更容易受光照/色差影响 |
| `backbone_name` | `wide_resnet50_2` | 特征骨干 | `full` 支持 `resnet18/resnet50/wide_resnet50_2`；`transformer` 支持 `vit_b_16/vit_b_32/vit_l_16/vit_l_32` | 较小 backbone 更快，可能降低复杂纹理区分能力 |
| `feature_layers` | `layer2/layer3` | CNN 中间层 | 只对 `full` 后端生效，增加层可能增强表达但计算更重 | 层太少会丢部分纹理或结构信息 |
| `feature_pool_kernel_size` | `3` | 对特征图做局部平均 | 更抗噪，热力图更平滑 | 更敏感，小缺陷更尖锐，也更容易误报 |
| `backbone_pretrained` | `true` | 是否使用预训练权重 | 必须启用或提供本地权重 | 随机初始化 backbone 没有工业检测意义 |
| `backbone_weights_path` | 空 | 本地预训练权重路径 | 离线现场推荐配置，避免下载依赖 | 为空时依赖 torchvision 缓存或网络 |
| `backbone_device` | `cuda` | CNN 特征提取设备 | 影响速度，不应影响质量 | CPU 更慢 |

注意：`full` 后端主要从 CNN feature map 提 embedding，`patch_size` 和 `stride` 对质量影响较弱，更多是历史 handcrafted 路径留下的配置项和模型元数据。`transformer` 后端从 ViT patch token 提 embedding，token 网格由 ViT 自身 patch size 决定；使用 torchvision 预训练 ViT 权重时，`image_size` 应设为 `224`，或改用与本地权重匹配的输入尺寸。

Transformer 后端示例：

```json
"patchcore": {
  "backend": "transformer",
  "image_size": 224,
  "texture_input": "lab_l",
  "backbone_name": "vit_b_16",
  "backbone_pretrained": true,
  "backbone_device": "cuda",
  "max_memory": 512,
  "coreset_sampling_ratio": 0.1,
  "min_target_coverage": 0.55,
  "max_ignore_overlap": 0.15
}
```

该后端只替代 PatchCore 的特征提取和异常评分步骤。YOLO 定位、ROI/mask、质量门控、多机位融合仍沿用当前确定性流程。

## 5. Memory Bank 和阈值参数

PatchCore 的“训练”本质是建立正常 patch embedding 的 memory bank，并根据正常样本分数校准阈值。

| 参数 | 当前值 | 作用 | 调大效果 | 调小效果 |
| --- | --- | --- | --- | --- |
| `max_memory` | `512` | memory bank 最大容量 | 正常分布覆盖更完整，可能减少误报，但推理变慢 | 更快，但可能丢掉正常变化，导致误报 |
| `coreset_sampling_ratio` | `0.1` | 从全部正常 patch 中抽样比例 | 保留更多正常 patch，模型更稳，但更慢 | 更快，但正常分布覆盖变窄 |
| `threshold_quantile` | `0.95` | 用正常样本分数的分位数校准阈值 | 阈值更高，误报少，但可能漏检 | 阈值更低，更敏感，但误报增加 |

经验：

- 正常样本变多后，可以适度提高 `max_memory`，例如从 `512` 到 `768` 或 `1024`，但要重新验证推理速度。
- 误报主要来自正常纹理覆盖不足时，优先增加训练样本，再考虑提高 `max_memory`。
- 漏检真实缺陷时，不要第一步就降低 `threshold_quantile`，应先看热力图是否已经打到缺陷位置。

## 6. 有效 Patch 过滤参数

这些参数决定哪些 patch 能进入训练和推理。它们直接影响模型是否被背景、边缘和错误 mask 污染。

| 参数 | 当前值 | 作用 | 调大效果 | 调小效果 |
| --- | --- | --- | --- | --- |
| `min_target_coverage` | `0.65` | patch 内目标前景占比至少多少才有效 | 更严格，背景更少，误报可能减少 | 更宽松，边缘/小区域缺陷更容易参与，但背景风险上升 |
| `max_ignore_overlap` | `0.1` | patch 与忽略区域重叠超过多少就剔除 | 更宽松，保留更多 patch | 更严格，减少干扰，但可能有效 patch 过少 |
| `min_valid_patch_ratio` | `0.35` | 推理时有效 patch 比例低于该值则 REJECT | 更严格，低质量 ROI 更容易拒判 | 更宽松，减少 REJECT，但可能让不可靠图像参与判定 |

经验：

- 热力图经常打到边缘或背景：提高 `min_target_coverage`，或检查 ROI/mask。
- 小缺陷靠近边界被漏掉：适当降低 `min_target_coverage`，例如从 `0.65` 到 `0.55`。
- 大量 `low_valid_patch_ratio`：先看 `valid_mask` 是否过小，再考虑降低 `min_valid_patch_ratio`。

## 7. 异常判定参数

这些参数不改变模型学到的 memory bank，但会改变 OK/NG 判定的敏感度。

| 参数 | 当前值 | 作用 | 调大效果 | 调小效果 |
| --- | --- | --- | --- | --- |
| `decision_score_margin` | `1.05` | 常规图像分数阈值倍率 | 更保守，误报减少 | 更敏感，漏检减少但误报增加 |
| `strong_patch_score_ratio` | `0.85` | 定义强异常 patch 的分数比例 | 强 patch 更少，判定更稳 | 强 patch 更多，更容易报 NG |
| `min_strong_patch_count` | `2` | 至少多少个强异常 patch 才可信 | 减少孤立噪声误报 | 小缺陷更容易触发 |
| `min_strong_component_count` | `2` | 最大异常连通区域至少多少 patch | 减少离散噪声误报 | 小面积缺陷更容易触发 |
| `min_strong_patch_ratio` | `0.006` | 强异常 patch 占有效 patch 的比例 | 大面积证据要求更高 | 小缺陷更容易报出 |
| `min_strong_component_ratio` | `0.004` | 最大异常连通域占比要求 | 更重视成片缺陷 | 更容易捕捉小连通缺陷 |
| `critical_score_margin` | `1.1` | 小缺陷快速规则的整体分数倍率 | 更保守 | 更容易报小缺陷 |
| `critical_peak_score_margin` | `1.15` | 小缺陷快速规则的局部峰值倍率 | 减少尖峰噪声误报 | 更敏感，点状缺陷更容易检出 |
| `critical_min_component_patch_count` | `2` | 小缺陷规则要求的最小连通 patch 数 | 减少单点噪声 | 小缺陷更容易触发 |

经验：

- 漏检点状、小划痕、小脏污：优先看 `critical_peak_score_margin`、`critical_score_margin`、`critical_min_component_patch_count`。
- 正常纹理误报：优先提高 `decision_score_margin` 或 `min_strong_component_count`。
- 热力图只有孤立噪点：提高 `min_strong_patch_count` 或 `min_strong_component_count`。

## 8. 上游配置也会影响 PatchCore 质量

PatchCore 训练输入来自上游链路，因此这些配置也要纳入调参范围。

| 配置区域 | 关键参数 | 对 PatchCore 的影响 |
| --- | --- | --- |
| `train_good_dir` | 正常样本目录 | 决定正常分布上限。样本少、覆盖不足、混入 NG 都会直接伤害模型 |
| `preprocess` | 降噪、白平衡、光照校正、CLAHE | 改变纹理和亮度分布。修改后必须重新训练 PatchCore |
| `detection` | `model_path`、`confidence`、`iou`、`fallback_box` | 决定目标定位和 mask 来源。定位漂移会让 ROI 分布漂移 |
| `roi` | `crop_expand_ratio`、`crop_shrink_ratio`、`edge_ignore_pixels`、`alignment` | 决定 PatchCore 看哪些区域、边缘是否被排除、输出尺寸是否稳定 |
| `color_insensitive_mode` | true/false | 当前为 true 时更偏亮度纹理检测，弱化颜色波动 |
| `color_branch` | `enabled`、`threshold_quantile` | 当前在 `color_insensitive_mode=true` 时会跳过颜色分支 |

## 9. 常见现象和处理路径

### 9.1 小缺陷漏检

先看：

- `patchcore_input.png` 中缺陷是否清晰可见。
- `heatmap.png` 是否在缺陷位置有热点。
- `target_mask / valid_mask` 是否覆盖缺陷位置。

处理顺序：

1. 若缺陷在输入图中已经被压没：提高 `roi.alignment.output_width / output_height` 和 `patchcore.image_size`，然后重新训练。
2. 若缺陷在 mask 外：修 YOLO mask、ROI 或降低 `min_target_coverage`。
3. 若 heatmap 有热点但没报 NG：降低 `critical_peak_score_margin` 或 `critical_score_margin`。
4. 若只有单个点状热点：降低 `critical_min_component_patch_count` 要谨慎，容易增加噪声误报。

### 9.2 正常样本误报

先看：

- 热点是否在真实座椅区域。
- 热点是否集中在边缘、背景、阴影、反光或黑底。
- 该正常纹理是否在训练集中出现过。

处理顺序：

1. 热点在背景或边缘：修 ROI/mask，提高 `min_target_coverage`，或增加 `edge_ignore_pixels`。
2. 热点是真实正常纹理：增加该类正常样本，再重新训练。
3. 样本覆盖足够但仍误报：提高 `max_memory` 或 `coreset_sampling_ratio`。
4. 仍误报：提高 `decision_score_margin`、`strong_patch_score_ratio` 或 `min_strong_component_count`。

### 9.3 大量 REJECT

看报告里的 reason：

- `target_not_found`：YOLO 没找到目标，检查 `model_path`、`target_class`、`confidence` 或 `fallback_box`。
- `quality_blur / underexposed / overexposed`：采图质量或质量门控问题。
- `low_valid_patch_ratio`：ROI/mask 有效区域太少，检查 `valid_mask`，必要时降低 `min_valid_patch_ratio`。

处理顺序：

1. 先修采图、曝光、焦距和 YOLO。
2. 再修 ROI/mask。
3. 最后再放宽 `min_valid_patch_ratio`。

### 9.4 热力图总在边缘亮

常见原因：

- ROI 裁剪边缘包含背景或黑边。
- mask 边缘锯齿进入了有效 patch。
- 训练集中边缘形态和推理时不一致。

处理顺序：

1. 增加 `roi.edge_ignore_pixels`。
2. 提高 `min_target_coverage`。
3. 检查透明 BGRA 输入是否正确，不能把黑底送进 PatchCore。
4. 固定 ROI 尺寸和裁剪策略后重新训练。

### 9.5 换光照后效果明显漂移

处理顺序：

1. 增加不同正常光照条件下的训练样本。
2. 检查 `preprocess` 的白平衡、光照校正和 CLAHE 是否稳定。
3. 保持 `texture_input = lab_l`，不要轻易切到 RGB。
4. 若颜色差异本身是缺陷，再单独评估颜色分支，而不是直接让 PatchCore 对颜色过敏。

### 9.6 训练速度太慢

优先检查：

- 是否使用 CUDA 环境运行。
- YOLO 检测是否仍在 CPU。
- 训练样本数量是否过大。
- `image_size` 是否过高。
- `max_memory` 是否过高。

处理顺序：

1. 保持 `backbone_device = cuda`。
2. 如 GPU 允许，把 YOLO `detection.device` 改成 `cuda:0`。
3. 不要盲目提高 `image_size`。
4. `max_memory` 先保持 `512`，质量不够再加。

## 10. 调参记录模板

每次改参数都建议记录：

```text
日期：
机位：
修改前参数：
修改后参数：
训练样本数量：
验证样本数量：
OK 误报数量：
NG 漏检数量：
REJECT 数量：
主要观察：
是否保留该参数：
```

一次只改一类参数。不要同时改 `image_size`、ROI、阈值和 memory bank，否则无法判断到底是哪一个参数起作用。

## 11. 必须重新训练的情况

出现以下变化后，必须重新执行：

```bash
seat-defect-inspection train-patchcore \
  --config configs/seat_defect_inspection.mvs.json
```

必须重训的变化包括：

- 正常样本集变化。
- `preprocess` 参数变化。
- YOLO 权重、`confidence`、`target_class` 或 `fallback_box` 变化。
- ROI 裁剪、输出尺寸、边缘忽略、mask 规则变化。
- `patchcore.image_size`、`texture_input`、backbone、`feature_layers` 变化。
- 透明 BGRA 输入规则变化。
- 从旧 handcrafted 模型迁移到 full PatchCore。

只调整纯运行判定参数时，代码支持部分运行时覆盖，但现场仍建议在固定验证集上完整验证后再上线。

## 12. 新手调参口诀

```text
先看图，再看 mask；
先补正常样本，再调阈值；
小缺陷看 image_size 和 critical；
误报先查背景边缘，再加样本；
REJECT 先查 YOLO / 质量 / valid_mask；
改了输入链路就重训。
```
