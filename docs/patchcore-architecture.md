# PatchCore 封装架构详解

> 面向小白的逐层拆解：从"PatchCore 是什么"到"本项目为什么这样封装"。

---

## 1. 一句话理解 PatchCore

PatchCore 是一种**工业缺陷检测算法**。工作原理可以概括为：

1. **训练阶段**：给模型看一堆"正常产品"的图片，它学会什么是"正常纹理"。
2. **推理阶段**：给它一张新图片，如果某个区域的纹理特征和"正常记忆"偏差太大，就判定为异常/缺陷。

核心的计算步骤是：

```
图片 → 切成很多小 patch → 用 CNN 提取每个 patch 的特征向量 
     → 从所有正常特征中精选一个"记忆银行(memory bank)"
     → 推理时计算新 patch 与记忆银行的最小距离 → 距离大 = 异常
```

---

## 2. 本项目对 PatchCore 的总体封装

### 2.1 文件地图

```
src/seat_defect_core/patchcore/       ← 核心层（推理+模型加载）
├── __init__.py        公开 PatchCoreService, LoadedModelBundle, ColorConsistencyService
├── features.py        特征提取（手工特征 + CNN 深度特征）       ~614 行
├── engine.py          PatchCore 运行时（预测、模型加载、配置覆写）~390 行
├── scoring.py         距离计算、阈值决策、coreset 采样           ~282 行
└── color_branch.py    颜色一致性分支（LAB 统计量）               ~130 行

src/seat_defect_core/service/
├── core.py            InspectionService + 模型缓存 + 特征提取器共享 ~462 行
├── inspection_camera.py  单机位检测编排（含 region 计划）       ~511 行
└── inspection.py      多机位检测总编排（含跨机位批处理）         ~322 行

src/seat_defect_core/
├── config.py          PatchCoreConfig 等配置模型                ~193 行
├── cvops/regions.py   ROI 局部区域切分                           ~106 行
└── types/results.py   TextureAnomalyResult 等类型               ~275 行

src/seat_defect_inspection/patchcore/
├── __init__.py        公开 PatchCoreTrainer
└── training.py        训练逻辑（PatchCoreTrainer + 模型保存）    ~183 行
```

### 2.2 两层架构：Core vs Tool

这个项目最根本的设计原则是 **严格的两层架构**：

```
┌──────────────────────────────────────────────┐
│  seat_defect_inspection (工具层)               │
│  - 负责训练模型、采集图像、CLI 命令              │
│  - 可以依赖 core，可以调用相机 SDK              │
└──────────────┬───────────────────────────────┘
               │ 调用
┌──────────────▼───────────────────────────────┐
│  seat_defect_core (核心层)                     │
│  - 只负责推理检测                               │
│  - 不采集图像、不训练模型、不遍历文件夹            │
│  - 对外暴露干净 API                             │
└──────────────────────────────────────────────┘
```

**为什么这样分？**

| 如果你需要... | 只需要装... | 为什么 |
|---|---|---|
| 在产线上跑推理 | 只装 `seat_defect_core` | 不带训练依赖和相机 SDK，部署包更小 |
| 训练新模型 | 装 `seat_defect_inspection` + `seat_defect_core` | 训练需要 GPU、相机、数据集扫描 |
| 换一个工业相机品牌 | 只改工具层 | core 完全不知"图像从哪来" |

---

## 3. 核心类的继承与组合关系

### 3.1 PatchCoreTrainer 继承 PatchCoreService

这是整个封装最巧妙的设计之一：

```python
# 训练器继承推理器 → 共享特征提取 + 距离计算
class PatchCoreTrainer(PatchCoreService):
    def fit(self, samples):   # 训练：建 memory bank + 算阈值
        """从正常样本训练模型"""
        ...
    
    def save(self, path):     # 保存为 .npz 文件
        """保存训练好的模型"""
        ...
```

**为什么用继承而不是两个独立类？**

1. **共享代码零重复**：`_normalize()`, `score_embeddings()`, `_get_torch_feature_extractor()`, `predict()` 等方法训练和推理都要用。继承天然共享这些实现。
2. **训练即验证**：`fit()` 方法里使用 `self.score_embeddings()` 做留一法交叉验证来标定阈值，这保证了训练时的打分逻辑和推理时**完全一致**。
3. **模型加载自然**：`PatchCoreService.load_bundle()` 是类方法，从 `.npz` 文件恢复出一个 `PatchCoreService` 实例，可以直接用于推理。训练器存的和推理器读的是同一格式。

### 3.2 组合而非继承的场景

不是所有关系都适合继承。以下是使用组合的例子：

```
LoadedModelBundle
  ├── patchcore: PatchCoreService      ← 组合
  └── color_profile: ColorReferenceProfile  ← 组合

InspectionService
  ├── _model_cache: ModelBundleCache    ← 组合
  └── _patchcore_predictor: PatchCorePredictorPool  ← 组合
```

**什么时候用组合？** 当对象之间的关系是"拥有/使用"而非"是一个"时。`LoadedModelBundle` 不是一个 `PatchCoreService`，它是一组模型的容器。

---

## 4. 模块逐层拆解

### 4.1 features.py — 特征提取层

这是 PatchCore 的"感官系统"，负责把图像的每个 patch 变成特征向量。

```
输入: (image, target_mask, ignore_mask)
  │
  ├── backend == "full"
  │     │
  │     ├── _TorchPatchFeatureExtractor (CNN backbone)
  │     │     ├── ResNet18 / ResNet50 / WideResNet50-2
  │     │     ├── 加载预训练权重或本地权重文件
  │     │     ├── 注册 forward hook 截取中间层特征图
  │     │     ├── 多尺度特征融合 (pool → bilinear align → concat)
  │     │     └── 按 valid mask 筛选有效 patch embedding
  │     │
  │     └── extract_many(): 一次 batch forward 处理多张图
  │
  └── backend == "handcrafted"
        │
        ├── 将图片转成指定色彩空间 (gray / lab_l / hsv_v / ycrcb_y)
        ├── 对每个滑动窗口 patch 提取手工特征:
        │     ├── 像素统计: mean, std, percentile(10), percentile(90)
        │     ├── 梯度统计: Sobel x/y grad, grad magnitude stats
        │     ├── Laplacian 方差和均值
        │     └── 下采样缩略图 (4×4 = 16 维)
        └── 返回每个 patch 的特征向量
```

#### 关键设计决策

**为什么支持两种 backend？**

| backend | 优点 | 缺点 | 适用场景 |
|---------|------|------|---------|
| `full` (CNN) | 表达能力强，适合复杂纹理 | 需要 PyTorch + GPU 内存 | 有明显纹理的座椅面料 |
| `handcrafted` | 不需要 PyTorch，极轻量 | 判别力有限 | 简单纹理或资源受限设备 |

两种 backend 共享同一套接口 `extract_patch_embeddings()`，调用方不需要关心底层实现。

**为什么用 hook 而不是 forward 返回值？**

```python
# 不在 backbone 顶层返回，而是 hook 内部层
self._handles = [
    _resolve_submodule(self.model, layer_name).register_forward_hook(...)
    for layer_name in self.layer_names  # 如 ["layer2", "layer3"]
]
```

因为 PatchCore 需要**中间层**的多尺度特征（浅层纹理 + 深层语义），而不是最后的分类输出。配置 `feature_layers: ["layer2", "layer3"]` 可以同时捕获局部纹理和全局结构。

**为什么要用 threading.Lock 保护特征提取？**

```python
def extract(self, image, ...):
    with self._lock:  # 同一 extractor 不能被多个线程同时使用
        self._features.clear()
        _ = self.model(input_tensor)
        feature_maps = [self._features[name] for name in self.layer_names]
```

Hook 回调写入 `self._features` 字典，如果两个线程同时 forward，A 的 hook 可能拿到 B 的特征图。加锁是线程安全的最低成本方案。

### 4.2 scoring.py — 打分与决策层

这是 PatchCore 的"判断系统"，决定一张图是否有异常。

```
embeddings (N × D 特征矩阵)
  │
  ├── coreset_subsample_indices()  ← 训练时精选 memory bank
  │     │
  │     └── 贪心最远点采样: 从所有正常 patch 中选出最有代表性的 M 个
  │
  ├── min_distance_to_bank() / min_distance_to_bank_torch()
  │     │
  │     ├── 对每个候选 patch: 计算到 memory bank 最近点的欧氏距离
  │     └── 用这个距离作为"异常分数"
  │
  └── _decide_patchcore_anomaly()  ← 判断是否异常
        │
        ├── normal_rule: 整体分数高 + 强异常 patch 形成连通域
        ├── critical_rule: 分数极高 + 峰值极高 (极小但强烈的缺陷)
        └── peak_rule: 热力图中已有超过阈值的连通区域
```

#### 三条判定规则的互补设计

这是本封装的一个重要创新点。传统的 PatchCore 只用"全局最高分 > 阈值"来判断，容易漏掉两种场景：

```
场景 A: 大面积浅色污渍
  → 整体分数可能不高，但许多 patch 都略高于阈值
  → 触发 normal_rule: strong_patch_count 多 + 连通域大

场景 B: 针尖大的深色瑕疵  
  → 整体分数可能被大量正常 patch 稀释
  → 触发 critical_rule: peak_patch_score 极高 → 极小缺陷放行

场景 C: 热力图峰值已超过判定线 ← 这个最容易被简单阈值漏掉
  → 触发 peak_rule: 即使不满足前两条，只要热力峰值越过 decision threshold
```

用代码表达就是：

```python
def _decide_patchcore_anomaly(score, threshold, evidence, config):
    normal_trigger  = (score > decision_threshold) and (enough strong patches with connectivity)
    critical_trigger = (score > critical_threshold) and (peak > critical_peak_threshold)
    peak_trigger     = (peak > decision_threshold) and (has a component)

    if normal_trigger and critical_trigger:  return True, "normal_and_critical"
    if critical_trigger:                     return True, "critical_rule"
    if normal_trigger:                       return True, "normal_rule"
    if peak_trigger:                         return True, "peak_rule"
    return False, "none"
```

#### 鲁棒的阈值估计

训练时阈值计算使用了**三重保护**：

```python
self.threshold = max(
    训练样本分数的指定分位数,          # 例: 0.99 分位数 → 99% 样本低于此值
    训练样本分数均值 + 3 倍标准差,     # 防止分位数被离群点扭曲
    训练样本分数的上界分位数,          # 例: 0.995 → 拒绝极端值
)
```

取三者的最大值，确保阈值不会被"看起来还行但实际有问题"的训练样本拉低。

#### GPU 加速的距离计算

```python
def _score_distances(self, embeddings, memory_bank):
    if device is CPU or backend is handcrafted:
        return min_distance_to_bank(embeddings, memory_bank)  # NumPy
    try:
        return min_distance_to_bank_torch(embeddings, bank, device=device)  # PyTorch GPU
    except Exception:
        return min_distance_to_bank(embeddings, memory_bank)  # 出错回退
```

- CPU/手工特征 → NumPy 逐 chunk 计算（`np.linalg.norm`）
- GPU → `torch.cdist` 批量计算，分 chunk 防止显存溢出
- GPU 出错 → 自动回退 NumPy，不中断流程

### 4.3 engine.py — 运行时引擎

这是 PatchCore 的"大脑"，整合特征提取 + 打分 + 决策。

```python
class PatchCoreService:
    config: PatchCoreConfig
    memory_bank: np.ndarray       # 训练好的正常特征精选集
    feature_mean: np.ndarray       # 训练集特征的均值（用于归一化）
    feature_std: np.ndarray        # 训练集特征的标准差（用于归一化）
    threshold: float               # 异常判定阈值
    
    def predict(image, target_mask, ignore_mask) -> TextureAnomalyResult:
        """完整预测: 提取特征 → 归一化 → 打分 → 建热力图 → 判定"""
    
    def predict_from_embeddings(embeddings, batch) -> TextureAnomalyResult:
        """从已提取的特征预测（批处理优化用）"""
    
    @classmethod
    def load_bundle(model_path, runtime_config, pipeline_signature) -> LoadedModelBundle:
        """从 .npz 文件加载训练好的模型"""
```

#### 为什么把 predict 拆成两步？

```python
# 正常流程（单张图）
embeddings = extract_patch_embeddings(image, config, masks)
result = patchcore.predict_from_embeddings(embeddings, batch)

# 批处理流程（多张图、多个区域）
# 1. 所有图的 embedding 一次性 batch forward
all_embeddings = extractor.extract_many([sample1, sample2, sample3, ...])
# 2. 各自的 embedding 分别走自己的 memory bank 打分
for (embeddings, batch), patchcore in zip(all_embeddings, patchcores):
    result = patchcore.predict_from_embeddings(embeddings, batch)
```

这个分离使得：**不同机位、不同区域可以共享同一个 CNN backbone 做 batch 特征提取，然后用各自专属的 memory bank 独立打分**。这是推理效率的核心优化。

#### 配置覆写机制

训练时存下的配置，推理时可以选择性收紧某些阈值：

```python
def _apply_runtime_patchcore_overrides(trained_config, runtime_config):
    # 只允许收紧，不允许放松
    overrides = {
        "min_target_coverage": max(trained, runtime),   # 只能要求更多有效像素
        "max_ignore_overlap":  min(trained, runtime),   # 只能要求更少忽略区域
    }
    # 判定相关阈值可以直接覆写
    for field in RUNTIME_DECISION_OVERRIDE_FIELDS:
        overrides[field] = runtime_config[field]
    return replace(trained_config, **overrides)
```

这解决了工业场景的常见需求：**产线调参时不想重新训练模型**，只想把阈值调严一点减少漏检。

**为什么只允许收紧不允许放松？** 因为放松阈值会让更多缺陷漏过去，违背安全原则。收紧是安全的操作。

#### 管道签名校验

```python
pipeline_signature = SHA256(检测配置 + ROI配置 + 质量配置)
if saved_signature != expected_signature:
    raise "模型不再匹配当前检测管道，请重新训练"
```

当 YOLO 模型换了、ROI 裁剪参数改了、质量门控阈值变了，之前训练好的 PatchCore 模型就不再适用——因为"喂给 PatchCore 的图像"已经变了。这个校验防止了"模型对不上数据"的隐藏 bug。

### 4.4 color_branch.py — 颜色一致性分支

这是对 PatchCore 纹理检测的补充，专门捕捉颜色偏差。

```
正常 ROI 样本 → 提取 LAB 空间特征 (L,a,b 均值和标准差)
              → 拟合正态分布 (mean + std)
              → 用马氏距离判定异常

推理时:
  新 ROI → 提取 LAB 特征 → 计算到正常分布的距离 → 距离 > 阈值 = 颜色异常
```

**为什么需要颜色分支？** 纹理特征（梯度、Laplacian）对颜色不敏感。一张被染色但纹理清晰的座椅面料，PatchCore 可能检测不到，但颜色分支能捕捉。

### 4.5 service/core.py — 运行时集成层

这是 PatchCore 与整个检测系统集成的核心。

#### 模型缓存 (ModelBundleCache)

```python
cache_key = (seat_model_id, camera_id, model_id, pipeline_signature, model_mtime_ns)
```

缓存键包含文件修改时间。当模型文件被替换时，自动重新加载，无需重启服务。

#### 特征提取器共享池 (PatchCorePredictorPool)

```python
class PatchCorePredictorPool:
    _feature_extractor_cache: dict[str, _TorchPatchFeatureExtractor]
    
    def predict_batch(self, items):
        # 1. 按特征提取器配置分组
        # 2. 同组内用 extract_many() 一次 batch forward
        # 3. 各 item 用各自 memory bank 独立打分
```

这是最核心的性能优化：

```
没有共享池:
  机位A PatchCore → 创建 ResNet50 → forward → 打分
  机位B PatchCore → 创建 ResNet50 → forward → 打分  ← 同样的 backbone 又跑一次!
  区域1 PatchCore → 创建 ResNet50 → forward → 打分  ← 又跑一次!
  
  内存: 3 个 ResNet50 副本
  推理: 3 次独立 forward

有共享池:
  机位A、机位B、区域1 都用 ResNet50
  → 只创建一个 ResNet50 实例
  → extract_many() 把 3 张图拼成 batch 一次 forward
  → 3 个各自 memory bank 独立打分
  
  内存: 1 个 ResNet50
  推理: 1 次 batch forward
```

**关键做法是 `set_feature_extractor()`**：把共享池里的 extractor 注入到每个 `PatchCoreService` 实例，这样它们调用 `predict()` 时就不创建自己的 backbone 了。

#### 预热机制

```python
def warmup(self, seat_model_id=None):
    # 遍历所有机位和区域
    # 用 dummy 数据跑一遍完整的特征提取 + 打分流程
    # 触发所有 CNN 模型的加载、JIT 编译、CUDA kernel 初始化
```

预热后，第一次真正的推理请求就不会有"加载模型"的延迟了。

### 4.6 service/inspection_camera.py — 区域模式

当相机配置了 `regions` 时，整个 ROI 被切分成多个子区域，每个区域有**独立的 PatchCore 模型**：

```
完整 ROI
  ├── 区域1 (左靠背) → 独立的 PatchCore 模型 A
  ├── 区域2 (右靠背) → 独立的 PatchCore 模型 B
  └── 区域3 (坐垫)   → 独立的 PatchCore 模型 C
```

**为什么需要区域模式？**

1. **不同纹理需要不同模型**：座椅的靠背可能是竖条纹，坐垫是菱形格。一个模型难以同时学好两种纹理。
2. **局部区域的缺陷更敏感**：如果你只关心坐垫是否有破损，用一个专门训练的小模型比全局模型更精确。
3. **灵活的阈值配置**：靠背可以容忍更多纹理变化（比如褶皱），坐垫需要更严格的缺陷判定。

#### 区域模式的批处理流程

```python
def build_region_patchcore_plan(service, frame_packet, camera, prepared, ...):
    # 1. 把 ROI 按配置切分成多个 RegionRoiSample
    region_samples = split_roi_regions(prepared.roi, camera.regions)
    
    # 2. 收集所有区域的 (patchcore_service, image, target_mask, ignore_mask)
    for region in camera.regions:
        model_bundle = service.load_region_model_bundle(camera, region, ...)
        patchcore_items.append((model_bundle.patchcore, region_sample.image, ...))
    
    # 3. 返回 RegionPatchCorePlan（延迟执行）
    return RegionPatchCorePlan(patchcore_items=patchcore_items, ...)
```

关键设计：`RegionPatchCorePlan` 是**延迟的**。不立即执行 PatchCore，而是把请求收集起来，等待跨机位的批量合并。

```python
# 在 inspection.py 中
def _finish_region_plans(service, plans, ordered_outputs):
    # 把所有 plan 的 patchcore_items 合并成一个大列表
    all_items = []
    for index, plan in plans:
        all_items.extend(plan.patchcore_items)
    
    # 一次批量预测，内部按 backbone 配置自动分组 + batch forward
    texture_results = service.predict_patchcore_batch(all_items)
```

这意味着如果有 3 个机位每个有 3 个区域，总共 9 个 PatchCore 请求：
- 如果它们都用同一个 backbone → 1 次 batch forward 处理 9 张子图
- 之前是 9 次独立 forward

---

## 5. 完整数据流

### 5.1 训练流程

```
工具层: train_good_dir/ 下的正常图片
  → 遍历文件夹 list_images()
  → 逐张图片经过 YOLO 检测 + ROI 精修 + 质量门控
  → 收集通过的 ROI 作为训练样本
  → 传入 PatchCoreTrainer.fit(samples)
```

```
核心层 fit():
  每个 ROI 样本 (image, target_mask, ignore_mask)
    → extract_patch_embeddings() 提取所有 patch 的 embedding
    → 收集所有 embedding 堆叠成大矩阵
    → 计算全局均值和标准差（用于归一化）
    → 归一化所有 embedding
    → coreset_subsample_indices() 精选 memory bank（默认取 10% 或最多 1024 个）
    → 留一法交叉验证: 每个样本用其余样本的 bank 打分
    → 三重保护计算阈值: max(分位数, 均值+3σ, 上界分位数)
    → 返回训练统计
```

```
工具层 save():
  → 保存为 .npz 文件:
      memory_bank.npy      # 精选的正常特征
      feature_mean.npy     # 均值
      feature_std.npy      # 标准差
      meta_json            # 所有配置参数的 JSON
      color_profile_json   # 颜色参考分布(可选)
      threshold            # 存在 meta 里
```

### 5.2 推理流程

```
外部图片 → InspectionService
  │
  ├── resolve_context() → 确定用哪个 seat_model 和哪些机位
  │
  ├── 对每个机位:
  │     ├── CameraPipeline.prepare_image()
  │     │     ├── YOLO 检测 → DetectionResult
  │     │     ├── ROI 精修 → RoiRefineResult
  │     │     └── 质量门控 → ImageQualityDecision
  │     │
  │     ├── 如果有 regions → build_region_patchcore_plan()
  │     │     └── 收集到全局批处理队列
  │     │
  │     └── 如果无 regions → 直接 PatchCore predict()
  │
  ├── predict_patchcore_batch(all_items)
  │     ├── 按 backbone 配置分组
  │     ├── 同组 extract_many() 一次 batch forward
  │     ├── 各自 predict_from_embeddings() 独立打分
  │     └── _decide_patchcore_anomaly() 判定
  │
  ├── _predict_color_branch() 颜色分支
  │
  ├── 融合每机位结果 → CameraInspectionResult
  │
  └── 多机位融合 → InspectionResult
```

---

## 6. 设计决策汇总表

| 设计决策 | 为什么这样做 | 不这样做的后果 |
|----------|-------------|---------------|
| 两层架构 (core/tool) | core 可独立部署，不带训练和相机依赖 | 产线部署包臃肿，CI/CD 慢 |
| Trainer 继承 Service | 训练和推理共享特征提取+打分逻辑 | 两套代码，训练时打分和推理时打分可能不一致 |
| predict 拆成 extract + predict_from_embeddings | 允许跨机位/跨区域的 batch backbone 共享 | 每个区域都要独立 forward，推理时间线性增长 |
| CNN backend 特征提取器共享池 | 多个机位共用 backbone 时只加载一次，多图 batch forward | GPU 内存爆炸(N份模型副本)，推理慢(N倍) |
| 管道签名 SHA256 校验 | 换 YOLO/ROI 参数后，旧 PatchCore 模型自动失效 | 静默地使用不匹配的模型，检测结果不可靠 |
| 三重阈值保护 (max of 3) | 防止训练集中的异常样本拉低阈值 | 阈值过低，缺陷漏检 |
| 三条判定规则 (normal + critical + peak) | 捕捉不同尺寸和严重程度的缺陷 | 要么大面积缺陷漏检，要么小尺寸缺陷漏检 |
| 运行时只允许收紧不允许放松 | 安全——产线调参不能比训练时更宽松 | 操作员误调导致缺陷漏检 |
| 颜色分支作为独立信号 | 纹理模型对纯颜色偏差不敏感 | 染色缺陷漏检 |
| 区域模式 | 不同纹理区域用专属模型 | 一个模型覆盖所有纹理，检测精度下降 |
| mtime 感知的模型缓存 | 模型文件更新后自动重载，无需重启 | 替换模型后要手动重启服务 |
| warmup 机制 | 提前加载模型和 GPU 编译 | 第一个请求延迟高（冷启动 ~数秒） |

---

## 7. 对"小白"的关键概念总结

### PatchCore 为什么适合工业缺陷检测？

- **只需要正常样本训练**：工业场景中，缺陷样本稀少且不可枚举，但正常样本唾手可得。
- **可解释**：输出热力图直接显示"哪里异常"，操作员可以直观理解判断依据。
- **无需标注**：不需要人工标注缺陷位置，只需要把正常图片放到一个文件夹。

### 这个封装解决了什么实际问题？

1. **产线需要快速推理** → batch backbone 共享 + GPU 加速 + 缓存 → 数十毫秒完成
2. **产线需要灵活调参** → 运行时配置覆写 → 不用重新训练就能调严/调松
3. **多条产线不同型号** → seat_models 机制 + 独立的 memory bank → 切换型号自动切换模型
4. **不同部位纹理不同** → 区域模式 → 每个区域专用模型
5. **模型更新后要自动生效** → mtime 缓存 → 替换文件即生效
6. **不会悄悄用错模型** → 管道签名校验 → 配置变更时自动报错而非静默失效
