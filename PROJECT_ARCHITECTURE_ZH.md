# Seat Defect Inspection 架构说明

本文档用于同步当前 `seat_defect_inspection` 项目在再次拆分后的真实结构，重点说明：

- 现在的目录与模块职责
- `capture / inspect / inspect-folder / train-patchcore / train-yolo` 五条主流程
- 主流程拆分后哪些文件负责“编排”，哪些文件负责“细节”
- 当前关键缓存与维护建议

## 1. 当前拆分原则

本轮代码调整遵循的是“做减法”，不是“做花活”：

1. 主入口文件只保留编排，不堆实现细节。
2. 配置解析、YOLO 训练、ROI、PatchCore、单机位检测都按功能拆到各自包内。
3. 不引入 manager / factory / registry 之类额外抽象。
4. 尽量维持对外 API 不变，例如：
   - `seat_defect_inspection.capture_samples`
   - `seat_defect_inspection.run_inspection`
   - `seat_defect_inspection.train_patchcore_models`
   - `seat_defect_inspection.train_yolo_model`

## 2. 顶层结构

```text
seat_defect_inspection/
├── PROJECT_ARCHITECTURE_ZH.md
├── README.md
├── README_ZH.md
├── configs/
│   ├── seat_defect_inspection.mvs.json
│   ├── seat_defect_inspection.multimodel.example.json
│   └── seat_defect_yolo.dataset.example.yaml
└── src/
    ├── media_inputs/
    ├── mvsCamera/
    └── seat_defect_inspection/
        ├── __init__.py
        ├── __main__.py
        ├── cli.py
        ├── cli_commands/
        │   ├── __init__.py
        │   ├── common.py
        │   ├── capture.py
        │   ├── inspect.py
        │   ├── inspect_folder.py
        │   ├── train_patchcore.py
        │   └── train_yolo.py
        ├── acquisition.py
        ├── config.py
        ├── debug_artifacts.py
        ├── fusion.py
        ├── reporting.py
        ├── runtime_config.py
        ├── runtime_config_parsers.py
        ├── runtime_config_camera_parsers.py
        ├── runtime_config_values.py
        ├── schemas.py
        ├── util.py
        ├── cvops/
        ├── preprocess/
        ├── patchcore/
        ├── service/
        │   ├── __init__.py
        │   ├── capture.py
        │   ├── core.py
        │   ├── inspection.py
        │   ├── inspection_camera.py
        │   ├── offline_inspection.py
        │   └── training.py
        └── yolo/
```

## 3. 模块分层

| 分层 | 文件/目录 | 作用 |
| --- | --- | --- |
| 命令入口层 | `cli.py` `cli_commands/` | `cli.py` 只组装命令树，每个子命令分别在独立文件里维护 |
| 配置层 | `config.py` `runtime_config.py` `runtime_config_parsers.py` `runtime_config_camera_parsers.py` `runtime_config_values.py` | dataclass 定义、配置加载、字段解析、路径解析、顶层校验 |
| 采图层 | `acquisition.py` `media_inputs/` `mvsCamera/` | 统一图片、视频、普通摄像头、MVS 相机输入 |
| OpenCV 中间层 | `cvops/` `preprocess/` | 图像质量门控、OpenCV 预处理、ROI 精修、纹理准备、调试图输出 |
| 检测层 | `yolo/detection.py` | YOLO 目标/忽略区检测与 fallback box |
| 异常检测层 | `patchcore/` | PatchCore 训练/推理、特征提取、打分判定、颜色分支 |
| 主流程编排层 | `service/` | 采图、训练、在线检测、离线批测四条业务主链路 |
| 融合与输出层 | `fusion.py` `reporting.py` | 多机位结果融合与落盘 |
| 公共结构/工具 | `schemas.py` `util.py` | 流程数据结构和通用辅助函数 |

## 4. 各目录职责

### 4.1 `service/`

这是主流程目录，现在已经拆成几块清晰职责：

- `service/__init__.py`
  对外路由层。只负责 new `InspectionService(config)` 再转发到对应流程文件。
- `service/core.py`
  共享骨架。包含：
  - `InspectionService`
  - `_ResolvedInspectionContext`
  - `_CameraPipeline`
  - `PreparedCameraSample`
- `service/capture.py`
  采图流程。
- `service/inspection.py`
  多机位检测编排，负责：
  - 遍历机位
  - 采图异常兜底
  - fail-fast
  - 最终融合和落盘
- `service/inspection_camera.py`
  单机位检测细节，负责：
  - `_CameraPipeline.prepare_image`
  - PatchCore 推理
  - 颜色分支推理
  - 调试图挂载
  - REJECT 结果构造
- `service/offline_inspection.py`
  离线图片文件夹检测流程，负责：
  - 识别输入目录布局
  - 为每个样本绑定各机位图片
  - 复用 `run_inspection` 批量跑完整检测链
  - 输出批量汇总 `summary.json`
- `service/training.py`
  PatchCore 训练流程。

### 4.2 `cvops/`

这是 OpenCV 中间层：

- `cvops/quality.py`
  图像质量门控。
- `cvops/roi.py`
  ROI 主流程编排。
- `cvops/roi_geometry.py`
  ROI 裁剪、框扩张、掩膜裁切等几何/掩膜辅助。
- `cvops/debug_artifacts.py`
  调试图生成与保存细节。

### 4.3 `preprocess/`

- `preprocess/engine.py`
  预处理链路，负责 resize、去噪、白平衡、光照校正、CLAHE、锐化等。

### 4.4 `patchcore/`

PatchCore 现在不再堆在一个文件里：

- `patchcore/engine.py`
  PatchCore 生命周期编排：
  - 训练
  - 推理
  - 模型保存/加载
- `patchcore/features.py`
  特征提取细节：
  - `handcrafted` 后端
  - `full` backbone 后端
  - patch embedding 提取
- `patchcore/scoring.py`
  记忆库采样和打分逻辑：
  - coreset
  - 最近邻距离
  - leave-one-out 校准
  - 强证据统计
  - 最终判定规则
- `patchcore/color_branch.py`
  LAB 颜色一致性分支。

### 4.5 `yolo/`

- `yolo/__init__.py`
  延迟导出，避免普通命令被训练依赖提前拖上。
- `yolo/detection.py`
  YOLO 推理和 fallback box。
- `yolo/training.py`
  YOLO 训练入口。
- `yolo/dataset_validation.py`
  训练前的数据集与标签校验。
- `yolo/labelme_to_yolo.py`
  标注格式转换。

### 4.6 配置解析链

配置链现在也拆开了：

- `runtime_config.py`
  入口和顶层校验。
- `runtime_config_parsers.py`
  主配置、型号配置、融合配置、YOLO 训练配置解析。
- `runtime_config_camera_parsers.py`
  相机子配置解析。
- `runtime_config_values.py`
  通用字段和路径解析小工具。

## 5. 关键文件与核心符号

### 5.1 入口层

文件：`src/seat_defect_inspection/cli.py` `src/seat_defect_inspection/cli_commands/*.py`

| 符号 | 作用 |
| --- | --- |
| `build_parser` | 构建 CLI 命令树 |
| `register_*_command` | 各子命令的参数注册入口 |
| `run_capture_command` | `capture` 命令入口 |
| `run_train_patchcore_command` | `train-patchcore` 命令入口 |
| `run_train_yolo_command` | `train-yolo` 命令入口 |
| `run_inspect_command` | `inspect` 命令入口 |
| `run_inspect_folder_command` | `inspect-folder` 离线批测入口 |
| `main` | 程序入口 |

### 5.2 配置层

文件：`src/seat_defect_inspection/runtime_config.py`

| 符号 | 作用 |
| --- | --- |
| `load_config` | 加载缺陷检测主配置 |
| `load_yolo_training_config` | 加载 YOLO 训练配置 |
| `_validate_inspection_config` | 顶层配置校验 |
| `_validate_camera_configs` | 检查重复 `camera_id` 与 PatchCore 后端约束 |

文件：`src/seat_defect_inspection/runtime_config_parsers.py`

| 符号 | 作用 |
| --- | --- |
| `_parse_inspection_config` | 解析主配置 |
| `_parse_seat_model_config` | 解析型号配置 |
| `_parse_fusion_config` | 解析融合配置 |
| `_parse_yolo_training_config` | 解析 YOLO 训练配置 |
| `_resolve_yolo_training_payload` | 根据 `seat_model_id` 选择训练块 |

文件：`src/seat_defect_inspection/runtime_config_camera_parsers.py`

| 符号 | 作用 |
| --- | --- |
| `_parse_camera_config` | 解析单机位总配置 |
| `_parse_quality_guard_config` | 解析质量门控配置 |
| `_parse_preprocess_config` | 解析预处理配置 |
| `_parse_alignment_config` | 解析对齐配置 |
| `_parse_roi_refine_config` | 解析 ROI 配置 |
| `_parse_detection_config` | 解析 YOLO 检测配置 |
| `_parse_patchcore_config` | 解析 PatchCore 配置 |
| `_parse_color_branch_config` | 解析颜色分支配置 |

### 5.3 采图层

文件：`src/seat_defect_inspection/acquisition.py`

| 符号 | 作用 |
| --- | --- |
| `AcquisitionService` | 统一媒体源采图服务 |
| `AcquisitionService.capture` | 按机位抓一帧并返回 `FramePacket` |

### 5.4 OpenCV 中间层

文件：`src/seat_defect_inspection/cvops/quality.py`

| 符号 | 作用 |
| --- | --- |
| `ImageQualityGuard` | 模糊、亮度、过曝、欠曝门控 |

文件：`src/seat_defect_inspection/preprocess/engine.py`

| 符号 | 作用 |
| --- | --- |
| `PreprocessEngine` | 图像预处理主链路 |

文件：`src/seat_defect_inspection/cvops/roi.py`

| 符号 | 作用 |
| --- | --- |
| `RoiRefineEngine` | ROI 精修主流程 |

文件：`src/seat_defect_inspection/cvops/roi_geometry.py`

| 符号 | 作用 |
| --- | --- |
| `_expand_box` | 扩框/缩框 |
| `_crop_mask` | 掩膜裁切 |
| `_resolve_crop_source_box` | 优先根据分割掩膜确定裁剪范围 |

### 5.5 YOLO 层

文件：`src/seat_defect_inspection/yolo/detection.py`

| 符号 | 作用 |
| --- | --- |
| `DetectionService` | YOLO 推理服务 |
| `DetectionService.detect` | 返回 `DetectionResult` |

文件：`src/seat_defect_inspection/yolo/training.py`

| 符号 | 作用 |
| --- | --- |
| `train_yolo_model` | YOLO 训练入口 |
| `_load_yolo_model` | 加载/兜底初始化分割模型 |

文件：`src/seat_defect_inspection/yolo/dataset_validation.py`

| 符号 | 作用 |
| --- | --- |
| `_prepare_training_dataset` | 数据集解析与预检 |
| `_validate_dataset_split` | train/val 切分校验 |
| `_validate_label_line` | 单行标签校验 |

### 5.6 PatchCore 层

文件：`src/seat_defect_inspection/patchcore/engine.py`

| 符号 | 作用 |
| --- | --- |
| `LoadedModelBundle` | PatchCore 模型包 |
| `PatchCoreService` | PatchCore 训练/推理主类 |
| `PatchCoreService.fit` | 训练模型 |
| `PatchCoreService.predict` | ROI 推理 |
| `PatchCoreService.save` | 保存模型 |
| `PatchCoreService.load_bundle` | 加载模型 |

文件：`src/seat_defect_inspection/patchcore/features.py`

| 符号 | 作用 |
| --- | --- |
| `extract_patch_embeddings` | 统一提取 patch embedding |
| `extract_handcrafted_patch_embeddings` | 轻量手工特征后端 |
| `_TorchPatchFeatureExtractor` | 完整 CNN 特征后端 |

文件：`src/seat_defect_inspection/patchcore/scoring.py`

| 符号 | 作用 |
| --- | --- |
| `coreset_subsample_indices` | 记忆库采样 |
| `min_distance_to_bank` | 最近邻距离计算 |
| `_score_embeddings_leave_one_out` | 训练期 LOOCV 校准 |
| `_analyze_patch_evidence` | 强 patch 证据统计 |
| `_decide_patchcore_anomaly` | 最终异常判定 |

文件：`src/seat_defect_inspection/patchcore/color_branch.py`

| 符号 | 作用 |
| --- | --- |
| `ColorReferenceProfile` | 颜色正常分布 |
| `ColorConsistencyService` | 颜色分支训练/推理 |

### 5.7 主流程编排层

文件：`src/seat_defect_inspection/service/core.py`

| 符号 | 作用 |
| --- | --- |
| `PreparedCameraSample` | 单机位共享中间结果 |
| `_ResolvedInspectionContext` | 当前型号上下文 |
| `_CameraPipeline` | 单机位预处理、检测、ROI 精修链 |
| `_CameraPipeline.prepare_image` | 线上/训练共用图像准备入口 |
| `InspectionService` | 总编排服务 |
| `InspectionService._resolve_context` | 解析型号路由与流程缓存 |
| `InspectionService._resolve_active_cameras` | 确定当前启用机位 |
| `InspectionService._build_patchcore_service` | 创建 PatchCore 服务 |
| `InspectionService._load_model_bundle` | 加载模型包 |

文件：`src/seat_defect_inspection/service/capture.py`

| 符号 | 作用 |
| --- | --- |
| `capture_samples` | 多机位采图主流程 |

文件：`src/seat_defect_inspection/service/inspection.py`

| 符号 | 作用 |
| --- | --- |
| `run_inspection` | 多机位检测编排 |
| `_build_exported_early_stop_result` | fail-fast 统一出口 |

文件：`src/seat_defect_inspection/service/inspection_camera.py`

| 符号 | 作用 |
| --- | --- |
| `_inspect_one_camera` | 单机位完整检测 |
| `_attach_debug_artifacts` | 挂载调试图路径 |
| `_build_reject_result` | 构造 REJECT 结果 |
| `_build_capture_failed_result` | 构造采图失败结果 |

文件：`src/seat_defect_inspection/service/training.py`

| 符号 | 作用 |
| --- | --- |
| `train_patchcore_models` | 按型号/机位训练 PatchCore |
| `_train_one_camera` | 单机位训练流程 |

文件：`src/seat_defect_inspection/service/offline_inspection.py`

| 符号 | 作用 |
| --- | --- |
| `inspect_image_folder` | 离线图片文件夹批量检测主流程 |
| `_discover_offline_samples` | 自动识别目录布局并解析样本 |
| `_discover_camera_layout_samples` | 解析“按机位分目录”布局 |

## 6. 五条主流程

### 6.1 `capture`

```text
cli.main
  -> cli_commands/capture.py:run_capture_command
  -> runtime_config.load_config
  -> service.capture_samples
  -> InspectionService(config)
  -> service/capture.py:capture_samples
```

内部顺序：

1. `_resolve_context` 选择当前型号和启用机位。
2. 循环机位调用 `AcquisitionService.capture`。
3. 把 `FramePacket` 写到采图目录。
4. 如启用 `save_to_train_good_dir`，同步写入训练目录。
5. 用 `export_capture_manifest` 输出 `manifest.json`。

### 6.2 `inspect`

```text
cli.main
  -> cli_commands/inspect.py:run_inspect_command
  -> runtime_config.load_config
  -> service.run_inspection
  -> InspectionService(config)
  -> service/inspection.py:run_inspection
```

内部顺序：

1. `_resolve_context` 选择当前型号和机位流程缓存。
2. 逐机位采图。
3. 每张图交给 `service/inspection_camera.py:_inspect_one_camera`。
4. `_inspect_one_camera` 内部顺序：
   - `_CameraPipeline.prepare_image`
   - `_load_model_bundle`
   - `PatchCoreService.predict`
   - 可选 `ColorConsistencyService.predict`
   - 保存调试图
   - 生成 `CameraInspectionResult`
5. `service/inspection.py` 处理 fail-fast。
6. 全部机位完成后 `fuse_camera_results`。
7. `export_inspection_report` 输出最终结果。

### 6.3 `train-patchcore`

```text
cli.main
  -> cli_commands/train_patchcore.py:run_train_patchcore_command
  -> runtime_config.load_config
  -> service.train_patchcore_models
  -> InspectionService(config)
  -> service/training.py:train_patchcore_models
```

内部顺序：

1. `_resolve_training_scope` 决定训练哪些型号。
2. 对每个型号 `_resolve_context`。
3. 每个机位读取 `train_good_dir` 图片。
4. 每张图复用 `_CameraPipeline.prepare_image`，保证训练和推理链路一致。
5. 成功样本送入 `PatchCoreService.fit`。
6. 如启用颜色分支，再执行 `ColorConsistencyService.fit`。
7. 保存 `.npz` 模型和 `.summary.json` 摘要。

### 6.4 `train-yolo`

```text
cli.main
  -> cli_commands/train_yolo.py:run_train_yolo_command
  -> runtime_config.load_yolo_training_config
  -> yolo.train_yolo_model
  -> yolo/training.py:train_yolo_model
```

内部顺序：

1. `load_yolo_training_config` 解析训练块。
2. `yolo/dataset_validation.py` 预检数据集和标签。
3. `yolo/training.py` 加载 Ultralytics 模型。
4. 调用 `model.train(...)`。
5. 输出 `best.pt / last.pt / training_summary.json`。

### 6.5 `inspect-folder`

```text
cli.main
  -> cli_commands/inspect_folder.py:run_inspect_folder_command
  -> runtime_config.load_config
  -> service.inspect_image_folder
  -> InspectionService(config)
  -> service/offline_inspection.py:inspect_image_folder
  -> service/inspection.py:run_inspection
```

内部顺序：

1. 先解析当前型号下的启用机位列表。
2. 自动识别输入目录是单样本、按样本分目录，还是按机位分目录。
3. 为每个离线样本绑定各机位图片路径。
4. 把图片路径临时写回各机位 `source`，复用现有 `run_inspection` 主流程。
5. 每个样本仍然走 `preprocess -> YOLO -> ROI -> PatchCore -> fusion -> report`。
6. 额外输出一次批量汇总 `summary.json`。

## 7. 当前最关键的几个入口

如果要快速读懂项目，优先看下面这些位置：

| 优先级 | 文件 | 符号 | 原因 |
| --- | --- | --- | --- |
| 1 | `cli.py` `cli_commands/` | `main`、`register_*_command`、`run_*_command` | 所有命令都从这里进 |
| 2 | `runtime_config.py` | `load_config` | 配置到运行对象的第一入口 |
| 3 | `service/core.py` | `InspectionService` | 总编排服务和共享缓存都在这里 |
| 4 | `service/core.py` | `_CameraPipeline.prepare_image` | 线上/训练共用的单机位图像链路 |
| 5 | `service/inspection.py` | `run_inspection` | 多机位检测主编排 |
| 6 | `service/inspection_camera.py` | `_inspect_one_camera` | 单机位完整检测细节 |
| 7 | `cvops/roi.py` | `RoiRefineEngine.refine` | ROI 和 PatchCore 输入如何构造 |
| 8 | `patchcore/engine.py` | `PatchCoreService.fit/predict` | 异常检测主入口 |

## 8. 关键缓存与隐式状态

| 位置 | 变量 | 作用 |
| --- | --- | --- |
| `InspectionService` | `_pipeline_cache` | 缓存每个型号下的 `_CameraPipeline` |
| `InspectionService` | `_model_cache` | 缓存 `(seat_model_id, camera_id)` 对应模型包 |
| `DetectionService` | `_model` | 延迟加载 YOLO 模型 |
| `media_inputs` / `mvsCamera` | 流对象与 SDK 状态 | 统一媒体源和工业相机资源管理 |

这些状态说明：

1. 当前默认是“单进程复用服务实例”的写法。
2. 如果以后要服务化或多进程化，需要重新评估缓存和资源释放。

## 9. 建议阅读顺序

### 9.1 想快速读懂整体结构

1. `src/seat_defect_inspection/cli.py`
2. `src/seat_defect_inspection/cli_commands/inspect.py`
3. `src/seat_defect_inspection/runtime_config.py`
4. `src/seat_defect_inspection/service/core.py`
5. `src/seat_defect_inspection/service/inspection.py`
6. `src/seat_defect_inspection/service/inspection_camera.py`
7. `src/seat_defect_inspection/cvops/roi.py`
8. `src/seat_defect_inspection/patchcore/engine.py`

### 9.2 现场排查时按问题读

采图失败：

1. `src/seat_defect_inspection/acquisition.py`
2. `src/media_inputs/core.py`
3. `src/mvsCamera/frame_source.py`
4. `src/mvsCamera/camera_controller.py`

YOLO 检不准：

1. `src/seat_defect_inspection/cvops/quality.py`
2. `src/seat_defect_inspection/preprocess/engine.py`
3. `src/seat_defect_inspection/yolo/detection.py`

ROI 或 PatchCore 不稳定：

1. `src/seat_defect_inspection/cvops/roi.py`
2. `src/seat_defect_inspection/cvops/roi_geometry.py`
3. `src/seat_defect_inspection/patchcore/engine.py`
4. `src/seat_defect_inspection/patchcore/scoring.py`

训练和线上结果不一致：

1. `src/seat_defect_inspection/service/training.py`
2. `src/seat_defect_inspection/service/core.py`
3. `src/seat_defect_inspection/patchcore/features.py`
4. `src/seat_defect_inspection/patchcore/engine.py`

## 10. 当前结论

经过这轮拆分后，当前结构相比最早版本有几个明显变化：

1. 主流程不再堆在一个大 `service.py` 里，而是拆成 `core / capture / inspection / inspection_camera / training`。
2. PatchCore 不再堆在一个大文件里，而是拆成 `engine / features / scoring / color_branch`。
3. 配置解析不再混在一个大文件里，而是拆成 `runtime_config / runtime_config_parsers / runtime_config_camera_parsers / runtime_config_values`。
4. YOLO 训练也拆出了独立的数据集校验文件 `dataset_validation.py`。
5. `cli.py`、`cli_commands/`、`service/__init__.py`、`yolo/__init__.py` 都尽量保持为薄入口，并通过按职责拆分降低不必要耦合。

这套结构更接近“按功能分文件、入口只做编排”的维护方式，后续继续改 ROI、PatchCore、YOLO 训练或配置解析时，影响面会更集中，也更容易做减法。
