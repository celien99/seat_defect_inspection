# Seat Defect Inspection 独立说明

`seat_defect_inspection` 是仓库内的独立子项目，已经和根项目 `seat_inspection` 解耦。

它面向汽车座椅缺陷检测，当前包含三条核心能力：

- 多机位采图
- 每机位独立 PatchCore 训练与推理
- YOLO 座椅定位模型训练

当前代码已经接入 `src/mvsCamera/` SDK，并支持 `mvs://...` 机位源。真机联调没有在这次改造里实测，后续可直接拿现场设备继续验证。

## 快速开始

```bash
cd seat_defect_inspection
conda create -n seat-defect-inspection python=3.10 -y
conda activate seat-defect-inspection
pip install -e .
seat-defect-inspection --help
```

也可以直接使用模块方式运行：

```bash
python -m seat_defect_inspection --help
```

## 主要命令

采集全部启用机位的一帧：

```bash
seat-defect-inspection capture \
  --config configs/seat_defect_inspection.mvs.json \
  --part-id seat_000001
```

把采集结果同时落到各机位 `train_good_dir`：

```bash
seat-defect-inspection capture \
  --config configs/seat_defect_inspection.mvs.json \
  --part-id seat_000001 \
  --save-to-train-good-dir
```

训练每个机位的 PatchCore：

```bash
seat-defect-inspection train-patchcore \
  --config configs/seat_defect_inspection.mvs.json
```

训练 YOLO：

```bash
seat-defect-inspection train-yolo \
  --config configs/seat_defect_inspection.mvs.json
```

执行一次完整检测：

```bash
seat-defect-inspection inspect \
  --config configs/seat_defect_inspection.mvs.json \
  --part-id seat_000001
```

离线图片文件夹批量检测：

```bash
seat-defect-inspection inspect-folder \
  --config configs/seat_defect_inspection.mvs.json \
  --input-dir offline_samples
```

## 推荐工作流

1. 先用 `capture` 采集正常样本。
2. 正常样本进入各机位 `train_good_dir`，再执行 `train-patchcore`。
   注意：`train_good_dir` 保存的是相机原图，真正训练 PatchCore 时仍会复用正式链路，先走整图 `preprocess`，再走 YOLO 掩膜驱动的 ROI 裁剪与有效区构造。
3. 单独准备 YOLO 数据集并执行 `train-yolo`。
4. 把每个机位的 `patchcore_model_path` 和 YOLO `model_path` 配好后，在线上跑 `inspect`，在线下批测跑 `inspect-folder`。

如果你已经训练过 PatchCore 模型，而最近又修改了 `preprocess`、ROI 掩膜、`valid_mask` 相关参数，或把 `patchcore.backend` 从 `full` 改成 `handcrafted`，必须重新执行 `train-patchcore`。旧模型对应的训练分布已经失效，继续拿来跑 `inspect` 没有参考价值。

如果现场还没有 YOLO 权重，可以先把 `detection.model_path` 设为 `null`，继续使用 `fallback_box` 走完整流程。

项目现在统一使用 `yolo11m-seg.pt`。代码会直接消费 YOLO segmentation mask 来裁剪目标区域，并构造 `target_mask / valid_mask`。当前 ROI 层已经压成轻量链路，主要保留裁剪、缩放和掩膜清理，不再承载之前那套重的局部增强流程。

`train-patchcore` 本身就是离线训练流程。只要每个机位的 `train_good_dir` 指向本地图片目录，就不需要连接真机。

`inspect-folder` 是新增的离线批量验证命令。它复用线上 `inspect` 的同一套 `preprocess -> YOLO -> ROI -> PatchCore -> fusion -> report` 链路，只是把机位输入从真机切换成图片文件夹。

`inspect-folder` 目前支持三种输入组织方式：

1. 单样本目录：
   根目录直接放每个机位一张图，例如 `cam_0.jpg`、`cam_1.jpg`。
2. 按样本分目录：
   `offline_samples/sample_001/cam_0.jpg`、`offline_samples/sample_001/cam_1.jpg`。
3. 按机位分目录：
   `offline_samples/cam_0/sample_001.jpg`、`offline_samples/cam_1/sample_001.jpg`。

## 目录约定

- `data/seat_defect_inspection/<camera_id>/train/good`
  PatchCore 正常样本目录
- `models/seat_defect_inspection/<camera_id>_patchcore.npz`
  每个机位独立 PatchCore 模型
- `outputs/seat_defect_inspection/capture`
  `capture` 命令采图输出
- `outputs/seat_defect_inspection/debug`
  `inspect` 过程中的调试图输出
- `<output_json_path 同目录>/<output_json_path.stem>_history`
  `inspect` 历史结果归档目录；`output_json_path` 继续保留为最新结果
- `outputs/seat_defect_inspection/yolo_training`
  YOLO 训练输出

## 当前文件结构

现在的拆分原则是“主入口只做编排，细节按功能放到同包文件中”，不再把主流程、ROI、PatchCore、YOLO 训练和配置解析都堆在几个超大文件里。

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
        │   ├── __init__.py
        │   ├── debug_artifacts.py
        │   ├── quality.py
        │   ├── roi.py
        │   └── roi_geometry.py
        ├── preprocess/
        │   ├── __init__.py
        │   └── engine.py
        ├── patchcore/
        │   ├── __init__.py
        │   ├── color_branch.py
        │   ├── engine.py
        │   ├── features.py
        │   └── scoring.py
        ├── service/
        │   ├── __init__.py
        │   ├── capture.py
        │   ├── core.py
        │   ├── inspection.py
        │   ├── inspection_camera.py
        │   ├── offline_inspection.py
        │   └── training.py
        └── yolo/
            ├── __init__.py
            ├── dataset_validation.py
            ├── detection.py
            ├── labelme_to_yolo.py
            └── training.py
```

## 模块职责

- `cli.py`
  命令行薄入口，只负责组装子命令解析树。
- `cli_commands/`
  按命令拆分的入口目录；每个文件各自维护参数注册和业务路由，不再把所有指令堆在 `cli.py`。
- `runtime_config.py`
  配置文件入口和顶层校验。
- `runtime_config_parsers.py`
  主配置、型号配置、融合配置、YOLO 训练配置解析。
- `runtime_config_camera_parsers.py`
  相机子配置解析，覆盖质量、预处理、检测、ROI、PatchCore、颜色分支。
- `runtime_config_values.py`
  通用字段读取、类型转换和路径解析小工具。
- `service/__init__.py`
  对外主流程入口，只负责路由到采图、检测、训练模块。
- `service/core.py`
  `InspectionService`、上下文缓存和 `_CameraPipeline`。
- `service/capture.py`
  多机位采图流程。
- `service/inspection.py`
  多机位检测编排、fail-fast 和最终结果落盘。
- `service/inspection_camera.py`
  单机位完整检测细节，包括 PatchCore、颜色分支和调试图挂载。
- `service/offline_inspection.py`
  离线图片文件夹批量检测流程，复用现有检测主链，只替换输入源。
- `service/training.py`
  PatchCore 训练流程。
- `cvops/`
  OpenCV 中间层，负责质量门控、ROI 精修、几何辅助、纹理准备和调试产物保存。
- `preprocess/engine.py`
  预处理链路，负责去噪、白平衡、光照校正、CLAHE、锐化等。
- `patchcore/engine.py`
  PatchCore 主流程编排，负责训练、推理、模型保存和加载。
- `patchcore/features.py`
  特征提取细节，包含 `handcrafted` 和 `full` 两种后端。
- `patchcore/scoring.py`
  记忆库采样、最近邻距离、证据分析和最终判定规则。
- `patchcore/color_branch.py`
  LAB 统计量颜色分支。
- `yolo/detection.py`
  YOLO 检测与 `fallback_box` 兜底。
- `yolo/training.py`
  YOLO 训练入口。
- `yolo/dataset_validation.py`
  数据集预检和标签格式校验。
- `util.py`
  公共小工具，例如 PatchCore 输入选择、JSON/图像写盘等。

## 主流程调用关系

当前 `inspect` 主流程已经压成“薄编排 + 细节下沉”的结构：

1. `cli.py` 组装命令树，`cli_commands/inspect.py` 加载配置并路由到 `service.run_inspection`
2. `service/__init__.py` 创建 `InspectionService`
3. `service/inspection.py` 负责多机位循环、采图异常处理、fail-fast 和最终融合
4. `service/inspection_camera.py` 负责单机位准备、PatchCore、颜色分支和调试图保存
5. `fusion.py` 负责多机位结果融合
6. `reporting.py` 负责结果落盘

离线批测则走同一套检测链，只是入口换成 `cli_commands/inspect_folder.py`，并由 `service/offline_inspection.py` 负责把图片文件夹解析成每个样本的机位输入。

训练流程也是同样思路：

1. `cli.py` 组装命令树，`cli_commands/train_patchcore.py` 路由到 `service.train_patchcore_models`
2. `service/training.py` 负责遍历型号与机位
3. `service/core.py` 里的 `_CameraPipeline.prepare_image` 复用线上链路
4. `patchcore/engine.py` + `patchcore/features.py` + `patchcore/scoring.py` 完成 PatchCore 训练
5. `patchcore/color_branch.py` 按需补充颜色参考分布

## 配置文件

- `configs/seat_defect_inspection.mvs.json`
  当前 5 路 MVS 相机配置示例，已默认开启颜色不敏感模式
- `configs/seat_defect_inspection.multimodel.example.json`
  多型号路由配置示例，适合一个项目里维护多个座椅型号
- `configs/seat_defect_yolo.dataset.example.yaml`
  YOLO 数据集配置示例

说明：

- 当前 JSON 中大部分路径都是相对配置文件目录解析的；放在 `configs/` 下的配置文件如果要指向仓库根目录，推荐使用 `../models/...`、`../outputs/...`、`../data/...` 这种写法。
- 现在配置解析链已经拆开：入口校验在 `runtime_config.py`，主解析在 `runtime_config_parsers.py`，相机子配置解析在 `runtime_config_camera_parsers.py`，公共小工具在 `runtime_config_values.py`。

## 更多架构细节

如果需要看更细的模块说明、主流程时序、关键缓存和阅读路径，请继续看 [PROJECT_ARCHITECTURE_ZH.md](./PROJECT_ARCHITECTURE_ZH.md)。
