# seat_defect_core 使用说明

本文档描述 `seat_defect_core` 作为外部 Python 项目检测 SDK 时的接入方式。`core` 只负责检测运行时，不负责相机采集、模型训练或 UI 展示。

## 适用边界

`seat_defect_core` 支持的能力：

- 加载检测配置文件。
- 加载训练好的 YOLO 分割模型和 PatchCore 模型。
- 接收外部项目传入的多机位图像。
- 执行 YOLO、ROI 对齐、PatchCore、region PatchCore、颜色分支和多机位融合。
- 返回结构化检测结果、错误码、耗时和报告路径。

`seat_defect_core` 不负责的能力：

- 不直接控制工业相机。
- 不负责模型训练数据采集。
- 不负责长期结果数据库存储。

## 安装和交付

推荐用 Python 包方式交付：

```bash
pip install /path/to/seat_defect_core_package
```

LabVIEW 公共机推荐使用独立 Python 3.8.5 CPU 环境：

```bash
conda create -n seat-defect-core-py38 python=3.8.5 -y
conda activate seat-defect-core-py38
pip install -r requirements-core-py38-cpu.txt
pip install --no-build-isolation /path/to/seat_defect_core_package
```

也可以在同一工程中通过源码方式使用，但需要保证：

- Python 版本 `>=3.8.5`。
- 依赖已安装并固定到公共机验证过的版本。CPU 运行时推荐使用 `requirements-core-py38-cpu.txt`。
- `seat_defect_core` 能被 Python import 到。
- 配置文件中的模型路径能被当前 Python 环境访问。
- `output_json_path` 和 `debug_dir` 指向 LabVIEW 进程可写目录。

不建议长期依赖手工复制目录作为正式交付方式。手工复制可以用于临时验证，但容易遗漏依赖、版本和包数据。

离线安装时，先在可联网的 Python 3.8.5 机器上准备 wheel 缓存：

```bash
python -m pip download --only-binary=:all: -r requirements-core-py38-cpu.txt -d wheelhouse
python -m pip wheel --no-deps --no-build-isolation /path/to/seat_defect_core_package -w wheelhouse
```

拷贝 `wheelhouse` 到 LabVIEW 公共机后离线安装：

```bash
python -m pip install --no-index --find-links wheelhouse -r requirements-core-py38-cpu.txt
python -m pip install --no-index --find-links wheelhouse seat-defect-core
```

## 最小调用示例

外部项目传入每个机位对应的图片路径：

```python
from seat_defect_core import SeatDefectInspector

inspector = SeatDefectInspector("/path/to/inspection_config.json")

response = inspector.inspect_paths(
    {
        "cam_front": "/data/current/cam_front.png",
        "cam_left": "/data/current/cam_left.png",
    },
    part_id="part_20260509_0001",
    seat_model_id="seat_model_a",
    frame_id="frame_0001",
    timestamp="2026-05-09T10:00:00+08:00",
)

payload = response.to_dict()
print(payload["status"])
print(payload["decision_reason"])
```

如果只运行一次，也可以使用函数入口：

```python
from seat_defect_core import inspect_paths_once

response = inspect_paths_once(
    "/path/to/inspection_config.json",
    {
        "cam_front": "/data/current/cam_front.png",
    },
    part_id="part_20260509_0001",
    seat_model_id="seat_model_a",
)
```

如果外部项目已经拿到了 `numpy.ndarray` 图像，可以直接传 frame：

```python
from seat_defect_core import SeatDefectInspector

inspector = SeatDefectInspector("/path/to/inspection_config.json")

response = inspector.inspect(
    [
        {
            "camera_id": "cam_front",
            "image": image_bgr,
            "source": "external://cam_front",
            "source_kind": "external_image",
            "frame_id": "frame_0001",
            "timestamp": "2026-05-09T10:00:00+08:00",
        }
    ],
    part_id="part_20260509_0001",
    seat_model_id="seat_model_a",
)
```

图像数组要求：

- OpenCV BGR 格式优先。
- 类型通常为 `uint8`。
- 不要传已经被外部裁剪、旋转或压缩破坏的图像，除非模型训练时使用的就是同样流程。

## 配置文件

配置文件支持 JSON 和 INI。JSON 可以直接是检测配置对象，也可以包在 `seat_defect_inspection` 顶层字段下。路径类字段会按配置文件所在目录解析相对路径。

INI 用于兼容 LabVIEW 和现场工具，核心流程仍会先把 INI 转成同一份配置 payload，再走统一校验。常用 section 约定如下：

- `[seat_defect_inspection]`：顶层路径、开关、默认工件等字段
- `[fusion]`：整件融合策略
- `[camera.<camera_id>]`：顶层单机位
- `[camera.<camera_id>.detection]`、`roi`、`roi.alignment`、`patchcore`、`color_branch`
- `[camera.<camera_id>.region.<region_id>]`：单机位局部区域
- `[seat_model.<seat_model_id>]` 和 `[seat_model.<seat_model_id>.camera.<camera_id>]`：多型号配置

示例：

```json
{
  "seat_defect_inspection": {
    "part_id": "seat_demo",
    "default_seat_model_id": "seat_model_a",
    "output_json_path": "../outputs/seat_defect_inspection/results.json",
    "debug_dir": "../outputs/seat_defect_inspection/debug",
    "debug_artifacts_enabled": false,
    "fusion": {
      "reject_on_any_reject": true,
      "ng_strategy": "any",
      "defect_overrides_reject": true
    },
    "seat_models": [
      {
        "seat_model_id": "seat_model_a",
        "display_name": "座椅型号A",
        "cameras": [
          {
            "camera_id": "cam_front",
            "patchcore_model_path": "../models/seat_model_a/cam_front_patchcore.npz",
            "color_insensitive_mode": true,
            "detection": {
              "model_path": "../models/yolo/seat_model_a_best.pt",
              "target_class": "seat",
              "confidence": 0.25,
              "iou": 0.45,
              "imgsz": 960
            },
            "roi": {
              "crop_expand_ratio": 0.02,
              "mask_erode_pixels": 1,
              "edge_ignore_pixels": 4,
              "alignment": {
                "output_width": 256,
                "output_height": 256
              }
            },
            "patchcore": {
              "backend": "full",
              "backbone_name": "wide_resnet50_2",
              "feature_layers": ["layer2", "layer3"],
              "backbone_pretrained": true,
              "backbone_device": "cpu",
              "texture_input": "lab_l",
              "min_target_coverage": 0.6,
              "min_valid_patch_ratio": 0.4
            },
            "regions": [
              {
                "region_id": "upper",
                "box": [0.03, 0.03, 0.97, 0.42],
                "patchcore_model_path": "../models/seat_model_a/cam_front_upper_patchcore.npz"
              }
            ]
          }
        ]
      }
    ]
  }
}
```

生产环境建议：

- `debug_artifacts_enabled` 设置为 `false`，避免保存大量调试图片拖慢检测。
- `output_json_path` 和 `debug_dir` 放到外部项目可写目录。
- `backbone_device` 根据现场硬件设为 `cpu`、`cuda:0` 或 `mps`。
- 若现场不能联网下载 torchvision 权重，配置 `backbone_weights_path` 指向本地预训练权重，或提前准备 `.torch_cache`。

## 输入格式

### `inspect_paths`

```python
inspect_paths(
    image_paths: Dict[str, str],
    *,
    part_id: Optional[str] = None,
    seat_model_id: Optional[str] = None,
    frame_id: Optional[str] = None,
    timestamp: Optional[str] = None,
)
```

`image_paths` 是 `{camera_id: image_path}`。

- `camera_id` 必须和配置中启用的机位一致。
- 缺少某个启用机位时，该机位返回 `REJECT`，错误码为 `missing_external_frame`。
- 图片读取失败时，该机位返回 `REJECT`，错误码为 `image_read_failed`。
- 传入未配置或未启用的 `camera_id` 会抛出 `ValueError`。
- 重复 `camera_id` 会抛出 `ValueError`。

### `inspect`

```python
inspect(
    frames: List[Union[InspectionFrame, Dict]],
    *,
    part_id: Optional[str] = None,
    seat_model_id: Optional[str] = None,
)
```

dict frame 必填字段：

- `camera_id`
- `image`

dict frame 可选字段：

- `source`
- `source_kind`
- `frame_id`
- `timestamp`
- `error_reason`

## 输出格式

`InspectionResponse.to_dict()` 返回适合 JSON 序列化的字典：

```json
{
  "part_id": "part_20260509_0001",
  "frame_id": "frame_0001",
  "timestamp": "2026-05-09T10:00:00+08:00",
  "status": "OK",
  "decision_reason": "all_checks_passed",
  "seat_model_id": "seat_model_a",
  "timings_ms": {
    "context": 0.1,
    "frames": 0.1,
    "cameras": 120.0,
    "fusion": 0.1,
    "total": 120.3
  },
  "report_path": "../outputs/seat_defect_inspection/results.json",
  "artifact_paths": {},
  "camera_results": [
    {
      "camera_id": "cam_front",
      "frame_id": "frame_0001",
      "source": "/data/current/cam_front.png",
      "source_kind": "image_path",
      "status": "OK",
      "reason": "all_regions_passed",
      "seat_model_id": "seat_model_a",
      "timings_ms": {
        "prepare": 30.0,
        "split_regions": 1.0,
        "region_patchcore_batch": 80.0,
        "debug_artifacts": 0.0,
        "total": 111.0
      },
      "error": null,
      "artifact_paths": {},
      "region_results": [
        {
          "region_id": "upper",
          "status": "OK",
          "reason": "all_checks_passed",
          "patchcore_model_path": "../models/seat_model_a/cam_front_upper_patchcore.npz",
          "timings_ms": {
            "patchcore": 80.0
          },
          "error": null,
          "artifact_paths": {}
        }
      ]
    }
  ]
}
```

状态含义：

- `OK`：检测通过。
- `NG`：检测到缺陷。
- `REJECT`：本次输入或中间流程不满足检测条件，不能作为合格/缺陷结论使用。

常见 `reason`：

- `all_checks_passed`
- `all_regions_passed`
- `texture_anomaly`
- `region_texture_anomaly:<region_id>`
- `color_anomaly`
- `target_not_found`
- `target_mask_missing`
- `low_valid_patch_ratio`
- `missing_external_frame`
- `image_read_failed`
- `pipeline_failed`

结构化错误字段：

```json
{
  "code": "image_read_failed",
  "message": "image_read_failed",
  "stage": "input"
}
```

外部系统应优先使用 `status`、`reason`、`error.code` 和 `error.stage` 做逻辑判断，不建议解析中文异常文本。

## 检测流程

单机位检测流程：

1. YOLO 检测目标座椅。
2. 根据分割 mask 做 ROI 裁剪和对齐。
3. 做图像质量检查。
4. 如果未配置 regions，执行完整 ROI PatchCore。
5. 如果配置了 regions，切分标准 ROI 并执行 region PatchCore。
6. 可选执行颜色一致性分支。
7. 汇总单机位结果。

多机位流程：

1. 校验外部传入机位。
2. 逐机位检测。
3. 按 fusion 配置汇总整件状态。
4. 写出 latest report。

## region 模式性能注意事项

region 模式会对一个机位内多个局部区域分别运行 PatchCore，因此天然比完整 ROI 单模型更慢。当前 core 已做以下优化：

- 相同 full-backend 配置共享 torch feature extractor。
- 相同 full-backend 配置的多个 region 使用 batch backbone 前向。
- 调试产物可通过 `debug_artifacts_enabled=false` 关闭。
- region 调试产物复用运行时已有 region sample，避免重复切图。

这些优化不会改变 ROI、region box、memory bank、阈值或最终判定规则，只可能带来极小的浮点差异。

## 模型和配置一致性

PatchCore 模型中保存了训练时的上游 pipeline signature。运行时如果修改了会影响 ROI 或特征输入的关键配置，core 会拒绝使用旧模型，并提示重新训练。

常见需要重新训练 PatchCore 的改动：

- YOLO 模型路径或目标类别发生变化。
- ROI 裁剪、mask、alignment 配置变化。
- region box 变化。
- PatchCore backend、image_size、texture_input、backbone 或 feature_layers 变化。

运行时可以调整部分判定阈值类配置，但不能用配置去掩盖训练数据不足的问题。

## 现场排查顺序

当结果为 `REJECT` 时，优先查看：

1. `camera_results[].error.code`
2. `camera_results[].error.stage`
3. `camera_results[].reason`
4. `camera_results[].timings_ms`
5. `report_path` 对应 JSON 报告

当速度偏慢时，优先检查：

1. `debug_artifacts_enabled` 是否为 `false`。
2. `timings_ms.cameras` 和单机位 `timings_ms.region_patchcore_batch`。
3. region 数量是否过多。
4. `backbone_device` 是否符合现场硬件。
5. 是否每次请求都重新创建 `SeatDefectInspector`。

## 版本稳定性建议

外部项目接入时，建议固定以下内容：

- `seat-defect-core` 包版本。
- 配置文件版本。
- YOLO 模型文件。
- PatchCore 模型文件。
- Python、torch、torchvision、ultralytics 版本。

LabVIEW 公共机建议固定 Python `3.8.5`，使用 CPU 版依赖，并在配置中设置 `backbone_device = cpu`。如果后续改用 GPU/CUDA，需要单独验证对应的 torch、torchvision 和驱动版本。

上线后不要直接替换模型或配置。任何模型或 ROI 配置调整，都应先在离线样本集上回归验证。
