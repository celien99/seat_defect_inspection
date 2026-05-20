# 检测报告（latest.json）字段说明

本文档说明检测完成后生成的 JSON 报告（通常命名为 `latest.json` 或 `results.json`）的完整字段结构与含义。

## 报告生成位置

- **在线检测**：由 `output_json_path` 配置项指定，默认 `outputs/seat_defect_inspection/results.json`
- **离线批量检测**：每次运行生成独立路径 `<base>/<run_id>/reports/latest.json`

---

## 顶层结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `part_id` | string | 工件编号，标识被检测的零件 |
| `frame_id` | string | 本次检测批次帧编号，通常由时间戳 + 序号组成 |
| `timestamp` | string | ISO 8601 格式的检测时间戳 |
| `status` | string | 整件判定状态：`"OK"`（合格）、`"NG"`（不合格）、`"REJECT"`（拒识） |
| `decision_reason` | string | 多机位融合后的判定原因描述 |
| `seat_model_id` | string | 本次检测使用的座椅型号 ID |
| `timings_ms` | object | 整件检测各阶段耗时（毫秒），详见下方 |
| `camera_results` | array | 所有机位的详细检测结果，每项结构见下方 |

### timings_ms（顶层）

记录从 `inspect_frames()` 开始到完成全流程的各阶段耗时：

| 键 | 说明 |
|----|------|
| `context` | 解析运行时上下文耗时（解析配置、加载模型列表等） |
| `frames` | 构建帧映射并校验相机 ID 的耗时 |
| `cameras` | 所有相机检测循环的总耗时（含 YOLO、ROI、PatchCore、颜色分支等） |
| `fusion` | 多机位结果融合耗时 |
| `total` | 从 `inspect_frames` 调用开始到返回结果的端到端总耗时 |

---

## camera_results 数组项

每个元素代表一个机位的完整检测输出。

| 字段 | 类型 | 说明 |
|------|------|------|
| `camera_id` | string | 机位 ID，对应配置中 `cameras[].camera_id` |
| `frame_id` | string | 该机位使用的帧编号 |
| `source` | string | 输入来源标识（如文件路径、相机设备名、`external://cam_0`） |
| `source_kind` | string | 输入来源类型（如 `"image_path"`、`"external_image"`） |
| `status` | string | 单机位判定状态：`"OK"`、`"NG"`、`"REJECT"` |
| `reason` | string | 单机位状态的原因描述，见下方[判定原因码](#判定原因码) |
| `seat_model_id` | string / null | 该机位使用的座椅型号 ID |
| `timings_ms` | object | 该机位各阶段耗时（毫秒），见下方 |
| `error` | object / null | 结构化错误信息，见下方 |
| `quality` | object / null | 图像质量门控结果，见下方 |
| `target_box` | object / null | YOLO 检测到的目标边界框，见下方 |
| `crop_box` | object / null | 原图坐标系下最终使用的 ROI 裁剪框 |
| `texture_result` | object / null | 纹理异常检测结果（完整 ROI 模式），见下方 |
| `region_results` | array | 区域检测结果列表（区域模式），每项见下方 |
| `color_result` | object / null | 颜色一致性检测结果，见下方 |
| `artifact_paths` | object | 该机位关联的调试产物路径映射 |

### timings_ms（单机位）

记录该机位从准备到结束的各阶段耗时。根据检测模式（完整 ROI / 区域模式）包含的键会有所不同：

| 键 | 出现条件 | 说明 |
|----|----------|------|
| `prepare` | 始终 | YOLO 检测 + 图像预处理的总耗时 |
| `patchcore` | 完整 ROI 模式 | PatchCore 纹理推理耗时 |
| `split_regions` | 区域模式 | 将 ROI 按配置区域划分的耗时 |
| `region_patchcore_batch` | 区域模式 | 该机位所有区域的 PatchCore 批处理总耗时 |
| `color` | 始终 | 颜色一致性预测耗时 |
| `debug_artifacts` | 始终 | 生成叠加图和保存调试产物的耗时 |
| `total` | 始终 | 该机位从开始到结束的端到端总耗时 |

### error（结构化错误）

当检测流程中发生异常时非 null：

| 字段 | 类型 | 说明 |
|------|------|------|
| `code` | string | 稳定错误码，供外部系统识别（如 `"low_valid_patch_ratio"`、`"pipeline_failed"`） |
| `message` | string | 面向日志/调试的错误描述 |
| `stage` | string | 错误发生的阶段（如 `"prepare"`、`"patchcore"`、`"camera_pipeline"`） |

### quality（图像质量门控）

图像通过质量门控时为 `null`（即 `accepted: true` 且无 `reason`）。不通过时有完整结构：

| 字段 | 类型 | 说明 |
|------|------|------|
| `accepted` | bool | 是否通过质量门控 |
| `reason` | string | 未通过时的原因描述 |
| `metrics` | object | 原始质量指标 |

**metrics 子字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `laplacian_variance` | float | 拉普拉斯方差，衡量图像清晰度 |
| `brightness_mean` | float | 平均亮度值 |
| `overexposed_ratio` | float | 过曝像素占比 |
| `underexposed_ratio` | float | 欠曝像素占比 |
| `is_black_frame` | bool | 是否判定为黑帧 |
| `is_white_frame` | bool | 是否判定为白帧 |

### target_box（检测目标框）

YOLO 模型检测到的目标（座椅）在原图中的边界框，未检测到时为 `null`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `x1` | float | 左上角 X 坐标（原图坐标系） |
| `y1` | float | 左上角 Y 坐标（原图坐标系） |
| `x2` | float | 右下角 X 坐标（原图坐标系） |
| `y2` | float | 右下角 Y 坐标（原图坐标系） |

### crop_box（ROI 裁剪框）

最终用于检测的 ROI 区域在原图中的位置：

| 字段 | 类型 | 说明 |
|------|------|------|
| `x1` | float | 左上角 X 坐标（原图坐标系） |
| `y1` | float | 左上角 Y 坐标（原图坐标系） |
| `x2` | float | 右下角 X 坐标（原图坐标系） |
| `y2` | float | 右下角 Y 坐标（原图坐标系） |

### texture_result（纹理异常结果）

完整 ROI 模式下 PatchCore 的纹理异常检测输出。区域模式下为 `null`（区域结果见 `region_results`）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `score` | float | 图像级异常分数，越高表示越异常 |
| `threshold` | float | 训练阶段得到的基础异常阈值（统计分位数） |
| `decision_threshold` | float | 最终工业判定使用的阈值 |
| `is_anomaly` | bool | 是否判定为纹理异常 |
| `valid_patch_ratio` | float | 有效 patch 占全部 patch 的比例 |
| `valid_patch_count` | int | 有效 patch 数量（覆盖目标区域足够的 patch） |
| `total_patch_count` | int | 全部 patch 数量 |
| `peak_patch_score` | float | 当前图像中最高 patch 的异常分数 |
| `strong_patch_count` | int | 达到强异常阈值的 patch 数量 |
| `largest_component_patch_count` | int | 最大强异常连通域包含的 patch 数量 |
| `strong_patch_ratio` | float | 强异常 patch 占有效 patch 的比例 |
| `largest_component_patch_ratio` | float | 最大强异常连通域占有效 patch 的比例 |
| `decision_patch_count` | int | 达到最终判定阈值的 patch 数量 |
| `largest_decision_component_patch_count` | int | 最大最终判定连通域包含的 patch 数量 |
| `decision_patch_ratio` | float | 达到最终判定阈值的 patch 占比 |
| `largest_decision_component_patch_ratio` | float | 最大最终判定连通域占比 |
| `decision_mode` | string | 最终命中的判定模式（如 `"none"`、`"peak"`、`"component"` 等） |

### region_results（区域检测结果）

仅在配置了 regions 且至少有一个区域 `enabled: true` 时出现。数组中每项结构：

| 字段 | 类型 | 说明 |
|------|------|------|
| `region_id` | string | 区域 ID，对应配置中 `regions[].region_id` |
| `status` | string | 区域判定状态：`"OK"`、`"NG"`、`"REJECT"` |
| `reason` | string | 区域状态原因描述 |
| `box` | object | 标准 ROI 坐标系下的区域矩形框 `{x1, y1, x2, y2}` |
| `patchcore_model_path` | string / null | 该区域使用的 PatchCore 模型路径 |
| `texture_result` | object / null | 同上方 `texture_result` 结构 |
| `artifact_paths` | object | 该区域关联的调试产物路径 |
| `timings_ms` | object | 只含 `patchcore` 一个键，为该区域分摊的 PatchCore 耗时 |
| `error` | object / null | 结构化错误，格式同上方 `error` |

### color_result（颜色一致性结果）

仅在配置中 `color_branch.enabled: true` 时非 null：

| 字段 | 类型 | 说明 |
|------|------|------|
| `score` | float | 颜色异常分数，越高越异常 |
| `threshold` | float | 颜色异常判定阈值 |
| `is_anomaly` | bool | 是否判定为颜色异常 |
| `diagnostics` | object | 颜色分支的诊断指标（键值对，内容由颜色模型决定） |

### artifact_paths（调试产物路径）

一个字符串键值对映射，记录该机位生成的调试产物文件名及其保存路径。仅在 `debug_artifacts_enabled: true` 时有内容。

---

## 判定原因码

### camera_results[].reason（单机位）

| reason 值 | 含义 |
|-----------|------|
| `all_checks_passed` | 所有检测分支均通过 |
| `texture_anomaly` | 纹理异常判定为 NG |
| `texture_and_color_anomaly` | 纹理异常 + 颜色异常同时触发 |
| `color_anomaly` | 仅颜色一致性异常 |
| `low_valid_patch_ratio` | 有效 patch 比例不足，拒识 |
| `texture_anomaly_quality_override` | 图像质量不通过但纹理仍判定为 NG（质量门控被纹理覆盖） |
| `color_anomaly_quality_override` | 图像质量不通过但颜色仍判定为 NG |
| `texture_and_color_anomaly_quality_override` | 质量不通过但纹理+颜色双异常 |
| `region_texture_anomaly:{ids}` | 区域模式：指定区域纹理异常（ids 为异常区域 ID 列表） |
| `region_texture_and_color_anomaly` | 区域模式：区域纹理异常 + 颜色异常 |
| `region_reject:{id}:{reason}` | 区域模式：指定区域拒识 |
| `all_regions_passed` | 区域模式：所有区域均通过 |
| `no_enabled_regions` | 区域模式：没有启用的区域 |
| `camera_prepare_failed` | 机位预处理失败（检测/ROI 阶段异常） |
| `pipeline_failed` | pipeline 执行异常 |
| `quality_*` | 以 `quality_` 开头的为各种图像质量拒识原因 |

### decision_reason（整件融合）

| decision_reason 值 | 含义 |
|--------------------|------|
| `all_cameras_ok` | 所有机位均为 OK |
| `ng_from_{camera_id}` | 某机位判定为 NG（单机位 NG 时） |
| `any_ng` | any 融合策略下存在 NG |
| `majority_ng` | majority 融合策略下多数机位 NG |
| `all_cameras_ng` | all 融合策略下所有机位 NG |
| `not_enough_cameras` | 有效机位数不足 |
| `no_enabled_cameras` | 没有启用的机位 |

---

## 完整示例

```json
{
  "part_id": "seat_001",
  "frame_id": "20260520_143010_000001",
  "timestamp": "2026-05-20T14:30:10+08:00",
  "status": "OK",
  "decision_reason": "all_cameras_ok",
  "seat_model_id": "seat_model_a",
  "timings_ms": {
    "context": 12.3,
    "frames": 1.2,
    "cameras": 345.6,
    "fusion": 0.8,
    "total": 360.0
  },
  "camera_results": [
    {
      "camera_id": "cam_0",
      "frame_id": "20260520_143010_000001",
      "source": "/data/images/cam_0_001.jpg",
      "source_kind": "image_path",
      "status": "OK",
      "reason": "all_checks_passed",
      "seat_model_id": "seat_model_a",
      "timings_ms": {
        "prepare": 120.5,
        "patchcore": 86.2,
        "color": 4.1,
        "debug_artifacts": 23.0,
        "total": 233.8
      },
      "error": null,
      "quality": null,
      "target_box": {
        "x1": 120.0,
        "y1": 80.0,
        "x2": 940.0,
        "y2": 620.0
      },
      "crop_box": {
        "x1": 115.0,
        "y1": 75.0,
        "x2": 945.0,
        "y2": 625.0
      },
      "texture_result": {
        "score": 0.213,
        "threshold": 0.852,
        "decision_threshold": 0.852,
        "is_anomaly": false,
        "valid_patch_ratio": 0.95,
        "valid_patch_count": 152,
        "total_patch_count": 160,
        "peak_patch_score": 0.341,
        "strong_patch_count": 0,
        "largest_component_patch_count": 0,
        "strong_patch_ratio": 0.0,
        "largest_component_patch_ratio": 0.0,
        "decision_patch_count": 0,
        "largest_decision_component_patch_count": 0,
        "decision_patch_ratio": 0.0,
        "largest_decision_component_patch_ratio": 0.0,
        "decision_mode": "none"
      },
      "region_results": [],
      "color_result": {
        "score": 0.12,
        "threshold": 0.50,
        "is_anomaly": false,
        "diagnostics": {
          "delta_e_mean": 1.2,
          "delta_e_std": 0.8
        }
      },
      "artifact_paths": {
        "overlay": "/outputs/debug/cam_0_overlay.png",
        "heatmap": "/outputs/debug/cam_0_heatmap.png"
      }
    }
  ]
}
```
