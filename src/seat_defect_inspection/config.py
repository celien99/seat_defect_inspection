"""座椅缺陷检测项目的运行配置。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import BoundingBox


@dataclass(slots=True)
class QualityGuardConfig:
    """检测前的图像质量阈值。

    字段：
    - min_laplacian_variance: 最小清晰度阈值，低于该值判定为模糊
    - min_brightness_mean: 最小平均亮度，低于该值判定为欠曝
    - max_brightness_mean: 最大平均亮度，高于该值判定为过曝
    - max_overexposed_ratio: 高亮像素占比阈值
    - max_underexposed_ratio: 低亮像素占比阈值
    """

    min_laplacian_variance: float = 80.0
    min_brightness_mean: float = 30.0
    max_brightness_mean: float = 225.0
    max_overexposed_ratio: float = 0.25
    max_underexposed_ratio: float = 0.35


@dataclass(slots=True)
class PreprocessConfig:
    """YOLO 前的 OpenCV 预处理参数。

    字段：
    - resize_width / resize_height: 送入 YOLO 前的目标尺寸
    - denoise_method: 去噪方式，支持 gaussian / bilateral / none
    - gaussian_kernel_size: 高斯去噪核大小
    - bilateral_diameter / bilateral_sigma_color / bilateral_sigma_space:
      双边滤波参数
    - white_balance_method: 白平衡方式，当前支持 none / gray_world
    - max_white_balance_gain: 白平衡最大增益，防止颜色校正过度
    - apply_illumination_correction: 是否启用大核光照校正
    - illumination_blur_kernel_size / illumination_strength: 光照校正参数
    - apply_clahe / clahe_clip_limit / clahe_tile_grid_size: CLAHE 参数
    - gamma: 伽马校正参数，None 表示关闭
    - sharpen / sharpen_sigma / sharpen_amount: 锐化开关与强度
    - camera_matrix / distortion_coeffs: 畸变校正参数
    """

    resize_width: int | None = None
    resize_height: int | None = None
    denoise_method: str = "gaussian"
    gaussian_kernel_size: int = 5
    bilateral_diameter: int = 5
    bilateral_sigma_color: float = 30.0
    bilateral_sigma_space: float = 30.0
    white_balance_method: str = "none"
    max_white_balance_gain: float = 1.25
    apply_illumination_correction: bool = False
    illumination_blur_kernel_size: int = 51
    illumination_strength: float = 0.7
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    gamma: float | None = None
    sharpen: bool = False
    sharpen_sigma: float = 1.2
    sharpen_amount: float = 1.0
    camera_matrix: list[list[float]] | None = None
    distortion_coeffs: list[float] | None = None


@dataclass(slots=True)
class DetectionConfig:
    """单机位 YOLO 检测配置。

    字段：
    - model_path: YOLO 权重路径；为空时走 fallback_box
    - target_class: 主目标类别名，通常为 seat_main
    - ignore_classes: 需要进入忽略掩膜的干扰类别
    - confidence / iou / device: YOLO 推理参数
    - prefer_segmentation_mask: 有分割头时优先使用 mask
    - fallback_box: YOLO 不可用或未检出时的兜底 ROI
    """

    model_path: str | None = None
    target_class: str = "seat_main"
    ignore_classes: list[str] = field(default_factory=list)
    confidence: float = 0.25
    iou: float = 0.45
    device: str = "cpu"
    prefer_segmentation_mask: bool = True
    fallback_box: BoundingBox | None = None


@dataclass(slots=True)
class AlignmentConfig:
    """ROI 裁剪后的对齐参数。

    字段：
    - enabled: 是否启用对齐
    - method: 对齐方式，当前主要支持 resize / ecc
    - template_image_path: ECC 模板图路径
    - output_width / output_height: 对齐后的统一尺寸
    - ecc_iterations: ECC 迭代次数
    """

    enabled: bool = False
    method: str = "resize"
    template_image_path: str | None = None
    output_width: int = 256
    output_height: int = 256
    ecc_iterations: int = 50


@dataclass(slots=True)
class RoiRefineConfig:
    """ROI 精修与掩膜生成配置。

    字段：
    - crop_expand_ratio / crop_shrink_ratio: 检测框扩缩比例
    - mask_mode: 目标掩膜生成方式，支持 grabcut / full
    - morphology_kernel_size: 目标掩膜开闭运算核大小
    - ignore_dilate_kernel_size: 干扰区域膨胀核大小
    - edge_ignore_pixels: ROI 边缘忽略像素数
    - texture_*: PatchCore 前 ROI 纹理链路参数
    - apply_texture_clahe: 是否仅在前景内做 CLAHE
    - texture_illumination_*: ROI 内局部光照归一化参数
    - mask_feather_kernel_size: 前景羽化权重核大小
    - edge_enhance_method / edge_enhance_weight: 纹理边缘增强方式和强度
    - suppress_background / background_fill_mode / background_blur_kernel_size:
      ROI 外背景压制策略
    - safe_margin_erode_kernel_size: 纹理分析安全边界腐蚀核
    - alignment: ROI 对齐配置
    """

    crop_expand_ratio: float = 0.05
    crop_shrink_ratio: float = 0.0
    mask_mode: str = "grabcut"
    morphology_kernel_size: int = 5
    ignore_dilate_kernel_size: int = 9
    edge_ignore_pixels: int = 6
    texture_denoise_method: str = "bilateral"
    texture_gaussian_kernel_size: int = 5
    texture_bilateral_diameter: int = 7
    texture_bilateral_sigma_color: float = 40.0
    texture_bilateral_sigma_space: float = 40.0
    apply_texture_clahe: bool = True
    texture_clahe_clip_limit: float = 2.0
    texture_clahe_tile_grid_size: int = 8
    texture_illumination_correction: bool = True
    texture_illumination_blur_kernel_size: int = 41
    texture_illumination_strength: float = 0.85
    mask_feather_kernel_size: int = 15
    edge_enhance_method: str = "scharr"
    edge_enhance_weight: float = 0.18
    suppress_background: bool = True
    background_fill_mode: str = "median"
    background_blur_kernel_size: int = 31
    safe_margin_erode_kernel_size: int = 3
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)


@dataclass(slots=True)
class PatchCoreConfig:
    """PatchCore 风格模型与 patch 过滤配置。

    字段：
    - backend: PatchCore 后端，支持 full / handcrafted
    - image_size: ROI 统一缩放尺寸
    - patch_size / stride: Patch 提取参数
    - max_memory: 记忆库最大容量
    - threshold_quantile: 阈值估计分位数
    - texture_input: 特征输入模式，支持 gray / lab_l / rgb_lab
    - min_target_coverage: patch 落在目标区域的最小覆盖率
    - max_ignore_overlap: patch 与忽略区域的最大重叠率
    - min_valid_patch_ratio: 推理时最小有效 patch 占比
    - decision_score_margin: 最终判定时对训练阈值再乘一层安全系数
    - strong_patch_score_ratio: 强异常 patch 判定比例，基于当前图像分数和训练阈值共同决定
    - min_strong_patch_count: 至少需要多少个强异常 patch 才允许判 NG
    - min_strong_component_count: 最大连通强异常区域至少包含多少个 patch
    - min_strong_patch_ratio: 强异常 patch 占全部有效 patch 的最小比例
    - min_strong_component_ratio: 最大连通强异常区域占全部有效 patch 的最小比例
    - critical_score_margin: 强异常快速命中时的整图分数安全系数
    - critical_peak_score_margin: 强异常快速命中时的峰值 patch 分数安全系数
    - critical_min_component_patch_count: 强异常快速命中要求的最小连通 patch 数
    - backbone_name / feature_layers: 完整 PatchCore 的 CNN backbone 与取特征层
    - backbone_pretrained / backbone_weights_path / backbone_device:
      完整 PatchCore 的权重加载与推理设备配置
    - feature_pool_kernel_size: 完整 PatchCore 对中间层做局部平均池化的核大小
    - coreset_sampling_ratio: 完整 PatchCore 记忆库抽样比例
    """

    backend: str = "full"
    image_size: int = 256
    patch_size: int = 32
    stride: int = 16
    max_memory: int = 512
    threshold_quantile: float = 0.99
    texture_input: str = "lab_l"
    min_target_coverage: float = 0.8
    max_ignore_overlap: float = 0.1
    min_valid_patch_ratio: float = 0.65
    decision_score_margin: float = 1.08
    strong_patch_score_ratio: float = 0.9
    min_strong_patch_count: int = 3
    min_strong_component_count: int = 2
    min_strong_patch_ratio: float = 0.015
    min_strong_component_ratio: float = 0.01
    critical_score_margin: float = 1.35
    critical_peak_score_margin: float = 1.45
    critical_min_component_patch_count: int = 2
    backbone_name: str = "wide_resnet50_2"
    feature_layers: list[str] = field(default_factory=lambda: ["layer2", "layer3"])
    backbone_pretrained: bool = False
    backbone_weights_path: str | None = None
    backbone_device: str = "cpu"
    feature_pool_kernel_size: int = 3
    coreset_sampling_ratio: float = 0.1


@dataclass(slots=True)
class ColorBranchConfig:
    """颜色一致性分支配置。

    字段：
    - enabled: 是否启用颜色分支
    - threshold_quantile: 阈值估计分位数
    - threshold: 手工指定阈值；不填时自动估计
    - min_valid_pixel_ratio: 训练颜色统计量时要求的最小有效像素占比
    """

    enabled: bool = False
    threshold_quantile: float = 0.99
    threshold: float | None = None
    min_valid_pixel_ratio: float = 0.4


@dataclass(slots=True)
class CameraConfig:
    """单机位完整配置。

    字段：
    - camera_id: 机位唯一标识
    - source: 输入源，支持图片、视频、普通相机、mvs://
    - patchcore_model_path: 该机位 PatchCore 模型路径
    - train_good_dir: 该机位正常样本目录
    - enabled: 是否启用该机位
    - color_insensitive_mode: 是否启用颜色不敏感模式
    - quality / preprocess / detection / roi / patchcore / color_branch:
      对应该机位各处理阶段的子配置
    """

    camera_id: str
    source: str
    patchcore_model_path: str
    train_good_dir: str | None = None
    enabled: bool = True
    color_insensitive_mode: bool = False
    quality: QualityGuardConfig = field(default_factory=QualityGuardConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    roi: RoiRefineConfig = field(default_factory=RoiRefineConfig)
    patchcore: PatchCoreConfig = field(default_factory=PatchCoreConfig)
    color_branch: ColorBranchConfig = field(default_factory=ColorBranchConfig)


@dataclass(slots=True)
class FusionConfig:
    """多机位融合判定策略。

    字段：
    - reject_on_any_reject: 是否只要有 REJECT 就整体 REJECT
    - ng_strategy: NG 融合策略，支持 any / all / majority
    - early_stop_on_ng: 当 NG 融合策略已满足时是否提前结束后续机位
    - defect_overrides_reject: 已确认存在缺陷时，是否优先输出 NG 而不是 REJECT
    """

    reject_on_any_reject: bool = True
    ng_strategy: str = "any"
    early_stop_on_ng: bool = True
    defect_overrides_reject: bool = True


@dataclass(slots=True)
class YoloTrainingConfig:
    """YOLO 训练配置。

    字段：
    - model_path: 预训练权重或基础模型名
    - data_config_path: Ultralytics 数据集 YAML
    - epochs / imgsz / batch / device: 训练核心参数
    - project / name: 输出目录和任务名
    - workers / patience / cache / pretrained: 训练辅助参数
    - seat_model_id: 当前训练配置所属座椅型号
    """

    model_path: str = "yolo11n.pt"
    data_config_path: str = "configs/seat_defect_yolo.dataset.example.yaml"
    epochs: int = 100
    imgsz: int = 1280
    batch: int = 8
    device: str = "cpu"
    project: str = "outputs/yolo_training"
    name: str = "seat_defect"
    workers: int = 4
    patience: int = 20
    cache: bool = False
    pretrained: bool = True
    seat_model_id: str | None = None


@dataclass(slots=True)
class SeatModelConfig:
    """按座椅型号组织的多机位配置。

    字段：
    - seat_model_id: 型号唯一标识
    - cameras: 该型号下的所有机位配置
    - display_name: 人类可读名称
    - yolo_training: 当前型号的 YOLO 训练配置
    """

    seat_model_id: str
    cameras: list[CameraConfig] = field(default_factory=list)
    display_name: str | None = None
    yolo_training: YoloTrainingConfig | None = None


@dataclass(slots=True)
class InspectionConfig:
    """项目顶层配置。

    字段：
    - cameras: 单型号模式下直接使用的机位列表
    - seat_models: 多型号路由模式下的型号列表
    - default_seat_model_id: 默认型号
    - output_json_path: 最终检测结果 JSON 路径
    - debug_dir: 调试图输出目录
    - capture_dir: 采图输出目录
    - save_debug_artifacts: 是否保存调试产物
    - capture_retries: 单机位取流重试次数
    - part_id: 默认工件编号
    - fusion: 多机位融合策略
    """

    cameras: list[CameraConfig] = field(default_factory=list)
    seat_models: list[SeatModelConfig] = field(default_factory=list)
    default_seat_model_id: str | None = None
    output_json_path: str = "outputs/seat_defect_inspection/results.json"
    debug_dir: str = "outputs/seat_defect_inspection/debug"
    capture_dir: str = "outputs/seat_defect_inspection/capture"
    save_debug_artifacts: bool = True
    capture_retries: int = 3
    part_id: str = "seat_demo"
    fusion: FusionConfig = field(default_factory=FusionConfig)
