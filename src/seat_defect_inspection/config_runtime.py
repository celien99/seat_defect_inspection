"""运行时路由与顶层项目配置。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config_anomaly import (
    ColorBranchConfig,
    DetectionConfig,
    PatchCoreConfig,
    YoloTrainingConfig,
)
from .config_image import PreprocessConfig, QualityGuardConfig
from .config_roi import RoiRefineConfig
from .debug_artifacts import DEFAULT_DEBUG_ARTIFACT_MODE


@dataclass(slots=True)
class CameraConfig:
    """单机位完整配置。"""

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
    """多机位融合判定策略。"""

    reject_on_any_reject: bool = True
    ng_strategy: str = "any"
    early_stop_on_ng: bool = True
    defect_overrides_reject: bool = True


@dataclass(slots=True)
class SeatModelConfig:
    """按座椅型号组织的多机位配置。"""

    seat_model_id: str
    cameras: list[CameraConfig] = field(default_factory=list)
    display_name: str | None = None
    yolo_training: YoloTrainingConfig | None = None


@dataclass(slots=True)
class InspectionConfig:
    """项目顶层配置。"""

    cameras: list[CameraConfig] = field(default_factory=list)
    seat_models: list[SeatModelConfig] = field(default_factory=list)
    default_seat_model_id: str | None = None
    output_json_path: str = "outputs/seat_defect_inspection/results.json"
    debug_dir: str = "outputs/seat_defect_inspection/debug"
    capture_dir: str = "outputs/seat_defect_inspection/capture"
    save_debug_artifacts: bool = True
    debug_artifact_mode: str = DEFAULT_DEBUG_ARTIFACT_MODE
    capture_retries: int = 3
    part_id: str = "seat_demo"
    fusion: FusionConfig = field(default_factory=FusionConfig)
