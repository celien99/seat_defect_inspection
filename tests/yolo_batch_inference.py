from pathlib import Path

from ultralytics import YOLO

from seat_defect_inspection.patchcore import list_images

# 直接改这里就可以：
MODEL_PATH = "best.pt"
IMAGE_DIR = "datasets/seat_defect/images/val"
CONFIDENCE = 0.5
IOU = 0.45
DEVICE = "cpu"
PROJECT_DIR = "outputs/yolo_debug"
RUN_NAME = "cam_0_batch"


def main() -> None:
    # 这里只测试训练好的 YOLO 分割模型本身。
    model = YOLO(MODEL_PATH)
    image_paths = [str(path) for path in list_images(Path(IMAGE_DIR))]

    model.predict(
        source=image_paths,
        conf=CONFIDENCE,
        iou=IOU,
        device=DEVICE,
        save=True,
        project=PROJECT_DIR,
        name=RUN_NAME,
        verbose=False,
    )


if __name__ == "__main__":
    main()
