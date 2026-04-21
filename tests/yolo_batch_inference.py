from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolo11n.pt")
    image_dir = Path(r"/Users/yyh/code/seat_defect_inspection/datasets/seat_defect/images/train")
    image_paths = sorted(str(path) for path in image_dir.glob("*.png"))

    model.predict(
        source=image_paths,
        conf=0.05,
        iou=0.45,
        device="cpu",
        save=True,
        project="outputs/yolo_debug",
        name="cam_0_batch",
        verbose=False,
    )


if __name__ == "__main__":
    main()
