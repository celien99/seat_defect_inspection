from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    model = YOLO("best.pt")
    image_dir = Path(r"E:\code\seat_defect_inspection\datasets\seat_defect\images\train")
    image_paths = sorted(str(path) for path in image_dir.glob("*.png"))

    model.predict(
        source=image_paths,
        conf=0.5,
        iou=0.45,
        device="cpu",
        save=True,
        project="outputs/yolo_debug",
        name="cam_0_batch",
        verbose=False,
    )


if __name__ == "__main__":
    main()
