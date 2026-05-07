# Seat Defect SDK

`seat_defect_sdk` is the clean runtime SDK for externally captured images. It does not open cameras and does not include CLI, capture, training, MVS camera, or offline-folder workflows in the SDK wheel.

```python
from seat_defect_sdk import CameraFrame, SeatDefectInspector

inspector = SeatDefectInspector("configs/seat_defect_inspection.mvs.json")
response = inspector.inspect(
    frames=[
        CameraFrame(camera_id="cam_0", image=cam_0_image),
        CameraFrame(camera_id="cam_1", image=cam_1_image),
    ],
    part_id="seat_000001",
)

print(response.status)
print(response.report_path)
print(response.archive_report_path)
print(response.artifact_paths)
```
