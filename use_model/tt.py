from ultralytics import YOLO

# Configure the tracking parameters and run the tracker
model = YOLO("use_model/yolo11m-seg_cow/weights/best.pt")
results = model.track(source="https://youtu.be/RTRdu-WzPeM", conf=0.9, iou=0.5, show=True)