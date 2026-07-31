from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model.predict(
    source="test.jpg",
    save=True,
    conf=0.25
)