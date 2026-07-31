from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data="coco128.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device="cpu"
)