from ultralytics import YOLO
import torch

# Load YOLO model
model = YOLO("yolo12m.pt")

# Train the model
train_results = model.train(
    data="data/yolo/data.yaml",
    epochs=100,
    time=None,
    patience=100,
    imgsz=640,
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch=16,
    iou=0.6,
    conf=0.001,
)

model.export(format="torchscript")
