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
    imgsz=800,
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch=3,
)

model.export(format="torchscript")
