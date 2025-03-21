from ultralytics import YOLO
import torch

# Load YOLO model
model = YOLO("yolo12m.pt")

# Train the model
train_results = model.train(
    data="data/yolo/data.yaml",
    epochs=100,
    imgsz=640,
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch=16 
)

model.export(format="pt")
