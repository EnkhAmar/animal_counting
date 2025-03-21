from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("trained_models/yolo12_best.pt")

# Run validation on the test dataset
results = model.val(data="data/yolo/data.yaml", split="test")

# Extract evaluation metrics
precision = results.results_dict.get("metrics/precision(B)", 0)
recall = results.results_dict.get("metrics/recall(B)", 0)
mAP50 = results.results_dict.get("metrics/mAP50(B)", 0)
mAP50_95 = results.results_dict.get("metrics/mAP50-95(B)", 0)

# Compute F1 Score
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Extract inference speed
inference_speed = results.speed.get("inference", 0)

# Print formatted results
print(f"\n🔹 **YOLO Model Evaluation Metrics**")
print(f"-----------------------------------")
print(f"📌 mAP @ IoU 0.5      : {mAP50:.4f}")
print(f"📌 mAP @ IoU 0.5:0.95  : {mAP50_95:.4f}")
print(f"📌 Precision          : {precision:.4f}")
print(f"📌 Recall             : {recall:.4f}")
print(f"📌 F1 Score           : {f1_score:.4f}")
print(f"📌 Inference Speed    : {inference_speed:.2f} ms/image")
