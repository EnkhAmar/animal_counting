import os
import glob
import numpy as np
import cv2
import json
import time
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Register dataset
register_coco_instances("my_dataset_train", {}, "data/coco/train/_annotations.coco.json", "data/coco/train")
register_coco_instances("my_dataset_test", {}, "data/coco/test/_annotations.coco.json", "data/coco/test")

# Load configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_test",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = os.path.join("/home/amra/edu/diplom/trained_models", "faster_rcnn_best.pth")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Adjust based on your dataset
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3    # IoU threshold for NMS

# Initialize predictor
predictor = DefaultPredictor(cfg)

# Evaluate model
evaluator = COCOEvaluator("my_dataset_test", output_dir="./rcnn_eval_output/")
val_loader = build_detection_test_loader(cfg, "my_dataset_test")

# Measure inference time
start_time = time.time()
results = inference_on_dataset(predictor.model, val_loader, evaluator)
end_time = time.time()

# Load ground truth annotations
coco_gt = COCO("data/coco/test/_annotations.coco.json")
coco_dt = coco_gt.loadRes("./rcnn_eval_output/coco_instances_results.json")

# Run COCO Evaluation
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Extract metrics
mAP50_95 = coco_eval.stats[0]  # mAP @ IoU 0.5:0.95
mAP50 = coco_eval.stats[1]      # mAP @ IoU 0.5
precision = coco_eval.stats[1]  # Approximate Precision @ IoU 0.5
recall = coco_eval.stats[8]     # AR @ 100 (recall)
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Inference Speed (Time per image)
num_images = len(DatasetCatalog.get("my_dataset_test"))
inference_speed = (end_time - start_time) / num_images * 1000  # Convert to milliseconds

# Print results
print("\n🔹 **Faster R-CNN Evaluation Metrics**")
print("-----------------------------------")
print(f"📌 mAP @ IoU 0.5      : {mAP50:.4f}")
print(f"📌 mAP @ IoU 0.5:0.95  : {mAP50_95:.4f}")
print(f"📌 Precision          : {precision:.4f}")
print(f"📌 Recall             : {recall:.4f}")
print(f"📌 F1 Score           : {f1_score:.4f}")
print(f"📌 Inference Speed    : {inference_speed:.2f} ms/image")
