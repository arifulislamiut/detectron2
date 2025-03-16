# -*- coding: utf-8 -*-
"""Textile Training with Detectron2"""

import os
import torch
import detectron2
import json
import cv2
import random
import pickle
import yaml
import numpy as np
import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from roboflow import Roboflow
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import torch.nn as nn

setup_logger()

# ✅ Define model name and configuration
model_name = "mask_rcnn_R_101_FPN_3x"
desired_number_of_epochs = 200
num_of_class = 1
is_resume = True
dataset_base = "Textile-defect-seg-uni-2"
dataset_version = 2
dataset_format = "coco-segmentation"
output_dir = f"Detectron2_Models/{model_name}"
os.makedirs(output_dir, exist_ok=True)

# ✅ Download dataset using Roboflow
rf = Roboflow(api_key="R8JQyQmyXKa0HMr7HzJT")
project = rf.workspace("ds-gfhaj").project("textile-defect-seg-uni")
version = project.version(dataset_version)
dataset = version.download(dataset_format)

# ✅ Check if dataset paths exist before registering
train_json = f"{dataset_base}/train/_annotations.coco.json"
val_json = f"{dataset_base}/valid/_annotations.coco.json"
test_json = f"{dataset_base}/test/_annotations.coco.json"

if os.path.exists(train_json):
    register_coco_instances("my_dataset_train", {}, train_json, f"{dataset_base}/train")
if os.path.exists(val_json):
    register_coco_instances("my_dataset_val", {}, val_json, f"{dataset_base}/valid")
if os.path.exists(test_json):
    register_coco_instances("my_dataset_test", {}, test_json, f"{dataset_base}/test")

# ✅ Load dataset metadata
train_metadata = MetadataCatalog.get("my_dataset_train")
train_dataset_dicts = DatasetCatalog.get("my_dataset_train")

# ✅ Save metadata
train_metadata_path = os.path.join(output_dir, "train_metadata.pkl")
with open(train_metadata_path, "wb") as f:
    pickle.dump(train_metadata, f)
print("Metadata saved to train_metadata.pkl")

# ✅ Display sample images
for d in random.sample(train_dataset_dicts, 2):
    img_plot = d["file_name"]
    img = cv2.imread(img_plot)
    visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)

    plt.figure(figsize=(30, 15))
    plt.imshow(vis.get_image()[:, :, ::-1])
    plt.show()

# ✅ Initialize and configure model
cfg = get_cfg()
cfg.OUTPUT_DIR = output_dir
cfg.merge_from_file(model_zoo.get_config_file(f"COCO-InstanceSegmentation/{model_name}.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_test",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-InstanceSegmentation/{model_name}.yaml")
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.00025
cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_of_class
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024

num_images = len(train_dataset_dicts)
iterations_per_epoch = num_images // cfg.SOLVER.IMS_PER_BATCH
cfg.SOLVER.MAX_ITER = iterations_per_epoch * desired_number_of_epochs
cfg.SOLVER.STEPS = []

# ✅ Build the model and load pre-trained weights
model = build_model(cfg)
checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)

# ✅ Reset model's predictor layers to match dataset classes
model.roi_heads.box_predictor.cls_score = nn.Linear(1024, num_of_class + 1)  # +1 for background
model.roi_heads.box_predictor.bbox_pred = nn.Linear(1024, num_of_class * 4)
model.roi_heads.mask_head.predictor = nn.Conv2d(256, num_of_class, kernel_size=(1, 1))

# ✅ Save updated model weights
updated_weights_path = os.path.join(output_dir, "model_updated.pth")
torch.save(model.state_dict(), updated_weights_path)
cfg.MODEL.WEIGHTS = updated_weights_path

# ✅ Save training config
config_yaml_path = os.path.join(output_dir, "training_config.yaml")
with open(config_yaml_path, 'w') as file:
    yaml.dump(cfg, file)

# ✅ Train the model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=is_resume)
trainer.train()

# ✅ Set trained model weights
model_weights_path = os.path.join(output_dir, "model_final.pth")
cfg.MODEL.WEIGHTS = model_weights_path

# ✅ Initialize predictor
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
predictor = DefaultPredictor(cfg)

# ✅ Evaluate model
evaluator = COCOEvaluator("my_dataset_test", output_dir=output_dir)
val_loader = build_detection_test_loader(cfg, "my_dataset_test")
inference_on_dataset(predictor.model, val_loader, evaluator)

# ✅ Visualize results
new_im = cv2.imread(img_plot)
outputs = predictor(new_im)
v = Visualizer(new_im[:, :, ::-1], metadata=train_metadata)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

new_im_rgb = cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB)
out_img_rgb = out.get_image()[:, :, ::-1]

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(new_im_rgb)
ax[0].set_title("Original Image")
ax[0].axis("off")
ax[1].imshow(out_img_rgb)
ax[1].set_title("Annotated Image")
ax[1].axis("off")
plt.show()

# ✅ Save inference config
inference_yaml_path = os.path.join(output_dir, "inference_config.yaml")
with open(inference_yaml_path, 'w') as file:
    yaml.dump(cfg, file)
