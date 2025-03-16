# -*- coding: utf-8 -*-
"""textile_training_R_50_FPN_3x(06032025)

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/textile-training-r-50-fpn-3x-06032025-f37c182f-25dc-4a52-8173-d50f4b1e5ac5.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20250306/auto/storage/goog4_request%26X-Goog-Date%3D20250306T233707Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D6382566de60674da3f94860fc018429322df506826b5fa090c957d0821e1efab2ae73e4e5e078e14c0007ff760afe655f21368af6406e8128fe8b19ce47fb8798fa9a9cded17482ab7842f83ff9518d0fb5781279f5f41ccbff05c9e243b30e3850c7f12bd49d31dc46ae36eb53008418d3146cdd5f06c666c1521810734aea60ba4eb6fe17500bdd1407090ee38e6defff29b21a6b85847e4075ac13ad50b57ff5273065c6778e3f25ab65b1866a718cfe5fe5d1a1a51e278824fdc2d6f4dafe4018491d5a9e5e06735081b68f007317d6f627615e9053a0f98b92674108943f842f04f4ec28200f7e893da1531bc10c4b03b3d69567f024150f2663da1d8d0

link https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
"""

# Define the model name
# model_name = "mask_rcnn_R_50_FPN_1x"
# model_name = "mask_rcnn_R_50_C4_3x"
# model_name = "mask_rcnn_R_50_DC5_3x"
# model_name = "mask_rcnn_R_50_FPN_3x"
model_name = "mask_rcnn_R_101_FPN_3x"
desired_number_of_epochs = 200
num_of_class = 1
is_resume = False
dataset_base = "Textile-defect-seg-uni-2"
robo_api_key = "R8JQyQmyXKa0HMr7HzJT"
robo_ws = "ds-gfhaj"
robo_project = "textile-defect-seg-uni"
dataset_version = 2
dataset_format = "coco-segmentation"


# Set the output directory based on the model name
import os
output_dir = f"Detectron2_Models/{model_name}"
os.makedirs(output_dir, exist_ok=True)

from roboflow import Roboflow
rf = Roboflow(api_key=robo_api_key)
project = rf.workspace(robo_ws).project(robo_project)
version = project.version(dataset_version)
dataset = version.download(dataset_format)

import torch, detectron2
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)


import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

register_coco_instances("my_dataset_train", {}, f"{dataset_base}/train/_annotations.coco.json",
                        f"{dataset_base}/train")

register_coco_instances("my_dataset_val", {}, f"{dataset_base}/valid/_annotations.coco.json",
                        f"{dataset_base}/valid")

register_coco_instances("my_dataset_test", {}, f"{dataset_base}/test/_annotations.coco.json",
                        f"{dataset_base}/test")



train_metadata = MetadataCatalog.get("my_dataset_train")
train_dataset_dicts = DatasetCatalog.get("my_dataset_train")

val_metadata = MetadataCatalog.get("my_dataset_val")
val_dataset_dicts = DatasetCatalog.get("my_dataset_val")

test_metadata = MetadataCatalog.get("my_dataset_test")
test_dataset_dicts = DatasetCatalog.get("my_dataset_test")



import pickle
metadata = MetadataCatalog.get("my_dataset_train")
train_metadata_path = os.path.join(output_dir, "train_metadata.pkl")
with open(train_metadata_path, "wb") as f:
    pickle.dump(metadata, f)
print("Metadata saved to train_metadata.pkl")

from matplotlib import pyplot as plt

img_plot = "inference_image_path_here"
for d in random.sample(train_dataset_dicts, 2):
  plt.figure(figsize=(30, 15))
  img_plot = d["file_name"]
  img = cv2.imread(img_plot)
  visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.5)
  vis = visualizer.draw_dataset_dict(d)
  plt.imshow(vis.get_image()[:, :, ::-1])
  plt.show()


import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import yaml



# Initialize configuration
cfg = get_cfg()
cfg.OUTPUT_DIR = output_dir
cfg.merge_from_file(model_zoo.get_config_file(f"COCO-InstanceSegmentation/{model_name}.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_test",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-InstanceSegmentation/{model_name}.yaml")
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.00025

num_images = len(train_dataset_dicts)
iterations_per_epoch = num_images // cfg.SOLVER.IMS_PER_BATCH
cfg.SOLVER.MAX_ITER = iterations_per_epoch * desired_number_of_epochs

cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024
cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_of_class


config_yaml_path = os.path.join(output_dir, "training_config.yaml")
with open(config_yaml_path, 'w') as file:
    yaml.dump(cfg, file)


trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=is_resume)
trainer.train()


model_weights_path = os.path.join(output_dir, "model_final.pth")
cfg.MODEL.WEIGHTS = model_weights_path


from detectron2.engine import DefaultPredictor
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
predictor = DefaultPredictor(cfg)


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("my_dataset_test", output_dir=output_dir)
val_loader = build_detection_test_loader(cfg, "my_dataset_test")
inference_on_dataset(predictor.model, val_loader, evaluator)


import cv2
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer


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


import yaml
config_yaml_path = os.path.join(output_dir, "inference_config.yaml")
with open(config_yaml_path, 'w') as file:
    yaml.dump(cfg, file)
