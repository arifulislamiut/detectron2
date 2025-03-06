import torch
import cv2
import os
import time
import pickle
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

model = "mask_rcnn_R_101_C4_3x_20250129"

# Paths
config_path = f"input/{model}/config.yaml"
pth_path = f"input/{model}/model_final.pth"
pkl_path = f"input/{model}/train_metadata.pkl"
video_path = "input/video/ad_short.mp4"
output_path = "output/"

# ✅ Check CUDA & Torch
print(f"CUDA Version: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")

# ✅ Load Detectron2 Config
cfg = get_cfg()
cfg.merge_from_file(config_path)
cfg.MODEL.WEIGHTS = pth_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.DEVICE = "cuda"  # Use GPU acceleration

print("✅ Model Config Loaded")

# ✅ Load the model predictor
predictor = DefaultPredictor(cfg)

# ✅ Load Metadata
with open(pkl_path, "rb") as f:
    metadata = pickle.load(f)
print("✅ Metadata Loaded")

# ✅ Select Camera Source
use_camera = True  # Change to False if using USB camera or video file

if use_camera:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
else:
    cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Camera not found!")
    exit()

print("✅ Camera Initialized")

fps_time = time.perf_counter()
# ✅ Process Frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to grab frame!")
        break

    # Run model inference
    outputs = predictor(frame)

    # ✅ Visualization
    v = Visualizer(frame[:, :, ::-1], metadata=metadata)
    out_frame = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1]  # Convert to BGR

    # ✅ Display the frame
    cv2.imshow("Jetson Detectron2", out_frame)

    fps = 1.0 / (time.perf_counter() - fps_time)
    print("Net FPS: %f" % (fps))
    fps_time = time.perf_counter()

    # ✅ Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Process Finished")
