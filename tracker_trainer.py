# -*- coding:utf8 -*-
# !/usr/bin/env python3
# Copyright 2025 Octadero.
# Author Volodymyr Pavliukevych

"""This is a Ylo8 implementation trainer
"""

import os
import os
import cv2
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np
import shutil

# Load YOLOv8 model (you can also specify another model, e.g., 'yolov8m.pt')
model_names = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
model_name = "models/yolov8x.pt"

# Set your input and output directories
input_dir = 'datasets/train/'
output_dir = f'datasets/train_{model_name}/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

root_path = os.getcwd()
model = YOLO(os.path.join(root_path, model_name))

# progress = tqdm(os.listdir(os.path.join(input_dir, 'labels', 'train')))
# # Iterate over files in the input directory
# for filename in progress:
#     image_file_name = filename.replace('.txt', '.jpg')
#     src_image_file_path = os.path.join(input_dir, "images", "full", image_file_name)
#     dst_image_file_path = os.path.join(input_dir, "images", "train", image_file_name)
#     shutil.copyfile(src_image_file_path, dst_image_file_path)
#     progress.set_description(f"{image_file_name}")

model.train(data="datasets/train/data.yaml", epochs=50, imgsz=640)

# Print all class IDs and names
for class_id, class_name in model.names.items():
    print(f"{class_id}: {class_name}")

# results = model.predict(input_dir, classes=[0,1,2], save_txt=True)

# # Supported image extensions
# image_extensions = ('.jpg', '.jpeg', '.png')
# progress = tqdm(os.listdir(input_dir))
# # Iterate over files in the input directory
# for filename in progress:
#     if filename.lower().endswith(image_extensions):
#         img_path = os.path.join(input_dir, filename)
#         progress.set_description(f"{filename}")
#         # Run YOLO detection
#         results = model(img_path)[0]

#         # Load image with OpenCV for drawing
#         image = cv2.imread(img_path)

#         # Draw bounding boxes and labels
#         for box in results.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = box.conf[0]
#             cls_id = int(box.cls[0])
#             label = model.names[cls_id]
#             color = (0, 255, 0)

#             cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#         # Save annotated image
#         out_path = os.path.join(output_dir, filename)
#         cv2.imwrite(out_path, image)

