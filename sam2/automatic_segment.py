from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
from PIL import Image
import os
import cv2
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

sam_checkpoint = "../SAGS/gaussiansplatting/dependencies/sam_ckpt/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

video_dir = "../../../Downloads/tandt_db/tandt/truck/images"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
image = cv2.imread(os.path.join(video_dir, frame_names[0]))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image)

with open("../automatic_points.txt", "w") as f:
    f.write(f"Frame {0}\n")
    for i, prompt in enumerate([mask["point_coords"] for mask in masks]):
        f.write(f"# Prompt {i}\n")
        for x, y in prompt:
            f.write(f"{x}, {y}\n")
        f.write("\n")  
