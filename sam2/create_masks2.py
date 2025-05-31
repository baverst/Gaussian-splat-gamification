import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


from sam2.build_sam import build_sam2_video_predictor
from segment_anything import sam_model_registry, SamPredictor

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam_checkpoint = "../SAGS/gaussiansplatting/dependencies/sam_ckpt/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
SamOnePredictor = SamPredictor(sam)

import cv2

input_point = np.array([[413.015625, 400.96875]]

)
input_label = np.array([1])
image = cv2.imread("../../../Downloads/tandt_db/tandt/truck/images/000001.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
SamOnePredictor.set_image(image)
masks, scores, logits = SamOnePredictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
masks = masks[scores > 0.9]


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "../../../Downloads/tandt_db/tandt/truck/images"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


inference_state = predictor.init_state(video_path=video_dir)

currentsub = None

for submask in masks:
    predictor.reset_state(inference_state)
    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        inference_state,
            0,
            1,
            submask,
    )
    video_segments = {'masks': {}, 'subsegments' : []}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments['masks'][out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    if currentsub is not None:
        video_segments['subsegments'].append(currentsub)

    currentsub = video_segments

import pickle


with open("../sam2_masks.pkl", "wb") as f:
    pickle.dump(currentsub, f)
