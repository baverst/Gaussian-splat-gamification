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


predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
SamOnePredictor = SamPredictor(sam)

import cv2



image = cv2.imread("../../../Downloads/tandt_db/tandt/truck/images/000001.jpg")
allsubmasks = []
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
i = 0
for mask in masks:
    i += 1
    print(i)
    input_point = np.array(mask["point_coords"])
    input_label = np.array([1])
    SamOnePredictor.set_image(image)
    submasks, scores, logits = SamOnePredictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    submasks = submasks[scores > 0.951]
    for tmp in submasks:
        allsubmasks.append(tmp)

unique_subs = []
for tmp in allsubmasks:
    allowed = True
    for compare in unique_subs:
        intersection = np.logical_and(tmp, compare).sum()
        union = np.logical_or(tmp, compare).sum()
        iou = 0
        if union != 0:
            iou = intersection / union
        if iou > 0.8:
            allowed = False
            break
    if allowed:
        unique_subs.append(tmp)

from dataclasses import dataclass

@dataclass
class LayeredMask:
    mask: np.ndarray
    subobjects: list['LayeredMask']

def is_contained(child, parent, threshold=0.95):
    intersection = (child & parent).sum()
    child_area = child.sum()
    return (intersection / child_area) >= threshold if child_area != 0 else False

sorted_masks = sorted(unique_subs, key=lambda m: np.count_nonzero(m), reverse=True)

layered_masks = []
mask_to_layered = []
for mask_i, mask in enumerate(sorted_masks):
    layered_obj = LayeredMask(mask,[])
    mask_to_layered.append(layered_obj)
    foundparent = False
    for bigger_j in range(mask_i-1,-1,-1):
        bigger_mask = sorted_masks[bigger_j]
        if(is_contained(mask, bigger_mask)):
            mask_to_layered[bigger_j].subobjects.append(layered_obj)
            foundparent = True
            break
    if not foundparent:
        layered_masks.append(layered_obj)





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
print_nr = 0

def recursiveMask(maskPrompt : LayeredMask):
    global print_nr
    print_nr += 1
    print(print_nr)
    predictor.reset_state(inference_state)

    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        inference_state,
            0,
            1,
            maskPrompt.mask,
    )

    # run propagation throughout the video and collect the results in a dict
    video_segments = {'masks': {}, 'subsegments' : []}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments['masks'][out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    for subprompt in maskPrompt.subobjects:
        video_segments['subsegments'].append(recursiveMask(subprompt))

    return video_segments


layered_video_segs = []
for mask in layered_masks:
    layered_video_segs.append(recursiveMask(mask))

import pickle
import os

def save_segment_recursive(segment, path, name):
    os.makedirs(path, exist_ok=True)

    segment_to_save = {
        "masks": segment["masks"],
        "subsegments": []
    }

    for i, sub in enumerate(segment["subsegments"]):
        sub_name = f"{name}_sub{i}"
        save_segment_recursive(sub, path, sub_name)
        segment_to_save["subsegments"].append(f"{sub_name}.pkl")

    with open(os.path.join(path, f"{name}.pkl"), "wb") as f:
        pickle.dump(segment_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_all_roots(roots, path):
    os.makedirs(path, exist_ok=True)
    for i, root in enumerate(roots):
        save_segment_recursive(root, path, f"root_{i}")

save_all_roots(layered_video_segs, "video_mask_tree/")
