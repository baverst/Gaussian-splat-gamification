import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


from dataclasses import dataclass
from typing import Optional

@dataclass
class framePrompt:
    points: np.ndarray
    labels: np.ndarray
    frame_number: int


@dataclass
class LayeredMaskPrompt:
    frameprompts: list[framePrompt]
    subobjects: list['LayeredMaskPrompt']
## INSERT VIDEO DIRECTORY HERE #####################################################################

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "../../../Downloads/tandt_db/tandt/truck/images"

## INSERT PROMPT HERE ##############################################################################
boxprompt = LayeredMaskPrompt([framePrompt(np.array([[500,200],[500,350],[400,150],[580,350]], dtype=np.float32),
                               np.array([1,0,1,0], np.int32),
                               31)], [])

testprompt = LayeredMaskPrompt([framePrompt(np.array([[155,500],[500,220],[200,200]], dtype=np.float32),
                               np.array([1,1,1], np.int32),
                               31)], [boxprompt])
#####################################################################################################


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

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

inference_state = predictor.init_state(video_path=video_dir)

predictor.reset_state(inference_state)



def recursiveMask(maskPrompt : LayeredMaskPrompt):

    predictor.reset_state(inference_state)

    for frameprompt in maskPrompt.frameprompts:
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frameprompt.frame_number,
            obj_id=1,
            points=frameprompt.points,
            labels=frameprompt.labels,
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

import pickle

video_segments = recursiveMask(testprompt)

with open("../sam2_masks.pkl", "wb") as f:
    pickle.dump(video_segments, f)
