import torch

from gaussiansplatting.scene.gaussian_model import GaussianModel
from gaussiansplatting.scene import Scene
from gaussiansplatting.gaussian_renderer import render

from argparse import ArgumentParser
from gaussiansplatting.arguments import ModelParams, PipelineParams

from argparse import ArgumentParser, Namespace

import os

def get_combined_args(parser : ArgumentParser, model_path):
    # cmdlne_string = sys.argv[1:]
    # cfgfile_string = "Namespace()"
    cmdlne_string = ['--model_path', model_path]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
# op = OptimizationParams(parser)
parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--quiet", action="store_true")

parser.add_argument("--threshold", default=0.7, type=float, help='threshold of label voting')
parser.add_argument("--gd_interval", default=20, type=int, help='interval of performing gaussian decomposition')

## choose your trained 3D-GS model path
model_path = r"C:\Users\verst\Downloads\tandt_db\tandt\truck"
args = get_combined_args(parser, model_path)

# Initialize system state (RNG)
# safe_state(args.quiet)

# 3D gaussians
dataset = model.extract(args)
dataset.eval = False
dataset.model_path = args.model_path
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

cameras = scene.getTrainCameras()

dataset.white_background = True
bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

xyz = gaussians.get_xyz

from PIL import Image
import numpy as np

folder_path = r"C:\Users\verst\Downloads\tandt_db\tandt\truck\images"
image_files = sorted(os.listdir(folder_path), key=lambda x: int(x.split('.')[0]))

first_image = Image.open(os.path.join(folder_path, image_files[0]))
image_shape = np.array(first_image).shape

images_3d_array = np.zeros((len(image_files), *image_shape), dtype=np.uint8)


for i, image_file in enumerate(image_files):
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path)
    images_3d_array[i] = np.array(image)

import torch.nn.functional as F
import torchvision.transforms.functional as func

from seg_utils import conv2d_matrix, compute_ratios, update

def get_3d_prompts(prompts_2d, point_image, xyz, depth=None):
    r = 4
    x_range = torch.arange(prompts_2d[0] - r, prompts_2d[0] + r)
    y_range = torch.arange(prompts_2d[1] - r, prompts_2d[1] + r)
    x_grid, y_grid = torch.meshgrid(x_range, y_range)
    neighbors = torch.stack([x_grid, y_grid], dim=2).reshape(-1, 2).to("cuda")
    prompts_index = [torch.where((point_image == p).all(dim=1))[0] for p in neighbors]
    indexs = []
    for index in prompts_index:
        if index.nelement() > 0:
            indexs.append(index)
    indexs = torch.unique(torch.cat(indexs, dim=0))
    indexs_depth = depth[indexs]
    valid_depth = indexs_depth[indexs_depth > 0]
    _, sorted_indices = torch.sort(valid_depth)
    valid_indexs = indexs[depth[indexs] > 0][sorted_indices[0]]
    
    return xyz[valid_indexs][:3].unsqueeze(0)

## Given 1st view point prompts, find corresponding 3D Gaussian point prompts
def generate_3d_prompts(xyz, viewpoint_camera, prompts_2d):
    w2c_matrix = viewpoint_camera.world_view_transform
    full_matrix = viewpoint_camera.full_proj_transform
    # project to image plane
    xyz = F.pad(input=xyz, pad=(0, 1), mode='constant', value=1)
    p_hom = (xyz @ full_matrix).transpose(0, 1)  # N, 4 -> 4, N
    p_w = 1.0 / (p_hom[-1, :] + 0.0000001)
    p_proj = p_hom[:3, :] * p_w
    # project to camera space
    p_view = (xyz @ w2c_matrix[:, :3]).transpose(0, 1)  # N, 3 -> 3, N
    depth = p_view[-1, :].detach().clone()
    valid_depth = depth >= 0

    h = viewpoint_camera.image_height
    w = viewpoint_camera.image_width

    point_image = 0.5 * ((p_proj[:2] + 1) * torch.tensor([w, h]).unsqueeze(-1).to(p_proj.device) - 1)
    point_image = point_image.detach().clone()
    point_image = torch.round(point_image.transpose(0, 1)).long()

    prompts_2d = torch.tensor(prompts_2d).to("cuda")
    prompts_3d = []
    for i in range(prompts_2d.shape[0]):
        prompts_3d.append(get_3d_prompts(prompts_2d[i], point_image, xyz, depth))
    prompts_3D = torch.cat(prompts_3d, dim=0)

    return prompts_3D

## Project 3D points to 2D plane
def porject_to_2d(viewpoint_camera, points3D):
    full_matrix = viewpoint_camera.full_proj_transform  # w2c @ K 
    # project to image plane
    if points3D.shape[-1] != 4:
        points3D = F.pad(input=points3D, pad=(0, 1), mode='constant', value=1)
    p_hom = (points3D @ full_matrix).transpose(0, 1)  # N, 4 -> 4, N   -1 ~ 1
    p_w = 1.0 / (p_hom[-1, :] + 0.0000001)
    p_proj = p_hom[:3, :] * p_w

    h = viewpoint_camera.image_height
    w = viewpoint_camera.image_width

    point_image = 0.5 * ((p_proj[:2] + 1) * torch.tensor([w, h]).unsqueeze(-1).to(p_proj.device) - 1) # image plane
    point_image = point_image.detach().clone()
    point_image = torch.round(point_image.transpose(0, 1))

    return point_image

## Single view assignment
def mask_inverse(xyz, viewpoint_camera, sam_mask):
    w2c_matrix = viewpoint_camera.world_view_transform
    # project to camera space
    xyz = F.pad(input=xyz, pad=(0, 1), mode='constant', value=1)
    p_view = (xyz @ w2c_matrix[:, :3]).transpose(0, 1)  # N, 3 -> 3, N
    depth = p_view[-1, :].detach().clone()
    valid_depth = depth >= 0

    h = viewpoint_camera.image_height
    w = viewpoint_camera.image_width
    

    if sam_mask.shape[0] != h or sam_mask.shape[1] != w:
        sam_mask = func.resize(sam_mask.unsqueeze(0), (h, w), antialias=True).squeeze(0).long()
    else:
        sam_mask = sam_mask.long()

    point_image = porject_to_2d(viewpoint_camera, xyz)
    point_image = point_image.long()

    # 判断x,y是否在图像范围之内
    valid_x = (point_image[:, 0] >= 0) & (point_image[:, 0] < w)
    valid_y = (point_image[:, 1] >= 0) & (point_image[:, 1] < h)
    valid_mask = valid_x & valid_y & valid_depth
    point_mask = torch.full((point_image.shape[0],), 0).to("cuda")

    point_mask[valid_mask] = sam_mask[point_image[valid_mask, 1], point_image[valid_mask, 0]]
    indices_mask = torch.where(point_mask == 1)[0]

    return point_mask, indices_mask

## Multi-view label voting
def ensemble(multiview_masks, threshold=0.7):
    # threshold = 0.7
    filtered_masks = [m for m in multiview_masks if m.any()]

    multiview_masks = torch.cat(filtered_masks, dim=1)
    vote_labels,_ = torch.mode(multiview_masks, dim=1)
    # # select points with score > threshold 
    matches = torch.eq(multiview_masks, vote_labels.unsqueeze(1))
    ratios = torch.sum(matches, dim=1) / multiview_masks.shape[1]
    ratios_mask = ratios > threshold
    labels_mask = (vote_labels == 1) & ratios_mask
    indices_mask = torch.where(labels_mask)[0].detach().cpu()

    return vote_labels, indices_mask

## Gaussian Decomposition
def gaussian_decomp(gaussians, viewpoint_camera, input_mask, indices_mask):
    xyz = gaussians.get_xyz
    point_image = porject_to_2d(viewpoint_camera, xyz)

    conv2d = conv2d_matrix(gaussians, viewpoint_camera, indices_mask, device="cuda")
    height = viewpoint_camera.image_height
    width = viewpoint_camera.image_width
    index_in_all, ratios, dir_vector = compute_ratios(conv2d, point_image, indices_mask, input_mask, height, width)

    decomp_gaussians = update(gaussians, viewpoint_camera, index_in_all, ratios, dir_vector)

    return decomp_gaussians

import pickle



print("Start Segmentation...")

from plyfile import PlyData, PlyElement
import copy

def save_gs(pc, indices_mask, save_path):
    
    xyz = pc._xyz.detach().cpu()[indices_mask].numpy()
    normals = np.zeros_like(xyz)
    f_dc = pc._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu()[indices_mask].numpy()
    f_rest = pc._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu()[indices_mask].numpy()
    opacities = pc._opacity.detach().cpu()[indices_mask].numpy()
    scale = pc._scaling.detach().cpu()[indices_mask].numpy()
    rotation = pc._rotation.detach().cpu()[indices_mask].numpy()

    dtype_full = [(attribute, 'f4') for attribute in pc.construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(save_path)

mask_id = 1

# generate 3D prompts
xyz = gaussians.get_xyz
import os

savename = 'point_cloud_seg.ply'
subfoldername = 'sub'
def process_prompt(current_prompt, current_path):
    global gaussians
    multiview_masks = []
    renders = []
    i = 0
    for view in cameras:
        if i % 10 != 0:
            i+=1
            continue

        sam_mask = current_prompt['masks'][i][1][0]
        if len(sam_mask.shape) != 2:
            sam_mask = torch.from_numpy(sam_mask).squeeze(-1).to("cuda")
        else:
            sam_mask = torch.from_numpy(sam_mask).to("cuda")
        sam_mask = sam_mask.long()
        
        # mask assignment to gaussians
        point_mask, indices_mask = mask_inverse(xyz, view, sam_mask)
        multiview_masks.append(point_mask.unsqueeze(-1))
        i+=1

        # # gaussian decomposition as an intermediate process
        #if args.gd_interval != -1 \
        #                     and i % args.gd_interval == 0:  # 
        #     gaussians = gaussian_decomp(gaussians, view, sam_mask, indices_mask)

    # multi-view label ensemble
    torch.cuda.empty_cache()
    multiview_masks = [t.cpu() for t in multiview_masks]
    _, final_mask = ensemble(multiview_masks, threshold=0.6)
    final_mask.cuda()

    layered_mask = torch.ones(final_mask.size(0), dtype=torch.bool)

    subpath = os.path.join(current_path, subfoldername)

    if len(current_prompt['subsegments']) > 0:
        for index, subsegment in enumerate(current_prompt['subsegments']):
            segpath = subpath+str(index)
            os.makedirs(segpath, exist_ok=True)
            submask = process_prompt(subsegment,segpath)
            layered_mask = layered_mask & ~torch.isin(final_mask, submask)
    # save before gaussian decomposition
    save_gs(gaussians, final_mask[layered_mask], os.path.join(current_path, savename))
    del multiview_masks, point_mask, sam_mask
    torch.cuda.empty_cache()
    return final_mask

def process_loaded_prompt(loaded_data, path):
    subpath = os.path.join(path, subfoldername)
    global_mask = torch.empty(0, dtype=torch.long)

    if len(loaded_data) > 0:
        layered_mask = torch.ones(xyz.size(0), dtype=torch.bool)
        final_mask = torch.where(layered_mask == 1)[0].detach().cpu()
        final_mask.cuda()
        for index, subsegment in enumerate(loaded_data):
            segpath = subpath+str(index)
            os.makedirs(segpath, exist_ok=True)
            submask = process_prompt(subsegment,segpath)
            layered_mask = layered_mask & ~torch.isin(final_mask, submask)
        global_mask = final_mask[layered_mask]

    # save before gaussian decomposition
    save_gs(gaussians, global_mask, os.path.join(path, savename))

save_path = '../layered_ply/'
os.makedirs(save_path, exist_ok=True)

def load_segment_recursive(path, name):
    with open(os.path.join(path, f"{name}.pkl"), "rb") as f:
        segment = pickle.load(f)

    subsegments = []
    for sub_file in segment["subsegments"]:
        sub_name = os.path.splitext(sub_file)[0]
        subsegments.append(load_segment_recursive(path, sub_name))

    segment["subsegments"] = subsegments
    return segment

import re
def load_all_roots(path):
    root_files = [f for f in os.listdir(path) if f.startswith("root_") and not "_sub" in f and f.endswith(".pkl")]
    root_files.sort(key=lambda x : int(re.match(r"root_(\d+)\.pkl", x).group(1)))
    return [load_segment_recursive(path, os.path.splitext(f)[0]) for f in root_files]
loaded_video_masks = load_all_roots("../sam2/notebooks/video_mask_tree/")

process_loaded_prompt(loaded_video_masks, save_path)
