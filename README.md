![Unity](https://img.shields.io/badge/engine-Unity-green)
![Python](https://img.shields.io/badge/python-3.10-blue)
![Python](https://img.shields.io/badge/python-3.9-blue)

# Multi-Splat Representations: Segmentation and Object Manipulation in Gaussian Splatting

This repository contains the code and tools developed as part of my master's thesis, which extends Gaussian splatting for real-time rendering and animation in Unity.

## Overview

This project introduces:
-  Global per-splat sorting in Unity for artifact-free rendering of overlapping Gaussian splats
-  A semi-automatic segmentation pipeline that combines [SAM](https://github.com/facebookresearch/segment-anything), [SAM2](https://github.com/facebookresearch/sam2), and [SAGD](https://github.com/XuHu0529/SAGS)

The full thesis is available in [`/Paper/Masterthesis.pdf`](./Paper/Masterthesis.pdf)

---

##  Repository Structure

```
/SAGS/ # Part of the segmentation pipeline (Includes SAM as a Git submodule in /SAGS/gaussiansplatting/dependencies/sam_ckpt/)
/sam2/ # Part of the segmentation pipeline
/UnityGaussianSplatting-main/ # Modified Unity extension with per-splat rendering
/Paper/ # Contains the final thesis PDF
environment-wsl.yml # WSL-based Python environment
environment-powershell.yml # Windows-based Python environment
```

---

##  Installation

This project uses multiple Conda environments across both WSL and Windows PowerShell.

### 1. Clone the repository on Powershell (with submodules)

```powershell
git clone --recurse-submodules https://github.com/baverst/Gaussian-splat-gamification.git
cd Gaussian-splat-gamification
```

### 2. Create Conda environments

####  On PowerShell (Windows):

```powershell
conda env create -f environment-powershell.yml
conda activate powershell-env

cd .\SAGS\gaussiansplatting\submodules\

git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

pip install ./simple-knn

cd ../dependencies/sam_ckpt/segment-anything

# Installing Grounding-DINO
git clone https://github.com/IDEA-Research/GroundingDINO.git

# Major issues with path length being to large
$orig = Resolve-Path "."
Copy-Item GroundingDINO "$env:USERPROFILE\GroundingDINO_temp" -Recurse
cd "$env:USERPROFILE\GroundingDINO_temp"
pip install -e "$orig\GroundingDino"
cd "$orig\GroundingDino"
Remove-Item "$env:USERPROFILE\GroundingDINO_temp" -Recurse -Force
mkdir weights;
```
Additionally, install [the weights ](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) (~661 MB) and copy the file to \SAGS\gaussiansplatting\dependencies\sam_ckpt\segment-anything\GroundingDINO\weights\

Finally, to use SAM, you need a model checkpoint (e.g., [`sam_vit_h.pth`](https://github.com/facebookresearch/segment-anything/tree/dca509fe793f601edb92606367a655c15ac00fdf#model-checkpoints), ~2.38GB). Place the model checkpoint in /SAGS/gaussiansplatting/dependencies/sam_ckpt/. 

#### On WSL (Linux):

```bash
conda env create -f environment-wsl.yml
conda activate wsl-env
cd sam2/checkpoints/
sudo apt install dos2unix
dos2unix download_ckpts.sh
./download_ckpts.sh
cd ../..
```
---
The datasets can be found found at [the github of the original Gaussian Splatting paper](https://github.com/graphdeco-inria/gaussian-splatting). The "T&T+DB COLMAP (650MB)" and the "Pre-trained Models (14 GB)" were both used in this research.
## Usage


### Unity Project and Scene

- Project path:  
  `UnityGaussianSplatting-main/UnityGaussianSplatting-main/projects/GaussianExample`

- Scene used:  
  `Assets/GSTestScene.unity` (inside the path above)
---

### Segmentation Pipeline
The segmentation pipeline can be run at three levels of automation, depending on how much user control is needed.
- **Option 1: Manual + hierarchical prompts** 
- **Option 2: Multi-mask automation** 
- **Option 3: Full automation** 
  
#### Option 1: Manual + Hierarchical Prompts (Maximum control)

```bash
# In WSL
cd sam2
python create_masks.py
```

```powershell
# In PowerShell
cd SAGS
python .\create_ply.py
```

> In Unity:  
> `Tools > Gaussian Splats > Create Hierarchical GaussianSplatAsset`

---

#### Option 2: Multi-mask Automation

```bash
# In WSL
cd sam2
python create_masks2.py
```

```powershell
# In PowerShell
cd SAGS
python .\create_ply.py
```

> In Unity:  
> `Tools > Gaussian Splats > Create Hierarchical GaussianSplatAsset`

---

#### Option 3: Full Automation

```bash
# In WSL
cd sam2
python full_pipeline.py
```

```powershell
# In PowerShell
cd SAGS
python .\pipeline_create_ply.py
```

> In Unity:  
> `Tools > Gaussian Splats > Create Hierarchical GaussianSplatAsset`

---

## References

- [Unity Gaussian Splatting](https://github.com/aras-p/UnityGaussianSplatting)
- [SAM](https://github.com/facebookresearch/segment-anything)
- [SAM2](https://github.com/facebookresearch/sam2)
- [SAGD](https://github.com/XuHu0529/SAGS)

---

## License & Citation

If using this work for research or development, please cite:

> Bavo Verstraeten. *Multi-Splat Representations: Segmentation and Object Manipulation in Gaussian Splatting*. Master's Thesis, Ghent University, 2025.  
> [`Paper/Masterthesis.pdf`](./Paper/Masterthesis.pdf)
