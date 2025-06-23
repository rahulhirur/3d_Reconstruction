# 3d_Reconstruction
An application to construct 3D point cloud using stereo images

🚀 Getting Started
The following needs to be accomplished to be able to run this on a local web server.

## Prerequisites
- Python 3.11 recommended
- Git

The next steps might be optional if you already have a virtual environment.

## Create a Python environment

### <img src="https://www.python.org/static/favicon.ico" width="20" height="20">  Option 1: Using `venv`
```bash
python -m venv 3D_construction_env
```
### 🐍 Option 2: Using `conda`
```bash
conda create -n 3D_construction_env python=3.11
```

## Activate environment

### On Linux/macOS

#### If using `venv`:
```bash
source 3D_construction_env/bin/activate
```

#### If using `conda`:
```bash
conda activate 3D_construction_env
```
### On Windows

#### If using `venv`:
```bash
3D_construction_env\Scripts\activate
```
#### If using `conda`:
```bash
conda activate 3D_construction_env
```
## Installation

Clone the repository:
```bash
git clone --recurse-submodules https://github.com/rahulhirur/3d_Reconstruction.git
cd 3d_Reconstruction
```

Install the torch dependencies:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```
