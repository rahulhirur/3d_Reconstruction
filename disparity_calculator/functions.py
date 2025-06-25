import os
import sys
import skimage
import logging
import pickle

import torch
import imageio
import cv2
import numpy as np

from omegaconf import OmegaConf
import plotly.graph_objects as go
import albumentations

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from FoundationStereo.core.utils.utils import InputPadder
from FoundationStereo.Utils import *
from FoundationStereo.core.foundation_stereo import *

def initialize_environment():
    code_dir = "FoundationStereo/"
    sys.path.append(f'{code_dir}/../')
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

# Create a function to create output directory if it doesn't exist
def create_output_directory(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logging.info(f"Output directory created at: {out_dir}")

def load_configuration():
    
    config_file = f'FoundationStereo/cfg.yaml'
    run_config_file = f'FoundationStereo/run_configuration.yaml'

    cfg = OmegaConf.load(config_file)
    run_cfg = OmegaConf.load(run_config_file)
    return OmegaConf.merge(cfg, run_cfg)

def initialize_model(args):
    ckpt_dir = "FoundationStereo/pretrained_models/model_best_bp2.pth"
    logging.info(f"args:\n{args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")
    model = FoundationStereo(args)
    if torch.cuda.is_available():
        ckpt = torch.load(ckpt_dir)
    else:
        ckpt = torch.load(ckpt_dir, map_location=torch.device('cpu'))
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")

    model.load_state_dict(ckpt['model'])

    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    model.eval()
    
    return model

def preprocess_images(left_file, right_file, scale):
    
    
    img0 = process_individual_image(left_file, scale)

    img1 = process_individual_image(right_file, scale)

    return img0, img1

def process_individual_image(file_name,scale):

    imgx = read_image(file_name)

    if imgx.shape[-1] == 4:
        imgx = imgx[:, :, :3]
    imgx = cv2.resize(imgx, fx=scale, fy=scale, dsize=None)

    return imgx

def read_image(file):
    if isinstance(file, str):
        return cv2.imread(file)
    elif hasattr(file, 'read'):
        return cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    else:
        return None

def prepare_images_for_model(img0, img1):
    """
    Prepares images for model input by converting them to tensors and padding them.

    Args:
        img0 (numpy.ndarray): First image.
        img1 (numpy.ndarray): Second image.

    Returns:
        tuple: Padded and converted images (img0, img1).
    """
    if torch.cuda.is_available():
        img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    else:
        img0 = torch.as_tensor(img0).float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(img1).float()[None].permute(0, 3, 1, 2)
    
    return img0, img1

def compute_disparity(model, img0, img1, args, unpad=True):

    H,W = img0.shape[:2]

    img0, img1 = prepare_images_for_model(img0, img1)

    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    if torch.cuda.is_available():
        with torch.cuda.amp.autocast(True):
            if not args.hiera:
                disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
            else:
                disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
    else:
        if not args.hiera:

            disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
        else:

            disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
    
    if unpad:

        disp = padder.unpad(disp.float())
        return disp.data.cpu().numpy().reshape(H,W)

    else:

        disp = padder.unpad(disp.float()).cpu().numpy()
        return disp

def save_disparity(disp, file_path):

    """
    Save disparity map based on the file extension.

    Args:
        disp (numpy.ndarray): Disparity map to save.
        file_path (str): Path to save the file. The format is determined by the file extension.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".npy":
        np.save(file_path, disp)
        print(f"Disparity map saved as .npy at {file_path}")
    elif file_extension == ".pkl":
        with open(file_path, 'wb') as f:
            pickle.dump(disp, f)
        print(f"Disparity map saved as .pkl at {file_path}")
    else:
        raise ValueError("Unsupported file extension. Use '.npy' or '.pkl'.")

def vis_disparity(disp, min_val=None, max_val=None, invalid_thres=np.inf, color_map=cv2.COLORMAP_TURBO, cmap=None, other_output={}):
  
  """
  @disp: np array (H,W)
  @invalid_thres: > thres is invalid
  """
  disp = disp.copy()
  H,W = disp.shape[:2]
  invalid_mask = disp>=invalid_thres
  if (invalid_mask==0).sum()==0:
    other_output['min_val'] = None
    other_output['max_val'] = None
    return np.zeros((H,W,3))
  if min_val is None:
    min_val = disp[invalid_mask==0].min()
  if max_val is None:
    max_val = disp[invalid_mask==0].max()
  other_output['min_val'] = min_val
  other_output['max_val'] = max_val
  vis = ((disp-min_val)/(max_val-min_val)).clip(0,1) * 255
  if cmap is None:
    vis = cv2.applyColorMap(vis.clip(0, 255).astype(np.uint8), color_map)[...,::-1]
  else:
    vis = cmap(vis.astype(np.uint8))[...,:3]*255
  if invalid_mask.any():
    vis[invalid_mask] = 0
  return vis.astype(np.uint8)
