import os

import numpy as np
from PIL import Image

import torch

from torch.utils.tensorboard import SummaryWriter


def set_require_grads(model, require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)


def create_grid(h, w, device, min_v=0, max_v=1):
    grid_y, grid_x = torch.meshgrid([torch.linspace(min_v, max_v, steps=h),
                                     torch.linspace(min_v, max_v, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid.to(device)


def read_img(path):
    img = np.array(Image.open(path).convert('RGB')) / 255.
    return img


def sample_B(mapping_size, scale, device):
    B_gauss = torch.randn((mapping_size, 2)).to(device) * scale
    return B_gauss


def make_exp_dirs(exp_name, log=True):
    os.makedirs(f'exps/{exp_name}/img', exist_ok=True)
    os.makedirs(f'exps/{exp_name}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{exp_name}/logs', exist_ok=True)
    if log:
        return SummaryWriter(f'exps/{exp_name}/logs')


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")