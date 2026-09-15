from functools import partial

import torch
from omegaconf import OmegaConf


def create(module, cfg, *args, **kwargs):
    cls_ = module.__dict__[cfg.type]
    params = OmegaConf.to_container(cfg, resolve=True)
    del params["type"]
    return cls_(*args, **params)


def find(module, cfg):
    func = module.__dict__[cfg.type]
    params = OmegaConf.to_container(cfg, resolve=True)
    del params["type"]
    return partial(func, **params)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_coords(shape, mask=None):
    """Make a coordinate tensor."""

    coords = [torch.linspace(-1, 1, size + 1)[:-1] + 1 / size for size in shape]
    coords = torch.meshgrid(*coords, indexing="ij")
    coords = torch.stack(coords[::-1], dim=-1)
    coords = coords.view(-1, 3)

    if mask is not None:
        coords = coords[mask.flatten()]

    return coords
