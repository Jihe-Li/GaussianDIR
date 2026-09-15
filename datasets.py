import SimpleITK as sitk
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import IterableDataset

import loaders
import utils


class RegDataset(IterableDataset):
    """This is a class for registrating implicitly represented images."""

    def __init__(self, case_idx, batch_size, neighs, loader: DictConfig, **kwargs):
        super().__init__()
        self.case_idx = case_idx
        self.batch_size = batch_size
        self.neighs = neighs
        # only used to convert the landmarks, see abs2rel and rel2abs
        self.offset = loader.pop('offset', 0)
        self.device = torch.device("cpu")

        load_func = utils.find(loaders, OmegaConf.create(loader))
        named_data = load_func(case_idx=self.case_idx)
        self.fix_arr = named_data["fix_arr"].unsqueeze(0).unsqueeze(0)
        self.mov_arrs = [
            move.unsqueeze(0).unsqueeze(0) for move in named_data["mov_arrs"]
        ]

        self.fix_mask = named_data["fix_mask"]
        self.mov_mask = named_data["mov_mask"]
        if "fix_oars" in named_data:
            self.fix_oars = named_data["fix_oars"]
            if "val_labels" in named_data:
                self.val_labels = named_data["val_labels"]
            else: 
                self.val_labels = torch.unique(self.fix_oars)[1:]
        if "mov_oars" in named_data:
            self.mov_oars = named_data["mov_oars"]
        if "fix_marks" in named_data:
            self.fix_marks = named_data["fix_marks"]
        if "mov_marks" in named_data:
            self.mov_marks = named_data["mov_marks"]
            
        self.params = named_data["params"]

        self.voxel_size = torch.FloatTensor(self.params["spacing"])
        self.image_size = torch.FloatTensor(self.params["size"])

        self.batch_center_num = self.batch_size // (self.neighs + 1)
        if self.neighs > 0:
            self.voxel_units = torch.eye(3)[None] * (2 / self.image_size)

        self.masked_coords = utils.make_coords(self.fix_mask.shape, self.fix_mask)
        self.shuffle()

    @property
    def shape(self):
        return self.fix_arr.shape[-3:]

    def arr2nii(self, array, save_path="", params=None):
        if params is None:
            params = self.params
        direction = params.get("direction", [1, 0, 0, 0, 1, 0, 0, 0, 1])
        origin = params.get("origin", [0, 0, 0])
        spacing = params.get("spacing", self.voxel_size.tolist())

        image = sitk.GetImageFromArray(array)
        image.SetDirection(direction)
        image.SetOrigin(origin)
        image.SetSpacing(spacing)

        if save_path:
            sitk.WriteImage(image, save_path)
        return image

    def abs2rel(self, coords, bias=True):
        if bias:
            return 2 * (coords + self.offset) / self.image_size - 1.0
        else:
            return 2 * coords / self.image_size

    def rel2abs(self, coords, bias=True):
        if bias:
            return (coords + 1.0) * self.image_size / 2 - self.offset
        else:
            return coords * self.image_size / 2

    def abs2phys(self, coords):
        return coords * self.voxel_size

    def shuffle(self):
        self.indices = torch.randperm(self.masked_coords.shape[0], device=self.device)
        self.iter_self = iter(range(0, len(self.indices), self.batch_center_num))

    def to(self, device):
        self.device = device
        self.fix_arr = self.fix_arr.to(device)
        self.mov_arrs = [move.to(device) for move in self.mov_arrs]
        self.fix_mask = self.fix_mask.to(device)
        self.mov_mask = self.mov_mask.to(device)
        if hasattr(self, "fix_marks"):
            self.fix_marks = self.fix_marks.to(device)
        if hasattr(self, "mov_marks"):
            self.mov_marks = self.mov_marks.to(device)
        if hasattr(self, "fix_oars"):
            self.fix_oars = self.fix_oars.to(device)
        if hasattr(self, "mov_oars"):
            self.mov_oars = self.mov_oars.to(device)
        if hasattr(self, "voxel_units"):
            self.voxel_units = self.voxel_units.to(device)
        if hasattr(self, "val_labels"):
            self.val_labels = self.val_labels.to(device)

        self.indices = self.indices.to(device)
        self.voxel_size = self.voxel_size.to(device)
        self.image_size = self.image_size.to(device)
        self.masked_coords = self.masked_coords.to(device)

        return self

    def __iter__(self):
        while True:
            try:
                idx = next(self.iter_self)
                coords = self.masked_coords[self.indices[idx: idx + self.batch_center_num]]
                if self.neighs == 3:
                    neigh_coords = coords[:, None] + self.voxel_units
                    neigh_coords = neigh_coords.reshape(-1, 3)
                    coords = torch.concat([coords, neigh_coords], dim=0)
                elif self.neighs == 6:
                    voxel_units = torch.concat([self.voxel_units, -self.voxel_units], dim=-2)
                    neigh_coords = coords[:, None] + voxel_units
                    neigh_coords = neigh_coords.reshape(-1, 3)
                    coords = torch.concat([coords, neigh_coords], dim=0)
                else:
                    pass
                
                yield coords

            except StopIteration:
                self.shuffle()
                continue

    def __len__(self):
        return len(self.indices)

    def masked_gather(self, tensor, mask=None, is_flow=False):
        if is_flow:
            full_size = self.shape + (3,)
            full_tensor = torch.zeros(full_size, device=self.device)
        else:
            full_size = self.shape
            full_tensor = torch.zeros(full_size, device=self.device) 
        if mask is None:
            mask = self.fix_mask
        full_tensor[mask] = tensor
        return full_tensor

    def __getitem__(self, index):
        return self.masked_coords[index]

    def _sampling(self, coords, tensor, mode="bilinear"):
        coords = coords.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return (
            F.grid_sample(tensor, coords, mode=mode, align_corners=False)
            .squeeze(0)
            .squeeze(0)
            .squeeze(0)
            .squeeze(0)
        )

    def samp_fix(self, coords):
        return self._sampling(coords, self.fix_arr)

    def samp_mov(self, coords, idx=-1):
        return self._sampling(coords, self.mov_arrs[idx])

    def samp_fix_oars(self, coords):
        return self._sampling(
            coords, self.fix_oars.unsqueeze(0).unsqueeze(0), "nearest"
        )

    def samp_mov_oars(self, coords):
        return self._sampling(
            coords, self.mov_oars.unsqueeze(0).unsqueeze(0), "nearest"
        )


class InterDataset(RegDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.half_voxel = 1 / self.image_size

    def to(self, device):
        super().to(device)
        self.half_voxel = self.half_voxel.to(device)

    def __iter__(self):
        while True:
            try:
                idx = next(self.iter_self)
                ori_coords = self.masked_coords[
                    self.indices[idx : idx + self.batch_size]
                ]
                noise = torch.rand_like(ori_coords, device=self.device) * 2 - 1
                yield ori_coords + noise * self.half_voxel
            except StopIteration:
                self.shuffle()
                continue
