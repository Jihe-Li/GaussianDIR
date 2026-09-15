import os
import time

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

import datasets
import losses
import networks
import utils
import metrics
from logger import logger


class GaussianRegistrator:
    """This is a class for registrating images with Gaussian primitives."""

    def __init__(self, cfg, dataset=None, network=None):
        """Initialize the learning model."""
        self.chunk_size = cfg.chunk_size
        self.log_freq = cfg.log_freq
        self.max_steps = cfg.max_steps
        self.warmup_steps = cfg.warmup_steps
        self.resume = cfg.resume
        self.save_dir = cfg.save_dir
        self.training = cfg.training
        self.use_wandb = cfg.use_wandb
        torch.manual_seed(cfg.seed)

        self.lambda_tv = cfg.regulizer.lambda_tv

        self.enable_densify = cfg.enable_densify  # whether to densify & prune
        if self.enable_densify:
            self.densify_from_iter = cfg.densify_from_iter            # iteration at which densify & prune starts
            self.densify_until_ratio = cfg.densify_until_ratio        # fraction of the schedule after which it stops
            self.densify_interval_ratio = cfg.densify_interval_ratio  # interval between two densify & prune steps

        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset: datasets.RegDataset = utils.create(datasets, cfg.dataset)
        if network is not None:
            self.network = network
        else:
            self.network: nn.Module = utils.create(networks, cfg.network)
        self.shape = self.dataset.shape

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset.to(self.device)
        self.network.reinitialize(self.dataset.fix_mask)  # drop the Gaussian primitives outside the mask
        self.network.to(self.device)

        if not self.training:
            self.network.eval()
        else:
            self.optimizer: optim.Optimizer = utils.create(
                optim, cfg.optimizer, self.network.trained_parameters(cfg.lr))
            self.lr = {}  # save initial learning rate
            for group in self.optimizer.param_groups:
                self.lr[group['name']] = group['lr']
            self.criterion = (
                losses.NCC()
                if cfg.loss.type.lower() == "ncc"
                else utils.create(nn, cfg.loss)
            )
            self.criterion.to(self.device)

            if self.use_wandb:
                wandb.init(
                    project="inr-time",
                    name=cfg.exp_name,
                    config=OmegaConf.to_container(cfg, resolve=True),
                )

        if self.resume:
            self.load()

    def close(self):
        if self.use_wandb:
            wandb.finish()

    def save(self):
        save_path = os.path.join(self.save_dir, "model.pth")
        torch.save(self.network.state_dict(), save_path)

    def load(self):
        save_path = os.path.join(self.save_dir, "model.pth")
        state_dict = torch.load(save_path, map_location=self.device, weights_only=True)
        try:
            self.network.load_state_dict(state_dict)
        except:
            for item in state_dict.items():
                setattr(self.network, item[0], nn.Parameter(torch.zeros_like(item[1])))
            self.network.load_state_dict(state_dict)

    def ODE(self, coords, forward=True):
        acc_flow = self.network(coords)
        if not forward:
            acc_flow = acc_flow * -1
        cur_coords = torch.add(acc_flow, coords)
        return acc_flow, cur_coords

    def tv_regulizer(self, acc_flow):
        center_flow = acc_flow[:acc_flow.shape[0] // (self.dataset.neighs + 1)]
        neighs_flow = acc_flow[acc_flow.shape[0] // (self.dataset.neighs + 1):].reshape(-1, self.dataset.neighs, 3)
        diff_norm = torch.norm(neighs_flow - center_flow[:, None], dim=-1)
        return diff_norm.mean()

    def train_step(self, step):
        """Perform one iteration of training."""
        coords = next(self.dataset_iter)
        with torch.no_grad():
            fix_val = self.dataset.samp_fix(coords)

        self.network.train()
        self.optimizer.zero_grad()
        acc_flow, tar_coords = self.ODE(coords)
        warp_val = self.dataset.samp_mov(tar_coords)

        loss = self.criterion(warp_val, fix_val)
        # tv regularization
        if self.lambda_tv > 0 and self.dataset.neighs != 0:
            loss += self.lambda_tv * self.tv_regulizer(acc_flow)
        loss.backward()

        self.adaptive_control(step, self.network, self.optimizer)
        self.optimizer.step()

    @torch.no_grad()
    def adaptive_control(self, step, network, optimizer):
        if self.enable_densify and step <= self.densify_until_ratio * self.max_steps \
                and self.densify_from_iter < self.max_steps * self.densify_until_ratio:  # first iteration < last iteration
            network.add_densification_stats()   # accumulate the gradients of the coordinates

            if step >= self.densify_from_iter and step % (self.densify_interval_ratio * self.max_steps) == 0:
                network.densify_and_prune(optimizer=optimizer, tag=self.log_tag(step))  # run densify & prune

    def get_cur_lr(self, step, lr):
        if step <= self.warmup_steps:
            cur_lr = step / self.warmup_steps * lr
        else:
            cur_lr = (
                (
                    math.cos(
                        math.pi
                        * (step - self.warmup_steps)
                        / (self.max_steps + 1 - self.warmup_steps)
                    )
                    + 1
                )
                / 2
                * lr
            )
        return cur_lr

    def train(self):
        """Train the network."""
        self.dataset_iter = iter(self.dataset)
        total_time = 0.0
        start_time = time.time()

        for step in tqdm(range(1, self.max_steps + 1), ncols=80):

            for group in self.optimizer.param_groups:
                group["lr"] = self.get_cur_lr(step, self.lr[group['name']]) 

            self.train_step(step)

            if step % self.log_freq == 0:
                pause_time = time.time()
                total_time += pause_time - start_time
                if hasattr(self.dataset, 'fix_marks'): # DIRLab
                    error, _ = self.val_marks(step)
                    if self.use_wandb:
                        wandb.log({"error": error[0], "time": total_time}, step=step)
                else:                                  # OASIS, ACDC
                    dice = self.val_dice(step)
                    if self.use_wandb:
                        wandb.log({"dice": dice, "time": total_time}, step=step)
                start_time = time.time()
        self.save()

    def log_tag(self, step=None):
        """Prefix the messages with the step while training and with the case index."""
        if step is None:
            return "Case%d" % self.dataset.case_idx
        return "[%*d/%d] Case%d" % (
            len(str(self.max_steps)), step, self.max_steps, self.dataset.case_idx
        )

    @torch.no_grad()
    def val_marks(self, step=None):
        self.network.eval()
        coords = self.dataset.abs2rel(self.dataset.fix_marks)
        _, tar_coords = self.ODE(coords.to(self.device))
        warp_marks = self.dataset.rel2abs(tar_coords)
        warp_marks = torch.round(warp_marks)

        mean, std = metrics.compute_landmark_accuracy(
            self.dataset.abs2phys(warp_marks),
            self.dataset.abs2phys(self.dataset.mov_marks),
        )
        message = "{} TRE (mm): {:.4f}±{:.4f}".format(
            self.log_tag(step), mean[0], std[0]
        )
        logger.info(message)
        return mean, std

    @torch.no_grad()
    def val_dice(self, step=None):
        self.network.eval()

        fix_oars = []
        warp_oars = []
        for i in range(0, len(self.dataset), self.chunk_size):
            coords_i = self.dataset[i : i + self.chunk_size]
            fix_oars_i = self.dataset.samp_fix_oars(coords_i)
            fix_oars.append(fix_oars_i)
            tar_coords_i = self.ODE(coords_i)[1]
            warp_oars_i = self.dataset.samp_mov_oars(tar_coords_i)
            warp_oars.append(warp_oars_i)

        fix_oars = torch.cat(fix_oars, dim=0)
        warp_oars = torch.cat(warp_oars, dim=0)
        dices = metrics.comp_dice(fix_oars, warp_oars, self.dataset.val_labels)

        message = "{} Dice: {:.4f}".format(
            self.log_tag(step), dices)
        logger.info(message)
        return dices

    @torch.no_grad()
    def val_hd95(self):
        self.network.eval()
        
        warp_oars = []
        for i in range(0, len(self.dataset), self.chunk_size):
            coords_i = self.dataset[i : i + self.chunk_size]
            tar_coords_i = self.ODE(coords_i)[1]
            warp_oars_i = self.dataset.samp_mov_oars(tar_coords_i)
            warp_oars.append(warp_oars_i)

        warp_oars = torch.cat(warp_oars, dim=0)
        warp_oars = self.dataset.masked_gather(warp_oars, is_flow=False)
        warp_oars = warp_oars.reshape(*self.shape)
        hd95 = metrics.comp_hd95(self.dataset.fix_oars, self.dataset.mov_oars, 
                     warp_oars, self.dataset.val_labels)
        message = "{} HD95: {:.4f}".format(
            self.log_tag(), hd95)
        logger.info(message)
        return hd95

    @torch.no_grad()
    def val_mae(self):
        self.network.eval()

        fix_vals = []
        warp_vals = []
        for i in range(0, len(self.dataset), self.chunk_size):
            coords_i = self.dataset[i : i + self.chunk_size]
            fix_vals_i = self.dataset.samp_fix(coords_i)
            fix_vals.append(fix_vals_i)
            tar_coords_i = self.ODE(coords_i)[1]
            warp_vals_i = self.dataset.samp_mov(tar_coords_i)
            warp_vals.append(warp_vals_i)

        fix_vals = torch.cat(fix_vals, dim=0)
        warp_vals = torch.cat(warp_vals, dim=0)
        mae = F.l1_loss(fix_vals, warp_vals)

        message = "{} MAE: {:.4f}".format(
            self.log_tag(), mae)
        logger.info(message)
        return mae

    @torch.no_grad()
    def val_folding_rate(self, out_jacobian_mask=False):
        """Return the folding rate for the given input-coordinates."""
        self.network.eval()

        flow = []
        for i in range(0, len(self.dataset), self.chunk_size):
            coords_i = self.dataset[i : i + self.chunk_size]
            flow_i = self.ODE(coords_i)[0]
            flow.append(flow_i)

        flow = torch.cat(flow, dim=0)
        flow = flow * self.dataset.image_size / 2
        flow = self.dataset.masked_gather(flow, is_flow=True)
        flow = flow.reshape(*self.shape, 3)
        flow = flow.flip(-1)
        flow = flow.permute(3, 0, 1, 2).unsqueeze(0)
        fold, jacobian_mask = metrics.comp_folding_rate(flow, out_jacobian_mask)

        message = "{} JacDet: {:.6f}".format(
            self.log_tag(), fold.cpu().numpy())
        logger.info(message)
        return jacobian_mask

    @torch.no_grad()
    def inf_flow(self):
        """Return the deformation field for the given input-coordinates."""
        self.network.eval()
        image_size = self.dataset.image_size.to(self.device)

        flow = []
        for i in range(0, len(self.dataset), self.chunk_size):
            coords_i = self.dataset[i : i + self.chunk_size]
            flow_i = self.ODE(coords_i)[0] * image_size / 2
            flow.append(flow_i)

        flow = torch.cat(flow, dim=0)
        flow = self.dataset.masked_gather(flow, is_flow=True)
        flow = flow.cpu().numpy().reshape(*self.shape, 3)
        return flow

    @torch.no_grad()
    def inf_warped(self):
        """Return the warped image-values for the given input-coordinates."""
        self.network.eval()

        warp_vals = []
        for i in range(0, len(self.dataset), self.chunk_size):
            coords_i = self.dataset[i : i + self.chunk_size]
            tar_coords_i = self.ODE(coords_i)[1]
            warp_val_i = self.dataset.samp_mov(tar_coords_i)
            warp_vals.append(warp_val_i)

        warp_val = torch.cat(warp_vals, dim=0)
        warp_val = self.dataset.masked_gather(warp_val, is_flow=False)
        warp_val = warp_val.cpu().detach().numpy().reshape(*self.shape)
        return warp_val

    @torch.no_grad()
    def inf_segmentation(self):
        """Return the segmentation map for the given input-coordinates."""
        self.network.eval()

        warp_vals = []
        for i in range(0, len(self.dataset), self.chunk_size):
            coords_i = self.dataset[i : i + self.chunk_size]
            tar_coords_i = self.ODE(coords_i)[1]
            warp_val_i = self.dataset.samp_mov_oars(tar_coords_i)
            warp_vals.append(warp_val_i)

        warp_val = torch.cat(warp_vals, dim=0)
        warp_val = self.dataset.masked_gather(warp_val, is_flow=False)
        warp_val = warp_val.cpu().detach().numpy().reshape(*self.shape)
        return warp_val

    def arr2nii(self, array, save_path):
        """Save the image-values to a NIfTI file."""
        return self.dataset.arr2nii(array, save_path=save_path)
