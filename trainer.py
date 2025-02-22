import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import logging
from einops import rearrange
from pathlib import Path

from models.model import GaussianPredictor
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from misc.depth import normalize_depth_for_display, depthmap_to_camera_coordinates_torch
from misc.util import sec_to_hm_str
from misc.visualise_3d import storePly
# from chamferdist import ChamferDistance

from models.encoder.layers import SSIM
from evaluate import evaluate, get_model_instance
from src.splatt3r_src.loss_mask import calculate_in_frustum_mask_single
from torch.utils.tensorboard import SummaryWriter
from IPython import embed

class Trainer(nn.Module):
    def __init__(self, cfg, pretrain):
        super().__init__()

        self.cfg = cfg
        self.step = 0
        self.pretrain = pretrain
        if cfg.model.name == "unidepth":
            self.model = GaussianPredictor(cfg)
            self.model._is_train = True
        elif cfg.model.name == "gat_voxsplat":
            from models.gat_model import GATModel
            self.model = GATModel(cfg)
        else:
            raise ValueError(f"Model {cfg.model.name} not supported")

        if cfg.loss.ssim.weight > 0:
            self.ssim = SSIM()
        if cfg.loss.lpips.weight > 0:
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
        self.logger = None
        base_dir = Path(__file__).resolve().parent
        self.output_path = base_dir / cfg.output_path
        self.output_path.resolve()

        self.warmup = cfg.train.pretrain_warmup > 0

    def set_warmup(self, warmup):
        self.warmup = warmup

    def set_logger(self, cfg, name='torch'):
        if name == 'torch':
            if self.pretrain:
                logdir = os.path.join(self.output_path, 'tensorboard', cfg.config['exp_name'], "preatrian")
            else:
                logdir = os.path.join(self.output_path, 'tensorboard', cfg.config['exp_name'], "gsm")
                
            logging.info(f"Tensorboard dir: {logdir}")
            self.logger = SummaryWriter(log_dir=logdir)
        elif name == 'fabric':
            self.logger = SummaryWriter(log_dir=str(self.output_path / 'tensorboard' / str(time.time()))) #only used for fabric train
        else:
            self.logger = None

    def forward(self, inputs):
        outputs = self.model.forward(inputs)
        if self.cfg.task_name == "flash3d":
            losses = self.compute_flash3d_losses(inputs, outputs)
        else:
            losses = self.compute_losses(inputs, outputs)
        self.get_grad_norm(outputs)
        return losses, outputs
    
    def compute_reconstruction_loss(self, pred, target, losses):
        """Computes reprojection loss between a batch of predicted and target images
        """
        cfg = self.cfg
        if torch.isnan(pred).any() or torch.isnan(target).any():
            print("Detected NaN values in the input tensors.")
            # valid_mask = ~torch.isnan(pred) & ~torch.isnan(target)
            # pred[~valid_mask] = 0  # Replace invalid values with 0
            # target[~valid_mask] = 0
            pred = torch.nan_to_num(pred, nan=0.0)
            target = torch.nan_to_num(target, nan=0.0)
        rec_loss = 0.0
        # pixel level loss
        if cfg.loss.mse.weight > 0:
            if cfg.loss.mse.type == "l1":
                mse_loss = (pred-target).abs().mean()
            elif cfg.loss.mse.type == "l2":
                mse_loss = ((pred-target)**2).mean()
            losses["loss/mse"] = mse_loss
            rec_loss += cfg.loss.mse.weight * mse_loss
        # patch level loss
        if cfg.loss.ssim.weight > 0:
            ssim_loss = self.ssim(pred, target).mean()
            losses["loss/ssim"] = ssim_loss
            rec_loss += cfg.loss.ssim.weight * ssim_loss
        # feature level loss
        if cfg.loss.lpips.weight > 0:
            if self.step > cfg.loss.lpips.apply_after_step:
                lpips_loss = self.lpips.to(pred.device)((pred * 2 - 1).clamp(-1,1), 
                                (target * 2 - 1).clamp(-1,1))
                losses["loss/lpips"] = lpips_loss
                rec_loss += cfg.loss.lpips.weight * lpips_loss
        
        return rec_loss
    
    def compute_depth_loss(self, inputs, outputs):
        cfg = self.cfg
        frames = [0] + cfg.model.gauss_novel_frames
        
        def scale_invariant_depth_loss(pred, target):
            pred_log = torch.log(pred)
            target_log = torch.log(target)
            
            diff = pred_log - target_log
            n = torch.numel(diff)
            
            loss = (diff ** 2).mean() - (diff.sum() ** 2) / (2 * n ** 2)
            return loss
        loss = 0.0
        for frame in frames:

            target, render = torch.stack(inputs[('depth_sparse', frame)]), outputs[('depth_gauss', frame, 0)].squeeze()
            target = target.to(render.device)
            mask = torch.logical_and((target > 1e-3), (target < 20)).bool()
            
            loss += scale_invariant_depth_loss(render[mask], target[mask])
        return loss
    
    def compute_bce_loss(self, outputs):
        bce_loss, rec_iou = 0.0, 0.0
        if outputs["error"]:
            return bce_loss, rec_iou
        binary_logits, binary_voxels = outputs['binary_logits'], outputs['binary_voxel']
        # breakpoint()
        bce_loss = F.binary_cross_entropy_with_logits(binary_logits, binary_voxels, reduction="mean")
        rec_iou = ((binary_logits.sigmoid() >= 0.5) & (binary_voxels >= 0.5)).sum() / (
            (binary_logits.sigmoid() >= 0.5) | (binary_voxels >= 0.5)
        ).sum()

        return bce_loss, rec_iou
    
    def compute_feature_loss(self, outputs):
        pred_feat, gt_feat = outputs["pred_feat"], outputs["gt_feat"]  # (N, 64), (N, 64)
        feature_loss = 0
        for b in range(len(pred_feat)):
            assert pred_feat[b].shape == gt_feat[b].shape, "Shapes of predicted and gt features must match"
            feature_loss += F.mse_loss(pred_feat[b], gt_feat[b], reduction="mean")
        feature_loss /= len(pred_feat)
        return feature_loss
    
    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        cfg = self.cfg
        B = cfg.data_loader.batch_size
        losses = {}
        total_loss = 0.0

        if self.cfg.loss.bce.weight > 0 and cfg.model.binary_predictor:
            bce_loss, rec_iou = self.compute_bce_loss(outputs)
            losses["loss/bce_loss"] = bce_loss
            losses["loss/rec_iou"] = rec_iou
            total_loss += self.cfg.loss.bce.weight * bce_loss

        if self.cfg.model.gaussian_rendering:
            # regularize too big gaussians
            if (big_g_lmbd := cfg.loss.gauss_max_scale.weight) > 0:
                big_gauss_reg_loss = 0
                for b in range(B):
                    scaling = outputs["gauss_scaling"][b]
                    big_gaussians = torch.where(scaling > cfg.loss.gauss_max_scale.thresh)
                    if len(big_gaussians[0]) > 0:
                        big_gauss_reg_loss += torch.mean(scaling[big_gaussians])

                losses["loss/big_gauss_reg_loss"] = big_gauss_reg_loss / B
                total_loss += big_g_lmbd * big_gauss_reg_loss / B

            # regularize too small gaussians
            if (small_g_lmbd := cfg.loss.gauss_min_scale.weight) > 0:
                small_gauss_reg_loss = 0
                for b in range(B):
                    scaling = outputs["gauss_scaling"][b]
                    small_gaussians = torch.where(scaling < cfg.loss.gauss_min_scale.single_thresh)
                    if len(small_gaussians[0]) > 0:
                        small_gauss_reg_loss += torch.mean(scaling[small_gaussians])

                losses["loss/small_gauss_reg_loss"] = small_gauss_reg_loss / B
                total_loss += small_g_lmbd * small_gauss_reg_loss / B

            # reconstruction loss
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                frame_ids = self.model.module.all_frame_ids(inputs)[: self.cfg.train.train_view_num]
            else:
                frame_ids = self.model.all_frame_ids(inputs)[: self.cfg.train.train_view_num]

            rec_loss = 0
            for frame_id in frame_ids:
                # compute gaussian reconstruction loss
                target = inputs[("color_aug", frame_id, 0)]
                target = target[:,:,cfg.dataset.pad_border_aug:target.shape[2]-cfg.dataset.pad_border_aug,
                                cfg.dataset.pad_border_aug:target.shape[3]-cfg.dataset.pad_border_aug,]
                pred = outputs[("color_gauss", frame_id, 0)]

                rec_loss += self.compute_reconstruction_loss(pred, target, losses)
            rec_loss /= len(frame_ids)
            losses["loss/rec"] = rec_loss
            total_loss += rec_loss

        if outputs["error"]:
            print("** Model error, return zero loss **")
            total_loss -= total_loss
        
        losses["loss/total"] = total_loss
        return losses
    
    def compute_flash3d_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        cfg = self.cfg
        losses = {}
        total_loss = 0.0

        if cfg.model.gaussian_rendering:
            # regularize too big or too small gaussians
            if (big_g_lmbd := cfg.loss.gauss_scale.weight) > 0:
                scaling = outputs["gauss_scaling"]
                big_gaussians = torch.where(scaling > cfg.loss.gauss_scale.thresh)
                if len(big_gaussians[0]) > 0:
                    big_gauss_reg_loss = torch.mean(scaling[big_gaussians])
                else:
                    big_gauss_reg_loss = 0
                losses["loss/big_gauss_reg_loss"] = big_gauss_reg_loss
                total_loss += big_g_lmbd * big_gauss_reg_loss
            
            # regularize too big offset
            if cfg.model.predict_offset and (offs_lmbd := cfg.loss.gauss_offset.weight) > 0:
                offset = outputs["gauss_offset"]
                big_offset = torch.where(offset**2 > cfg.loss.gauss_offset.thresh**2)
                if len(big_offset[0]) > 0:
                    big_offset_reg_loss = torch.mean(offset[big_offset]**2)
                else:
                    big_offset_reg_loss = 0.0
                losses["loss/gauss_offset_reg"] = big_offset_reg_loss
                total_loss += offs_lmbd * big_offset_reg_loss

            # reconstruction loss
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                frame_ids = self.model.module.all_frame_ids(inputs)[:3]
            else:
                frame_ids = self.model.all_frame_ids(inputs)[:3]
                
            rec_loss = 0
            for frame_id in frame_ids:
                # compute gaussian reconstruction loss
                target = inputs[("color_aug", frame_id, 0)]
                target = target[:,:,cfg.dataset.pad_border_aug:target.shape[2]-cfg.dataset.pad_border_aug,
                                cfg.dataset.pad_border_aug:target.shape[3]-cfg.dataset.pad_border_aug,]
                pred = outputs[("color_gauss", frame_id, 0)]
                rec_loss += self.compute_reconstruction_loss(pred, target, losses)
            rec_loss /= len(frame_ids)
            losses["loss/rec"] = rec_loss
            total_loss += rec_loss

        if outputs["error"]:
            print("** Model error, return zero loss **")
            total_loss -= total_loss

        losses["loss/total"] = total_loss
        return losses

    def get_grad_norm(self, outputs):
        # Compute gradient norm
        grad_norm = 0.0

        # Check if model is wrapped in DistributedDataParallel
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            parameters = self.model.module.parameters()
        else:
            parameters = self.model.parameters()

        # Compute the gradient norm across all parameters
        for param in parameters:
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        outputs["grad_norm"] = grad_norm
        
    
    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.cfg.optimiser.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
    
    def log_scalars(self, mode, outputs, losses, lr):
        """log the scalars"""
        cfg = self.cfg
        B = cfg.data_loader.batch_size
        logger = self.logger
        if logger is None:
            return
        logger.add_scalar(f"{mode}/learning_rate", lr, self.step)
        logger.add_scalar(f"{mode}/grad_norm", outputs['grad_norm'], self.step)

        for l, v in losses.items():
            logger.add_scalar(f"{mode}/{l}", v, self.step)

        if cfg.model.gaussian_rendering:
            logger.add_scalar(f"{mode}/gauss/scale/mean", torch.mean(outputs["gauss_scaling"][0]).item(), self.step)

        if not cfg.train.pretrain:
            if cfg.model.gaussian_rendering:
                if self.cfg.model.predict_offset and self.cfg.model.direct_gaussian:
                    offset_mag = torch.linalg.vector_norm(outputs["gauss_offset"], dim=1)
                    mean_offset = offset_mag.mean()
                    logger.add_scalar(f"{mode}/gauss/offset/mean", mean_offset.item(), self.step)

            if cfg.dataset.scale_pose_by_depth:
                depth_scale = outputs[("depth_scale", 0)]
                logger.add_scalar(f"{mode}/depth_scale", depth_scale.mean().item(), self.step)


    def log(self, mode, inputs, outputs):
        """Write images to Neptune
        """
        cfg = self.cfg
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            frame_ids = self.model.module.all_frame_ids(inputs)[:4]
        else:
            frame_ids = self.model.all_frame_ids(inputs)[:4]
        scales = cfg.model.scales
        logger = self.logger
        if logger is None:
            return

        for j in range(min(4, cfg.data_loader.batch_size)): # write a maxmimum of 4 images
            for s in scales:
                assert cfg.model.gaussian_rendering
                for frame_id in frame_ids:
                    logger.log_image(
                        f"{mode}/color_gauss/{j}/gt_aug/{frame_id}",
                        inputs[("color_aug", frame_id, 0)][j].data.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy(),
                        self.step
                    )
                
                for frame_id in frame_ids:
                    logger.log_image(
                        f"{mode}/color_gauss/{j}/gt/{frame_id}",
                        inputs[("color", frame_id, 0)][j].data.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy(),
                        self.step
                    )

                for frame_id in frame_ids:
                    logger.log_image(
                        f"{mode}/color_gauss/{j}/pred/{frame_id}",
                        outputs[("color_gauss", frame_id, 0)][j].data.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy(),
                        self.step
                    )

                for i in range(self.cfg.model.gaussians_per_pixel):
                    logger.log_image(
                        f"{mode}/gauss_opacity_gaussian_{i}/{j}",
                        outputs["gauss_opacity"][j * self.cfg.model.gaussians_per_pixel + i].data.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy(),
                        self.step
                    )

                depth = rearrange(outputs[("depth", 0)], "(b n) ... -> b n ...", n=self.cfg.model.gaussians_per_pixel)
                depth_sliced = depth[j][0].detach().cpu().numpy()
                depth_img, normalizer = normalize_depth_for_display(depth_sliced, return_normalizer=True)
                depth_img = np.clip(depth_img, 0, 1)

                logger.log_image(f"{mode}/depth_{s}/{j}", depth_img, self.step)

                for layer in range(1, self.cfg.model.gaussians_per_pixel):
                    depth_sliced = depth[j][layer].detach().cpu().numpy()
                    depth_img =  normalize_depth_for_display(depth_sliced, normalizer=normalizer)
                    depth_img = np.clip(depth_img, 0, 1)
                    logger.log_image(
                        f"{mode}/depth_{layer}_gaussian_{s}/{j}",
                        depth_img,
                        self.step
                    )

    def validate(self, model, evaluator, val_loader, device, output_path = None):
        """
        model may not be the same as trainer, in case of wrapping it in EMA
        sets model to eval mode by evaluate()
        """
        if not output_path:
            output_path = self.output_path

        score_dict_by_name = evaluate(model, self.cfg, evaluator, val_loader, device, self.cfg.save_vis, output_path)
        split = "val"
        # out = {}
        # for metric in evaluator.metric_names():
        #     out[f"{split}/{metric}/avg"] = \
        #         torch.tensor([scores[metric] for f_id, scores in score_dict_by_name.items() if f_id != 0]).mean().item()
        #     for f_id, scores in score_dict_by_name.items():
        #         out[f"{split}/{metric}/{f_id}"] = scores[metric]
        # if self.logger is not None:
        #    self.logger.log(out, self.step)
        model_model = get_model_instance(model)
        model_model.set_train()
