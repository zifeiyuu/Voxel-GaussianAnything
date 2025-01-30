import torch
import logging
import time
import torch.nn as nn

from pathlib import Path
from einops import rearrange

from models.encoder.layers import BackprojectDepth
from models.decoder.gauss_util import focal2fov, getProjectionMatrix, K_to_NDC_pp, render_predicted, debug_vis_pointcloud
from misc.util import add_source_frame_id
from misc.depth import estimate_depth_scale, estimate_depth_scale_ransac, estimate_depth_scale_by_depthmap
from IPython import embed
from matplotlib import pyplot as plt
import numpy as np
import scipy

from submodules.extra import project_point_cloud_to_depth_map, save_point_cloud_with_plyfile

def minmax(data):
    return (data - data.min()) / (data.max() - data.min())

def default_param_group(model):
    return [{'params': model.parameters()}]


def to_device(inputs, device):
    for key, ipt in inputs.items():
        if isinstance(ipt, torch.Tensor):
            inputs[key] = ipt.to(device)
    return inputs


class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.parameters_to_train = []
        self.predict_sh_offset = cfg.model.predict_sh_offset
    
    def get_parameter_groups(self):
        return self.parameters_to_train

    def target_frame_ids(self, inputs):
        return inputs["target_frame_ids"]
    
    def all_frame_ids(self, inputs):
        return add_source_frame_id(self.target_frame_ids(inputs))

    def set_train(self):
        """Convert all models to training mode
        """
        self.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        self.train(False)

    def is_train(self):
        return self.training

    @torch.no_grad()
    def process_gt_poses(self, inputs, outputs, pretrain=False):
        cfg = self.cfg
        keyframe = 0
        for f_i in self.target_frame_ids(inputs):
            if ("T_c2w", f_i) not in inputs:
                continue
            T_0 = inputs[("T_c2w", keyframe)]
            T_i = inputs[("T_c2w", f_i)]
            if ("T_w2c", keyframe) in inputs.keys():
                T_0_inv = inputs[("T_w2c", keyframe)]
            else:
                T_0_inv = torch.linalg.inv(T_0.float())
            if ("T_w2c", f_i) in inputs.keys():
                T_i_inv = inputs[("T_w2c", f_i)]
            else:
                T_i_inv = torch.linalg.inv(T_i.float())

            if T_i_inv.dtype == torch.float16 and T_0.dtype == torch.float16:
                outputs[("cam_T_cam", 0, f_i)] = (T_i_inv @ T_0).half()
            else:
                outputs[("cam_T_cam", 0, f_i)] = T_i_inv @ T_0
            if T_0_inv.dtype == torch.float16 and T_i.dtype == torch.float16:
                outputs[("cam_T_cam", f_i, 0)] = (T_0_inv @ T_i).half()
            else:
                outputs[("cam_T_cam", f_i, 0)] = T_0_inv @ T_i
        outputs[("cam_T_cam", 0, 0)] = torch.eye(4, device=T_0.device).unsqueeze(0).repeat(cfg.data_loader.batch_size, 1, 1)
        
        if cfg.dataset.scale_pose_by_depth:
            B = cfg.data_loader.batch_size
            if pretrain:
                depth_padded = torch.stack(inputs[('depth_sparse', 0)]).unsqueeze(1).detach()
            else:
                depth_padded = outputs[("depth_pred", 0)].detach()
            
            # only use the depth in the unpadded image for scale estimation
            depth = depth_padded[:, :, 
                                 self.cfg.dataset.pad_border_aug:depth_padded.shape[2]-self.cfg.dataset.pad_border_aug,
                                 self.cfg.dataset.pad_border_aug:depth_padded.shape[3]-self.cfg.dataset.pad_border_aug]
            sparse_depth = inputs[("depth_sparse", 0)]
            
            scales = []
            for k in range(B):
                depth_k = depth[[k * self.cfg.model.gaussians_per_pixel], ...]
                sparse_depth_k = sparse_depth[k]
                if ("scale_colmap", 0) in inputs.keys():
                    scale = inputs[("scale_colmap", 0)][k]
                else:
                    scale = estimate_depth_scale_by_depthmap(depth_k, sparse_depth_k)
                scales.append(scale)
            scale = torch.tensor(scales, device=depth.device).unsqueeze(dim=1)
            outputs[("depth_scale", 0)] = scale

            for f_i in self.target_frame_ids(inputs):
                T = outputs[("cam_T_cam", 0, f_i)]
                scale = scale.to(T.device)
                T[:, :3, 3] = T[:, :3, 3] * scale
                outputs[("cam_T_cam", 0, f_i)] = T
                T = outputs[("cam_T_cam", f_i, 0)]
                T[:, :3, 3] = T[:, :3, 3] * scale
                outputs[("cam_T_cam", f_i, 0)] = T

    
    def render_images(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        cfg = self.cfg
        B, _, H, W = inputs["color", 0, 0].shape

        for scale in [0]: #cfg.model.scales:
            pos_input_frame = outputs["gauss_means"][0].float()
            K = inputs[("K_tgt", 0)]
            device = pos_input_frame.device
            dtype = pos_input_frame.dtype

            frame_ids = self.all_frame_ids(inputs)[:4]

            for frame_id in frame_ids:
                if frame_id == 0:
                    T = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
                else:
                    if ('cam_T_cam', 0, frame_id) not in outputs:
                        continue
                    T = outputs[('cam_T_cam', 0, frame_id)]

                point_clouds = {
                    "xyz": outputs["gauss_means"],
                    "opacity": outputs["gauss_opacity"],
                    "scaling": outputs["gauss_scaling"],
                    "rotation": outputs["gauss_rotation"],
                    "features_dc": outputs["gauss_features_dc"]
                }

                if cfg.model.max_sh_degree > 0:
                    point_clouds["features_rest"] = rearrange(
                        outputs["gauss_features_rest"], 
                        "b s (sh rgb) n -> b (s n) sh rgb", 
                        rgb=3, s=self.cfg.model.gaussians_per_pixel
                    )

                rgbs = []
                depths = []

                for b in range(B):
                    # get camera projection matrix
                    if cfg.dataset.name in ["kitti", "nyuv2", "waymo"]:
                        K_tgt = inputs[("K_tgt", 0)]
                    else:
                        K_tgt = inputs[("K_tgt", frame_id)]
                    focals_pixels = torch.diag(K_tgt[b])[:2]

                    debug = False
                    if debug and frame_id == 0:
                        import numpy as np
                        image = inputs[("color", 0, 0)][b].detach().permute(1, 2, 0).cpu().numpy()
                        image = (image * 255).astype(np.uint8)
                        xyz = outputs["gauss_means"].squeeze().transpose(0, 1).detach().cpu()
                        debug_vis_pointcloud(xyz, K_tgt[b], H, W, inputs[("frame_id", 0)], image)
                        debug_vis_pointcloud([], K_tgt[b], H, W, inputs[("frame_id", 0)], image)

                    fovY = focal2fov(focals_pixels[1].item(), H)
                    fovX = focal2fov(focals_pixels[0].item(), W)
                    if cfg.dataset.name in ["co3d", "re10k", "mixed", "pixelsplat"]:
                        px_NDC, py_NDC = 0, 0
                    else:
                        px_NDC, py_NDC = K_to_NDC_pp(Kx=K_tgt[b][0, 2], Ky=K_tgt[b][1, 2], H=H, W=W)
                    proj_mtrx = getProjectionMatrix(cfg.dataset.znear, cfg.dataset.zfar, fovX, fovY, pX=px_NDC, pY=py_NDC).to(device)
                    world_view_transform = T[b].transpose(0, 1).float()
                    camera_center = (-world_view_transform[3, :3] @ world_view_transform[:3, :3].transpose(0, 1)).float()
                    proj_mtrx = proj_mtrx.transpose(0, 1).float() # [4, 4]
                    full_proj_transform = (world_view_transform@proj_mtrx).float()
                    # use random background for the better opacity learning
                    if cfg.model.randomise_bg_colour and self.is_train():
                        bg_color = torch.rand(3, dtype=dtype, device=device)
                    else:
                        bg_color = torch.tensor(cfg.model.bg_colour, dtype=dtype, device=device)

                    pc = {k: v[b].contiguous().float() for k, v in point_clouds.items()}

                    out = render_predicted(
                        cfg,
                        pc,
                        world_view_transform,
                        full_proj_transform,
                        proj_mtrx,
                        camera_center,
                        (fovX, fovY),
                        (H, W),
                        bg_color,
                        cfg.model.max_sh_degree
                    )
                    rgb = minmax(out["render"])
                    rgbs.append(rgb)
                    if "depth" in out:
                        depths.append(out["depth"])

                rbgs = torch.stack(rgbs, dim=0)
                outputs[("color_gauss", frame_id, scale)] = rbgs

                if "depth" in out:
                    depths = torch.stack(depths, dim=0)
                    outputs[("depth_gauss", frame_id, scale)] = depths
    

    def checkpoint_dir(self):
        return Path("checkpoints")

    def save_model(self, optimiser, step, ema=None, save_folder = None, pretraining=False):
        """save model weights to disk"""
        if save_folder == None:
            save_folder = self.checkpoint_dir()
        save_folder.mkdir(exist_ok=True, parents=True)

        phase = "pretrain" if pretraining else "finetune"
        save_path = save_folder / f"model_{phase}_{step:07}.pth"
        logging.info(f"saving checkpoint to {str(save_path)}")

        # Save EMA model state
        ema_model_state = ema.ema_model.state_dict() if ema is not None else None
        # Save current model state
        current_model_state = self.state_dict()

        save_dict = {
            "ema_model": ema_model_state,
            "current_model": current_model_state,
            "version": "1.0",
            "optimiser": optimiser.state_dict(),
            "step": step,
            "pretraining": pretraining
        }
        torch.save(save_dict, save_path)

        num_ckpts = self.cfg.run.num_keep_ckpts
        ckpts = sorted(list(save_folder.glob("model_*.pth")), reverse=True)
        if len(ckpts) > num_ckpts:
            for ckpt in ckpts[num_ckpts:]:
                ckpt.unlink()

    def load_model(self, weights_path, optimiser=None, device="cpu", ckpt_ids=0, load_optimizer=True, load_ema=False):
        """load model(s) from disk"""
        weights_path = Path(weights_path)

        if weights_path.is_dir():
            ckpts = sorted(list(weights_path.glob("model_*.pth")), reverse=True)
            weights_path = ckpts[ckpt_ids]
        logging.info(f"Loading weights from {weights_path}...")
        
        state_dict = torch.load(weights_path, map_location=torch.device(device))

        if load_ema:
            model_dict = state_dict["ema_model"]  
        else:
            model_dict = state_dict["current_model"]
        self.load_state_dict(model_dict, strict=False)
        
        # loading adam state
        if optimiser is not None and load_optimizer:
            optimiser.load_state_dict(state_dict["optimiser"])
            self.step = state_dict["step"]


    def rgb_to_sh0(self, rgb_tensor):
        """
        Convert a batch tensor of RGB values in range [0, 1] to SH0 coefficients.
        
        Args:
        - rgb_tensor (Tensor): A tensor of shape (B, 3, H, W) with RGB values normalized to [0, 1].
        
        Returns:
        - Tensor: SH0 coefficients for each channel at each pixel in each image of the batch.
        """
        factor = 1 / (2 * np.sqrt(np.pi))
        sh0_coeffs = rgb_tensor * factor
        return sh0_coeffs
    