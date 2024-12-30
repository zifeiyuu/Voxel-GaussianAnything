import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from einops import rearrange


def estimate_depth_scale_kitti(depth, depth_gt):
    """
    depth: [1, 1, H, W]
    depth_gt: [N, 2]
    """
    eps = 1e-7
    depth = rearrange(depth, "1 1 h w -> (h w)")
    depth_gt = rearrange(depth_gt, "1 1 h w -> (h w)")
    valid_depth = depth_gt != 0
    depth = depth[valid_depth]
    depth_gt = depth_gt[valid_depth]

    scale = (depth.log() - depth_gt.log()).mean().exp()
    return scale


def estimate_depth_scale(depth, sparse_depth):
    """
    depth: [1, 1, H, W]
    sparse_depth: [N, 3]
    """
    eps = 1e-7
    device = depth.device
    sparse_depth = sparse_depth.to(device)
    if sparse_depth.shape[0] < 10:
        return torch.tensor(1.0, device=device, dtype=torch.float32)
    xy = sparse_depth[:, :2]
    z = sparse_depth[:, 2]
    xy = xy.unsqueeze(0).unsqueeze(0)
    depth_pred = F.grid_sample(depth, xy.to(depth.device), align_corners=False)
    depth_pred = depth_pred.squeeze()
    # z = torch.max(z, torch.tensor(eps, dtype=z.dtype, device=z.device))
    good_depth = torch.logical_and(z > eps, depth_pred > eps)
    z = z[good_depth]
    depth_pred = depth_pred[good_depth]

    if z.shape[0] < 10:
        return torch.tensor(1.0, device=device, dtype=torch.float32)

    scale = (depth_pred.log() - z.log()).mean().exp()
    return scale

def estimate_depth_scale_by_depthmap(depth, tgt_depth, max_depth=20):
    """
    depth: [1, 1, H, W]
    tgt_depth: [H, W]
    """

    eps = 1e-3
    device = depth.device
    tgt_depth = tgt_depth.to(device)
    
    valid_mask = torch.logical_and((tgt_depth > eps), (tgt_depth < max_depth)).bool()
    scale = (depth.squeeze()[valid_mask].log() - tgt_depth[valid_mask].log()).mean().exp()
    # scale = depth.squeeze()[valid_mask].median() / tgt_depth[valid_mask].median()
    return scale

def estimate_depth_scale_bias(depth, tgt_depth, max_depth=20):
    """
    depth: [1, 1, H, W]
    tgt_depth: [H, W]
    """

    eps = 1e-3
    device = depth.device
    tgt_depth = tgt_depth.to(device)
    
    valid_mask = torch.logical_and((tgt_depth > eps), (tgt_depth < max_depth)).bool()
    source_flat = depth[valid_mask].view(-1)  
    target_flat = tgt_depth[valid_mask].view(-1) 

    A = torch.cat((source_flat.unsqueeze(1), torch.ones(source_flat.size(0), 1, device=source_flat.device)), dim=1)

    A_t = torch.transpose(A, 0, 1)
    AtA = torch.matmul(A_t, A)
    A_t_D2 = torch.matmul(A_t, target_flat)
    X = torch.pinverse(AtA).matmul(A_t_D2)
    s, b = X[0], X[1]
    return s, b



def estimate_depth_scale_ransac(depth, sparse_depth, num_iterations=1000, sample_size=5, threshold=0.1):
    best_scale = None
    best_inliers = 0

    device = depth.device
    sparse_depth = sparse_depth.to(device)

    xy = sparse_depth[:, :2]
    z = sparse_depth[:, 2]
    xy = xy.unsqueeze(0).unsqueeze(0)
    depth_pred = F.grid_sample(depth, xy.to(depth.device), align_corners=False)
    depth_pred = depth_pred.squeeze()
    eps=1e-7
    # z = torch.max(z, torch.tensor(eps, dtype=z.dtype, device=z.device))
    good_depth = torch.logical_and(z > eps, depth_pred > eps)

    if good_depth.shape[0] < 10:
        return torch.tensor(1.0, device=device, dtype=torch.float32)
    z = z[good_depth]
    depth_pred = depth_pred[good_depth]

    if z.shape[0] < 10:
        return torch.tensor(1.0, device=device, dtype=torch.float32)

    if z.shape[0] <= sample_size:
        return (depth_pred.log() - z.log()).mean().exp()

    for _ in range(num_iterations):
        # Step 1: Random Sample Selection
        sample_indices = random.sample(range(z.shape[0]), sample_size)
        # Step 2: Estimation of Scale
        scale = (depth_pred[sample_indices].log() - z[sample_indices].log()).mean().exp()

        # Step 3: Inlier Detection
        inliers = torch.abs(depth_pred.log() - (z*scale).log()) < threshold

        # Step 5: Consensus Set Selection
        num_inliers = torch.sum(inliers)
        if num_inliers > best_inliers:
            best_scale = scale
            best_inliers = num_inliers
    if best_scale is None:
        return (depth_pred.log() - z.log()).mean().exp()
    return best_scale

CMAP_DEFAULT = 'plasma'
def gray2rgb(im, cmap=CMAP_DEFAULT):
    cmap = plt.get_cmap(cmap)
    result_img = cmap(im.astype(np.float32))
    if result_img.shape[2] > 3:
        result_img = np.delete(result_img, 3, 2)
    return result_img


def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None, cmap=CMAP_DEFAULT,
                                return_normalizer=False):
    """Converts a depth map to an RGB image."""
    # Convert to disparity.

    depth = np.squeeze(depth)

    depth_f = depth.flatten()
    depth_f = depth_f[depth_f != 0]
    disp_f = 1.0 / (depth_f + 1e-6)
    percentile = np.percentile(disp_f, pc)

    disp = 1.0 / (depth + 1e-6)
    if normalizer is not None:
        disp /= normalizer
    else:
        disp /= (percentile + 1e-6)
    disp = np.clip(disp, 0, 1)
    disp = gray2rgb(disp, cmap=cmap)
    keep_h = int(disp.shape[0] * (1 - crop_percent))
    disp = disp[:keep_h]
    if return_normalizer:
        return disp, percentile + 1e-6
    return disp

def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    # Mask for valid coordinates
    valid_mask = (depthmap > 0.0)
    return X_cam, valid_mask

def depthmap_to_camera_coordinates_torch(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Convert a depth map to 3D camera coordinates.

    Args:
        - depthmap (HxW tensor): Depth map.
        - camera_intrinsics (3x3 tensor): Camera intrinsic matrix.
        - pseudo_focal (HxW tensor or None): Optional tensor of focal lengths, used if not available in intrinsics.

    Returns:
        - X_cam (HxWx3 tensor): Point map of absolute coordinates.
        - valid_mask (HxW tensor): Mask specifying valid pixels.
    """

    
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0

    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = pseudo_focal
        fv = pseudo_focal

    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    # Create grid of pixel coordinates
    u, v = torch.meshgrid(torch.arange(W, dtype=depthmap.dtype, device=depthmap.device), torch.arange(H, dtype=depthmap.dtype, device=depthmap.device), indexing='xy')
    
    z_cam = depthmap * camera_intrinsics[2, 2]
    x_cam = (u - cu) * depthmap / fu
    y_cam = (v - cv) * depthmap / fv
    
    # Stack to get the 3D coordinates
    X_cam = torch.stack((x_cam, y_cam, z_cam), dim=-1)
    
    # Mask for valid coordinates
    valid_mask = (depthmap > 0.0)
    
    return X_cam, valid_mask


def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, **kw):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels."""
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)

    X_world = X_cam # default
    if camera_pose is not None:
        # R_cam2world = np.float32(camera_params["R_cam2world"])
        # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates (invalid depth values)
        X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]

    return X_world, valid_mask

def depthmap_to_absolute_camera_coordinates_torch(depthmap, camera_intrinsics, camera_pose, **kw):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels."""
    X_cam, valid_mask = depthmap_to_camera_coordinates_torch(depthmap, camera_intrinsics)

    X_world = X_cam # default
    if camera_pose is not None:
        # R_cam2world = np.float32(camera_params["R_cam2world"])
        # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates (invalid depth values)
        X_world = torch.matmul(X_cam, R_cam2world.T) + t_cam2world[None, None, :]

    return X_world, valid_mask