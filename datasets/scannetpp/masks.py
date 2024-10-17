import einops
import torch
import torch.nn.functional as F
from datasets.scannetpp.geometry import unproject_depth, world_space_to_camera_space, camera_space_to_pixel_space

@torch.no_grad()
def calculate_in_frustum_mask_single(depth_1, intrinsics_1, c2w_1, depth_2, intrinsics_2, c2w_2):
    """
    A function that takes in the depth, intrinsics, and c2w matrices for one pair
    of views (no batches, no multiple views), and then computes a mask where pixels
    in the first view correspond to valid pixels in the second view.

    Args:
        depth_1: (h, w)
        intrinsics_1: (3, 3)
        c2w_1: (4, 4)
        depth_2: (h, w)
        intrinsics_2: (3, 3)
        c2w_2: (4, 4)

    Returns:
        torch.Tensor: Mask with shape (h, w), where 1 indicates valid correspondence.
    """
    h, w = depth_1.shape

    # Step 1: Unproject depth map 1 to get world-space points
    points_3d_1 = unproject_depth(depth_1.unsqueeze(0).unsqueeze(0), intrinsics_1.unsqueeze(0).unsqueeze(0), c2w_1.unsqueeze(0).unsqueeze(0))  # (1, 1, h, w, 3)
    points_3d_1 = points_3d_1.squeeze(0).squeeze(0)  # (h, w, 3)

    # Step 2: Project world points to camera space of the second view
    camera_points_2 = world_space_to_camera_space(points_3d_1.unsqueeze(0), c2w_2.unsqueeze(0))  # (1, h, w, 3)
    camera_points_2 = camera_points_2.squeeze(0)  # (h, w, 3)

    # Step 3: Project camera-space points to pixel-space in the second view
    pixel_points_2 = camera_space_to_pixel_space(camera_points_2.unsqueeze(0), intrinsics_2.unsqueeze(0))  # (1, h, w, 2)
    pixel_points_2 = pixel_points_2.squeeze(0)  # (h, w, 2)

    # Step 4: Check if the projected points fall within the image boundaries
    in_frustum_mask = (
        (pixel_points_2[..., 0] > 0) &
        (pixel_points_2[..., 0] < w) &
        (pixel_points_2[..., 1] > 0) &
        (pixel_points_2[..., 1] < h)
    )

    # Step 5: Check if the depth is valid (non-zero) in depth_1
    non_zero_depth_1 = depth_1 > 1e-6

    # Step 6: Normalize pixel points to [-1, 1] for grid sampling in depth_2
    pixel_points_2[..., 0] = (pixel_points_2[..., 0] / w) * 2 - 1
    pixel_points_2[..., 1] = (pixel_points_2[..., 1] / h) * 2 - 1

    # Step 7: Sample the depth map of the second view
    depth_2 = depth_2.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
    sampled_depth_2 = F.grid_sample(depth_2, pixel_points_2.unsqueeze(0).unsqueeze(0), align_corners=False)[0, 0]

    # Step 8: Check if the depth matches between the two views
    depth_matches = torch.isclose(camera_points_2[..., 2], sampled_depth_2, atol=1e-1)

    # Step 9: Combine all conditions to form the final mask
    mask = in_frustum_mask & non_zero_depth_1 & depth_matches

    return mask