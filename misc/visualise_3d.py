import torch
import os
import numpy as np
from pathlib import Path
from einops import rearrange, einsum
from matplotlib import pyplot as plt
import torch.nn.functional as F
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from torch import Tensor
from scipy.spatial.transform import Rotation as R

from misc.depth import normalize_depth_for_display
from models.encoder.layers import Project3DSimple

from IPython import embed
import time
def depth_to_img(d):
    d = d.detach().cpu().numpy()
    depth_img = normalize_depth_for_display(d)
    return (np.clip(depth_img, 0, 1) * 255).astype(np.uint8)


def vis_2d_offsets(model, inputs, outputs, out_dir, frame_id):
    input_f_id = 0
    scale = 0
    B, H, W = model.tensor_image_dims()

    xyz = outputs[("gauss_means", input_f_id, scale)]
    K = inputs[("K", scale)]

    p3d = Project3DSimple(1, H, W)
    bp3d = model.backproject_depth[str(scale)]

    pix_coords = p3d(xyz, K)
    pix_coords = rearrange(pix_coords, "1 h w c -> 1 c h w")
    id_coords = rearrange(bp3d.id_coords, "c h w -> 1 c h w")

    s = 8
    pix_coords = F.interpolate(
        pix_coords,
        (H // s, W // s),
        mode='nearest',
    )
    id_coords = F.interpolate(
        id_coords,
        (H // s, W // s),
        mode='nearest',
    )
    v = pix_coords - id_coords
    id_coords = rearrange(id_coords, "1 c h w -> (h w) c")
    v = rearrange(v, "1 c h w -> (h w) c")

    id_coords = id_coords.cpu().numpy()
    v = v.cpu().numpy()

    X = id_coords[:, 0]
    Y = id_coords[:, 1]
    U = v[:, 0]
    V = v[:, 1]
    # print(np.histogram(U)[0], np.histogram(U)[1])

    plt.quiver(X, Y, U, V, color='b', units='xy', scale=1) 
    plt.title('Gauss offset') 

    # x-lim and y-lim 
    plt.xlim(-50, W+50) 
    plt.ylim(-50, H+50) 

    plt.axis('equal')

    # print(mpl.rcParams["savefig.dpi"])

    plt.savefig(out_dir / f"{frame_id}.png", dpi=300.0)
    plt.cla()
    plt.clf()

    # import pdb
    # pdb.set_trace()


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


def export_ply(
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, "gaussian"],
    path: Path,
):
    f_dc = harmonics

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0)]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = (
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        f_dc.detach().cpu().contiguous().numpy(),
        opacities.detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations.detach().cpu().numpy(),
    )
    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def save_ply(outputs, path, gaussians_per_pixel=3, name=None, batch=0):
    if name == "unidepth":  
        means = rearrange(outputs["gauss_means"], "(b v) c n -> b (v n) c", v=gaussians_per_pixel)[0, :, :3]
        scales = rearrange(outputs["gauss_scaling"], "(b v) c h w -> b (v h w) c", v=gaussians_per_pixel)[0]
        rotations = rearrange(outputs["gauss_rotation"], "(b v) c h w -> b (v h w) c", v=gaussians_per_pixel)[0]
        opacities = rearrange(outputs["gauss_opacity"], "(b v) c h w -> b (v h w) c", v=gaussians_per_pixel)[0]
        harmonics = rearrange(outputs["gauss_features_dc"], "(b v) c h w -> b (v h w) c", v=gaussians_per_pixel)[0]
        f_rest = rearrange(outputs["gauss_features_rest"], "(b v) c h w -> b (v h w) c", v=gaussians_per_pixel)[0]
    else:
        means = outputs["gauss_means"][batch]
        scales = outputs["gauss_scaling"][batch]
        rotations = outputs["gauss_rotation"][batch]
        opacities = outputs["gauss_opacity"][batch]
        harmonics = outputs["gauss_features_dc"][batch].squeeze(1)

    export_ply(
        means,
        scales,
        rotations,
        harmonics,
        opacities,
        path / "predicted.ply"
    )

    if name != "unidepth" and outputs["gt_points"]:
        means2 = outputs["gt_points"][batch]
        scales2 = torch.ones_like(means2)
        rotations2 = torch.ones((means2.shape[0], 4), dtype=means2.dtype, device=means2.device)
        opacities2 = torch.ones((means2.shape[0], 1), dtype=means2.dtype, device=means2.device)
        harmonics2 = torch.ones((means2.shape[0], 3), dtype=means2.dtype, device=means2.device) 
        export_ply(
            means2,
            scales2,
            rotations2,
            harmonics2,
            opacities2,
            path / "GT.ply"
        )
        means3 = torch.cat([means, means2], dim=0)
        scales3 = torch.cat([scales, scales2], dim=0)
        rotations3 = torch.cat([rotations, rotations2], dim=0)
        opacities3 = torch.cat([opacities, opacities2], dim=0)
        harmonics3 = torch.cat([harmonics, harmonics2], dim=0)
        export_ply(
            means3,
            scales3,
            rotations3,
            harmonics3,
            opacities3,
            path / "combine.ply"
        )

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)