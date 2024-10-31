from plyfile import PlyData, PlyElement
import torch
import numpy as np
from PIL import Image

def project_point_cloud_to_depth_map(points_3d, intrinsic_matrix, image_size, output_file):
    """
    Projects a 3D point cloud to a 2D depth map and saves it as a PNG.

    Args:
        points_3d (torch.Tensor): The 3D points in shape (B, 3, H*W).
        intrinsic_matrix (torch.Tensor): The camera intrinsic matrix in shape (3, 3).
        image_size (tuple): The (height, width) of the output depth map.
        output_file (str): Path to save the PNG depth map.
    """
    intrinsic_matrix = intrinsic_matrix.squeeze()
    height, width = image_size
    # Extract XYZ and reshape for easier processing
    x, y, z = points_3d[:, 0, :], points_3d[:, 1, :], points_3d[:, 2, :]  # Shape: (B, H*W)
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    # Project 3D points onto the 2D plane (camera intrinsic formula)
    u = (fx * x / z + cx).long()  # X pixel coordinates
    v = (fy * y / z + cy).long()  # Y pixel coordinates

    # Initialize a depth map and populate with z-values
    depth_map = torch.zeros(height, width, dtype=torch.float32, device=points_3d.device)
    
    # Apply depth values at projected pixel locations, using z-values (depths)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    depth_map[v[valid], u[valid]] = z[valid]


    print("depth map!!")
    print(depth_map)

    # Convert depth map to uint16 format and save as PNG
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_map_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 65535).cpu().numpy().astype(np.uint16)

    print("depth map norm!!&&&&&")
    print(depth_map_normalized)

    depth_image = Image.fromarray(depth_map_normalized)
    depth_image.save(output_file)
    print(f"Depth map saved as {output_file}")


def save_point_cloud_with_plyfile(points_3d, output_file, colors=None):
    points_3d_np = points_3d.cpu().numpy() if isinstance(points_3d, torch.Tensor) else points_3d

    # Check if colors are provided
    if colors is not None:
        # Ensure colors has the correct shape and convert if necessary
        colors = colors.cpu().numpy() if isinstance(colors, torch.Tensor) else colors
        if len(colors) != len(points_3d_np):
            raise ValueError("The number of colors must match the number of 3D points.")
        
        # Create vertices with XYZ and RGB values
        vertices = [(*p, *c) for p, c in zip(points_3d_np, colors)]
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    else:
        # Only XYZ values
        vertices = [(*p,) for p in points_3d_np]
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

    # Convert to structured numpy array
    vertex_data = np.array(vertices, dtype=dtype)

    # Save as .ply file
    ply_data = PlyData([PlyElement.describe(vertex_data, 'vertex')], text=True)
    ply_data.write(output_file)
    print(f"Point cloud saved to {output_file}")

# # Example usage
# points_3d = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
# colors = np.array([[255, 0, 0], [0, 255, 0]])  # Example colors for each point
# save_point_cloud_with_plyfile(points_3d, "point_cloud_colored.ply", colors)