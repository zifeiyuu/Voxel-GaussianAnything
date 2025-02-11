import os
from tqdm import tqdm
import numpy as np
import cv2
import sys
import pickle
import gzip
from pathlib import Path
import torch

from IPython import embed

def convert_angle_axis_to_matrix3(angle_axis):
    """Return a Matrix3 for the angle axis.
    Arguments:
        angle_axis {Point3} -- a rotation in angle axis form.
    """
    matrix, jacobian = cv2.Rodrigues(angle_axis)
    return matrix

def st2_camera_intrinsics(filename):
    w, h, fx, fy, hw, hh = np.loadtxt(filename)
    return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])

def TrajStringToMatrix(traj_str):
    """ convert traj_str into translation and rotation matrices
    Args:
        traj_str: A space-delimited file where each line represents a camera position at a particular timestamp.
        The file has seven columns:
        * Column 1: timestamp
        * Columns 2-4: rotation (axis-angle representation in radians)
        * Columns 5-7: translation (usually in meters)

    Returns:
        ts: translation matrix
        Rt: rotation matrix
    """
    # line=[float(x) for x in traj_str.split()]
    # ts = line[0];
    # R = cv2.Rodrigues(np.array(line[1:4]))[0];
    # t = np.array(line[4:7]);
    # Rt = np.concatenate((np.concatenate((R, t[:,np.newaxis]), axis=1), [[0.0,0.0,0.0,1.0]]), axis=0)
    tokens = traj_str.split()
    assert len(tokens) == 7
    ts = tokens[0]
    # Rotation in angle axis
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))
    # Translation
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)
    return (ts, Rt)


def process_scenes(root, output_dir):
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    scenes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    print("total scenes: ", len(scenes))
    bad_scene = []
    all_data = {}  # Dictionary to hold all data

    for scene in tqdm(scenes, position=0, leave=True): 

        scene_output_file = Path(root) / scene / f"{scene}_data.npz"
        if scene_output_file.exists():
            print(f"Loading existing {scene}.npz...")
            data = np.load(scene_output_file)

            all_data[scene] = {
                "timestamps": data["timestamps"].tolist(),
                "poses": data["poses"].tolist(),
                "intrinsics": data["intrinsics"].tolist()
            }
            continue 
      
        if bad_scene and scene in bad_scene:
            print(f"skip bad scene: {scene}")
            continue

        data_dir = os.path.join(root, scene, f"{scene}_frames")

        # traj
        traj_file = os.path.join(data_dir, 'lowres_wide.traj')
        with open(traj_file) as f:
            trajs = f.readlines()
        timestamps = []
        poses = []
        for line in trajs:
            traj_timestamp = line.split(" ")[0]
            timestamps.append(f"{round(float(traj_timestamp), 3):.3f}")
            poses.append(TrajStringToMatrix(line)[1].tolist())

        # intrinsic
        intrinsics = []
        for frame_id in timestamps:
            intrinsic_fn = os.path.join(data_dir, "lowres_wide_intrinsics", f"{scene}_{frame_id}.pincam")
            if not os.path.exists(intrinsic_fn):
                intrinsic_fn = os.path.join(data_dir, "lowres_wide_intrinsics",
                                            f"{scene}_{float(frame_id) - 0.001:.3f}.pincam")
            if not os.path.exists(intrinsic_fn):
                intrinsic_fn = os.path.join(data_dir, "lowres_wide_intrinsics",
                                            f"{scene}_{float(frame_id) + 0.001:.3f}.pincam")
            if not os.path.exists(intrinsic_fn):
                print("intrinsic file not found, frame_id: ", frame_id, intrinsic_fn)
            intrinsics.append(st2_camera_intrinsics(intrinsic_fn))
        
        np.savez_compressed(scene_output_file, 
                            timestamps=np.array(timestamps), 
                            poses=np.array(poses), 
                            intrinsics=np.array(intrinsics))
        print(f"Saved {scene} to {scene_output_file}")

        all_data[scene] = {
            "timestamps": timestamps,
            "poses": poses,
            "intrinsics": intrinsics
        }

    # # Save all data to a compressed pickle file
    with gzip.open(output_dir / 'all_data.pickle.gz', 'wb') as f:
        pickle.dump(all_data, f)


def extract_test_split(root, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    open(output_file, "w").close()

    root = Path(root)
    file_path = root / "all_data.pickle.gz"
    with gzip.open(file_path, "rb") as f:
        pose_data = pickle.load(f)
        print("data loaded!")

    neighbor_frame_num = 3
    seq_keys = list(pose_data.keys())
    for key in tqdm(seq_keys, position=0, leave=True):
        seq_len = len(pose_data[key]["poses"])
        if seq_len < 2 * neighbor_frame_num + 1:
            continue  # Skip if there are not enough frames
        frame_indices = seq_len // 2
        target_frame_idxs = torch.arange(-4, 5)
        target_frame_idxs = target_frame_idxs[target_frame_idxs != 0]
        sorted_frame_idxs = sorted(target_frame_idxs.tolist(), key=lambda x: abs(x))
        sorted_frame_idxs = [i for i in sorted_frame_idxs if abs(i) > 1]
        src_and_tgt_frame_idxs = [frame_indices] + [
            max(min(i + frame_indices, seq_len - 1), 0) for i in sorted_frame_idxs
        ][1:4] 

        exist_check = False
        idx_len = len(src_and_tgt_frame_idxs)
        while not exist_check:
            for frame_idx in src_and_tgt_frame_idxs:
                img_name = f"{key}_{pose_data[key]['timestamps'][frame_idx]}.png"
                img_path = root / key / f"{key}_frames" / "lowres_wide" / img_name

                if not img_path.exists():
                    print(f"Change needed for {key}")
                    if idx_len + frame_idx + 1 > seq_len:
                        print(f"No valid image in dataset scene {key}")
                        src_and_tgt_frame_idxs = None
                        exist_check = True 
                        break

                    src_idx = frame_idx + 4
                    src_and_tgt_frame_idxs = [src_idx] + [
                        max(min(i + src_idx, seq_len - 1), 0) for i in sorted_frame_idxs
                    ][1:4] 
                    
                    break 
            else:
                exist_check = True

        if src_and_tgt_frame_idxs is not None:
            with open(output_file, "a") as f:  # Fix: Writing to file correctly
                f.write(f"{key} {src_and_tgt_frame_idxs[0]} {src_and_tgt_frame_idxs[1]} {src_and_tgt_frame_idxs[2]} {src_and_tgt_frame_idxs[3]}\n")

if __name__ == '__main__':
    root = '/mnt/datasets/arkitscenes/3dod/Training'
    # original_dir = '/mnt/datasets/arkitscenes/3dod/Validation'
    output_dir = '/mnt/datasets/arkitscenes/3dod/Training'
    test_split_dir = '/mnt/datasets/arkitscenes/3dod/Training/splits/test_split2.txt'
    # process_scenes(root, output_dir)

    extract_test_split(root, test_split_dir)
