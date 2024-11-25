import os
import argparse
import os.path as osp
import re
from tqdm import tqdm
import json
from scipy.spatial.transform import Rotation
import pyrender
import trimesh
import trimesh.exchange.ply
import numpy as np
import cv2
import PIL.Image as Image
import sys
import pickle
import gzip
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from IPython import embed

def is_continuous(pose1, pose2, threshold=1.0):
    # Extract translation vectors
    t1 = pose1[:3, 3]
    t2 = pose2[:3, 3]
    
    # Calculate translation distance
    translation_dist = np.linalg.norm(t1 - t2)
    
    # Extract rotation matrices
    rot1 = pose1[:3, :3]
    rot2 = pose2[:3, :3]
    
    def rotation_matrix_to_quaternion(rot_matrix):
        return R.from_matrix(rot_matrix).as_quat()
    # Convert rotations to quaternions
    quat1 = rotation_matrix_to_quaternion(rot1)
    quat2 = rotation_matrix_to_quaternion(rot2)
    
    # Calculate angular difference between quaternions
    rotation_dist = R.from_quat([quat1]).inv() * R.from_quat([quat2])
    angle = rotation_dist.magnitude()  # This gives the angle in radians
    
    # Combine the distances
    d = translation_dist + angle

    return True if d < threshold else False

def process_scenes(root, original_dir, output_dir):
    count = 0
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    scenes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and len(d) == 10]
    print(scenes)

    all_data = {'train': {}, 'test': {}}  # Dictionary to hold all data

    for scene in tqdm(scenes, position=0, leave=True): 
        print(scene)
        all_data['train'][scene] = []
        all_data['test'][scene] = []
        data_dir = os.path.join(root, scene)
        scene_metadata_path = osp.join(data_dir, 'scene_metadata.npz')

        with np.load(scene_metadata_path) as data:
            images = data['images']
            intrinsics = data['intrinsics'].astype(np.float32)
            poses = data['trajectories'].astype(np.float32)

        start_idx = 0

        # Iterate over poses and segment based on continuity
        for i in range(1, len(poses)):
            if not is_continuous(poses[i - 1], poses[i]):
                # Save current segment
                segment = {
                    'poses': poses[start_idx:i],
                    'intrinsics': intrinsics[start_idx:i],
                    'images': images[start_idx:i]
                }
                all_data['train'][scene].append(segment)
                start_idx = i

        # Add the last segment
        segment = {
            'poses': poses[start_idx :],
            'intrinsics': intrinsics[start_idx :],
            'images': images[start_idx :]
        }
        all_data['train'][scene].append(segment)
        
        # if len(poses) != train_frame_num + test_frame_num:
        #     count += 1
        #     print(all_data[scene]['test'])
            # for i in all_data[scene]['train']:
            #     print(i['images'])

    # # Save all data to a compressed pickle file
    with gzip.open(output_dir / 'all_data.pickle.gz', 'wb') as f:
        pickle.dump(all_data, f)
    print(f"error: {count}")

if __name__ == '__main__':
    # root = '/mnt/datasets/scannetpp_processed/iphone'
    # original_dir = '/mnt/datasets/scannetpp'
    # output_dir = '/mnt/datasets/scannetpp_processed/iphone'
    root = '/mnt/datasets/scannetpp_processed/iphone'
    original_dir = '/mnt/datasets/scannetpp'
    output_dir = '/mnt/datasets/scannetpp_processed/iphone'
    process_scenes(root, original_dir, output_dir)
