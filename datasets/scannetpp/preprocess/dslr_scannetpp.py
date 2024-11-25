#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Script to pre-process the scannet++ dataset.
# Usage:
# python datasets_preprocess/modified_scannetpp.py --scannetpp_dir /mnt/datasets/scannetpp_processed/demo2 --output_dir /mnt/datasets/scannetpp_processed/out --pyopengl-platform egl
# --------------------------------------------------------
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
sys.path.append('/mnt/ziyuxiao/code/dust3r2')
from dust3r.datasets.utils.cropping import rescale_image_depthmap
import dust3r.utils.geometry as geometry

from IPython import embed

inv = np.linalg.inv
norm = np.linalg.norm
REGEXPR_DSLR = re.compile(r'^DSC(?P<frameid>\d+).JPG$')
REGEXPR_IPHONE = re.compile(r'frame_(?P<frameid>\d+).jpg$')

DEBUG_VIZ = None  # 'iou'
if DEBUG_VIZ is not None:
    import matplotlib.pyplot as plt  # noqa


OPENGL_TO_OPENCV = np.float32([[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]])


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scannetpp_dir', required=True)
    parser.add_argument('--output_dir', default='data/scannetpp_processed')
    parser.add_argument('--target_resolution', default=920, type=int, help="images resolution")
    parser.add_argument('--pyopengl-platform', type=str, default='', help='PyOpenGL env variable')
    return parser


def pose_from_qwxyz_txyz(elems):
    qw, qx, qy, qz, tx, ty, tz = map(float, elems)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
    pose[:3, 3] = (tx, ty, tz)
    return np.linalg.inv(pose)  # returns cam2world


def get_frame_number(name, cam_type='dslr'):
    if cam_type == 'dslr':
        regex_expr = REGEXPR_DSLR
    elif cam_type == 'iphone':
        regex_expr = REGEXPR_IPHONE
    else:
        raise NotImplementedError(f'wrong {cam_type=} for get_frame_number')
    matches = re.match(regex_expr, name)
    return matches['frameid']


def load_sfm(sfm_dir, cam_type='dslr'):
    # load cameras
    with open(osp.join(sfm_dir, 'cameras.txt'), 'r') as f:
        raw = f.read().splitlines()[3:]  # skip header

    intrinsics = {}
    for camera in tqdm(raw, position=1, leave=False):
        camera = camera.split(' ')
        intrinsics[int(camera[0])] = [camera[1]] + [float(cam) for cam in camera[2:]]

    # load images
    with open(os.path.join(sfm_dir, 'images.txt'), 'r') as f:
        raw = f.read().splitlines()
        raw = [line for line in raw if not line.startswith('#')]  # skip header

    img_idx = {}
    img_infos = {}
    for image, points in tqdm(zip(raw[0::2], raw[1::2]), total=len(raw) // 2, position=1, leave=False):
        image = image.split(' ')
        points = points.split(' ')

        idx = image[0]
        img_name = image[-1]
        assert img_name not in img_idx, 'duplicate db image: ' + img_name
        img_idx[img_name] = idx  # register image name

        current_points2D = {int(i): (float(x), float(y))
                            for i, x, y in zip(points[2::3], points[0::3], points[1::3]) if i != '-1'}
        img_infos[idx] = dict(intrinsics=intrinsics[int(image[-2])],
                              path=img_name,
                              frame_id=get_frame_number(img_name, cam_type),
                              cam_to_world=pose_from_qwxyz_txyz(image[1: -2]),
                              sparse_pts2d=current_points2D)

    return img_idx, img_infos


def subsample_img_infos(img_infos, num_images, allowed_name_subset=None):
    img_infos_val = [(idx, val) for idx, val in img_infos.items()]
    if allowed_name_subset is not None:
        img_infos_val = [(idx, val) for idx, val in img_infos_val if val['path'] in allowed_name_subset]

    if len(img_infos_val) > num_images:
        img_infos_val = sorted(img_infos_val, key=lambda x: x[1]['frame_id'])
        kept_idx = np.round(np.linspace(0, len(img_infos_val) - 1, num_images)).astype(int).tolist()
        img_infos_val = [img_infos_val[idx] for idx in kept_idx]
    return {idx: val for idx, val in img_infos_val}


def undistort_images(intrinsics, rgb, mask):
    camera_type = intrinsics[0]

    width = int(intrinsics[1])
    height = int(intrinsics[2])
    fx = intrinsics[3]
    fy = intrinsics[4]
    cx = intrinsics[5]
    cy = intrinsics[6]
    distortion = np.array(intrinsics[7:])

    K = np.zeros([3, 3])
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy
    K[2, 2] = 1

    K = geometry.colmap_to_opencv_intrinsics(K)
    if camera_type == "OPENCV_FISHEYE":
        assert len(distortion) == 4

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K,
            distortion,
            (width, height),
            np.eye(3),
            balance=0.0,
        )
        # Make the cx and cy to be the center of the image
        new_K[0, 2] = width / 2.0
        new_K[1, 2] = height / 2.0

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1)
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, distortion, (width, height), 1, (width, height), True)
        map1, map2 = cv2.initUndistortRectifyMap(K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1)

    undistorted_image = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    undistorted_mask = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    new_K = geometry.opencv_to_colmap_intrinsics(new_K)
    return width, height, new_K, undistorted_image, undistorted_mask


def process_scenes(root, output_dir, target_resolution):
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # default values from
    # https://github.com/scannetpp/scannetpp/blob/main/common/configs/render.yml
    znear = 0.05
    zfar = 20.0

    # we will undistort them and render their depth
    renderer = pyrender.OffscreenRenderer(0, 0)
    scenes = [d for d in os.listdir(os.path.join(root, 'data')) if os.path.isdir(os.path.join(root, 'data', d))]

    all_data = {}  # Dictionary to hold all data
    log_dir = os.path.join(output_dir, 'missing_scans.txt')
    
    for scene in tqdm(scenes, position=0, leave=True): 
        print(scene)
        data_dir = os.path.join(root, 'data', scene)
        dir_dslr = os.path.join(data_dir, 'dslr')
        dir_scans = os.path.join(data_dir, 'scans')

        assert os.path.isdir(data_dir)
        assert os.path.isdir(dir_dslr)

        output_dir_scene = os.path.join(output_dir, scene)
        scene_metadata_path = osp.join(output_dir_scene, 'scene_metadata.npz')
        if osp.isfile(scene_metadata_path):
            continue
        
        has_scans = os.path.isdir(dir_scans)
        if has_scans:
            ply_path = os.path.join(dir_scans, 'mesh_aligned_0.05.ply')
            # load the mesh
            with open(ply_path, 'rb') as f:
                mesh_kwargs = trimesh.exchange.ply.load_ply(f)
            mesh_scene = trimesh.Trimesh(**mesh_kwargs) 
            mesh = pyrender.Mesh.from_trimesh(mesh_scene, smooth=False)
            pyrender_scene = pyrender.Scene()
            pyrender_scene.add(mesh)
        else:
            with open(log_dir, 'a') as log_file: 
                log_file.write(scene + '\n') 

        # set up the output paths
        output_dir_scene_rgb = os.path.join(output_dir_scene, 'images')
        output_dir_scene_depth = os.path.join(output_dir_scene, 'depth')
        os.makedirs(output_dir_scene_rgb, exist_ok=True)
        os.makedirs(output_dir_scene_depth, exist_ok=True)

        sfm_dir_dslr = os.path.join(dir_dslr, 'colmap')
        rgb_dir_dslr = os.path.join(dir_dslr, 'resized_images')
        mask_dir_dslr = os.path.join(dir_dslr, 'resized_anon_masks')

        # read colmap reconstruction, we will only use the intrinsics and pose here
        img_idx_dslr, img_infos_dslr = load_sfm(sfm_dir_dslr, cam_type='dslr')
        dslr_paths = {
            "in_colmap": sfm_dir_dslr,
            "in_rgb": rgb_dir_dslr,
            "in_mask": mask_dir_dslr,
        }

        # resize the image to a more manageable size and render depth
        rgb_dir = dslr_paths['in_rgb']
        mask_dir = dslr_paths['in_mask']
        trajectories_t = []
        intrinsics_t = []

        scene_data = {'intrinsics': [], 'poses': [], 'images': []}
        scene_img_names = os.listdir(rgb_dir)

        for imgname in tqdm(scene_img_names, position=1, leave=False):
            if not imgname.endswith('.JPG'):
                continue  # skip non-JPG files
            imgidx = img_idx_dslr[imgname]
            img_infos_idx = img_infos_dslr[imgidx]
            rgb = np.array(Image.open(os.path.join(rgb_dir, img_infos_idx['path'])))
            mask = np.array(Image.open(os.path.join(mask_dir, img_infos_idx['path'][:-3] + 'png')))

            _, _, K, rgb, mask = undistort_images(img_infos_idx['intrinsics'], rgb, mask)

            # rescale_image_depthmap assumes opencv intrinsics
            intrinsics = geometry.colmap_to_opencv_intrinsics(K)
            image, mask, intrinsics = rescale_image_depthmap(
                rgb, mask, intrinsics, (target_resolution, target_resolution * 3.0 / 4))

            W, H = image.size
            intrinsics = geometry.opencv_to_colmap_intrinsics(intrinsics)

            # update inpace img_infos_idx
            img_infos_idx['intrinsics'] = intrinsics
            rgb_outpath = os.path.join(output_dir_scene_rgb, img_infos_idx['path'][:-3] + 'jpg')
            image.save(rgb_outpath)

            # render depth image
            if has_scans:
                depth_outpath = os.path.join(output_dir_scene_depth, img_infos_idx['path'][:-3] + 'png')
                renderer.viewport_width, renderer.viewport_height = W, H
                fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
                camera = pyrender.camera.IntrinsicsCamera(fx, fy, cx, cy, znear=znear, zfar=zfar)
                camera_node = pyrender_scene.add(camera, pose=img_infos_idx['cam_to_world'] @ OPENGL_TO_OPENCV)

                depth = renderer.render(pyrender_scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
                pyrender_scene.remove_node(camera_node)  # dont forget to remove camera

                depth = (depth * 1000).astype('uint16')
                # invalidate depth from mask before saving
                depth_mask = (mask < 255)
                depth[depth_mask] = 0
                Image.fromarray(depth).save(depth_outpath)

            if imgname.startswith('DSC'):
                imgidx = img_idx_dslr[imgname]
                img_infos_idx = img_infos_dslr[imgidx]
            else:
                raise ValueError('invalid image name')

            intrinsics_t.append(img_infos_idx['intrinsics'])
            trajectories_t.append(img_infos_idx['cam_to_world'])

            # scene_data['images'].append(imgname[:-4])
            # scene_data['intrinsics'].append(img_infos_idx['intrinsics'])
            # scene_data['poses'].append(img_infos_idx['cam_to_world'])

        # all_data[scene] = scene_data

        intrinsics_t = np.stack(intrinsics_t, axis=0)
        trajectories_t = np.stack(trajectories_t, axis=0)
        # save metadata for this scene
        np.savez(scene_metadata_path,
                trajectories=trajectories_t,
                intrinsics=intrinsics_t,
                images=scene_img_names)
        

        del img_infos_dslr
        if has_scans:
            del pyrender_scene

    # # Save all data to a compressed pickle file
    # with gzip.open(output_dir / 'all_data.pickle.gz', 'wb') as f:
    #     pickle.dump(all_data, f)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.pyopengl_platform.strip():
        os.environ['PYOPENGL_PLATFORM'] = args.pyopengl_platform
    process_scenes(args.scannetpp_dir, args.output_dir, args.target_resolution)


# python modified_scannetpp.py --scannetpp_dir /mnt/datasets/scannetpp_processed/demo2 --output_dir /mnt/datasets/scannetpp_processed/out --pyopengl-platform egl

# python modified_scannetpp.py --scannetpp_dir /mnt/datasets/scannetpp --output_dir /mnt/datasets/scannetpp_processed --pyopengl-platform egl