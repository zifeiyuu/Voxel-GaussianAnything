import torch
from PIL import Image
import numpy as np
import io
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
import random
import yaml  
import os
import logging
import json
import cv2
import pickle
import gzip
from tqdm import tqdm
from matplotlib import pyplot as plt

from datasets.scannetpp.masks import calculate_in_frustum_mask_single
from datasets.data import process_projs, data_to_c2w, pil_loader, get_sparse_depth
from datasets.tardataset import TarDataset
from misc.localstorage import copy_to_local_storage, extract_tar, get_local_dir
from misc.depth import estimate_depth_scale, estimate_depth_scale_ransac, depthmap_to_camera_coordinates_torch
from misc.visualise_3d import storePly#, save_depth # for debugging

from IPython import embed
logger = logging.getLogger(__name__)

class scannetppDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg

        self.split = split
        self.dataset_folder = Path(self.cfg.dataset.data_path)
        self.is_train = self.split == "train"
        self.mode = cfg.dataset.mode
        # if this is a relative path to the code dir, make it absolute
        if not self.dataset_folder.is_absolute(): 
            code_dir = Path(__file__).parents[1]
            relative_path = self.dataset_folder
            self.dataset_folder = code_dir / relative_path
            if not self.dataset_folder.exists():
                raise FileNotFoundError(f"Relative path {relative_path} does not exist")
        elif not self.dataset_folder.exists():
            raise FileNotFoundError(f"Absolute path {self.dataset_folder} does not exist")
        
        self.dataset_folder = self.dataset_folder.resolve()

        self.image_size = (self.cfg.dataset.height, self.cfg.dataset.width)
        self.color_aug = self.cfg.dataset.color_aug
        # Padding function if border augmentation is required
        if self.cfg.dataset.pad_border_aug != 0:
            self.pad_border_fn = T.Pad((self.cfg.dataset.pad_border_aug, self.cfg.dataset.pad_border_aug))
        self.num_scales = len(cfg.model.scales)
        self.novel_frames = list(cfg.model.gauss_novel_frames)
        self.frame_count = len(self.novel_frames) + 1
        self.max_fov = cfg.dataset.max_fov
        self.interp = Image.LANCZOS
        # self.loader = pil_loader

        self.to_tensor = T.ToTensor()
        # self.loader = pil_loader
        if cfg.dataset.normalize:
            self.to_tensor = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.to_tensor = T.ToTensor()

        self.specific_files = self.cfg.dataset.get('specific_files', []) 

        try:
            # Newer version with tuple ranges
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
        except TypeError:
            # Fallback for older versions
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize, self.depth_resize = {}, {}
        for i in range(self.num_scales):
            s = 2 ** i
            new_size = (self.image_size[0] // s, self.image_size[1] // s)
            self.resize[i] = T.Resize(new_size, interpolation=self.interp)
            self.depth_resize[i] = T.Resize(new_size, interpolation=Image.NEAREST)
        
        # load dilation file
        self.dilation = cfg.dataset.dilation
        self.max_dilation = cfg.dataset.max_dilation
        if isinstance(self.dilation, int):
            self._left_offset = ((self.frame_count - 1) // 2) * self.dilation ######
            fixed_dilation = self.dilation
        else: # enters here when cfg.dataset.dilation = random
            self._left_offset = 0
            fixed_dilation = 0

        # self.png_depth_scale = cfg.dataset.png_depth_scale  #######

        self._pose_data = self._load_pose_data()

        # Fetch the seq_keys to use
        if self.specific_files:
            self._seq_keys = self.specific_files
        else:
            self._seq_keys = []
            if self.split == "train":
                bad_scenes = [] #['303745abc7']
                train_list = cfg.dataset.train_split_path
                for train_list_file in train_list:
                    with open(os.path.join(self.dataset_folder, "splits", train_list_file), "r") as f:
                        # Read lines and extend them to the sequence keys
                        self._seq_keys.extend([line.strip() for line in f if line.strip() not in bad_scenes])
            elif self.split == "val" or self.split == "test":
                bad_scenes = [] #['cc5237fd77']
                test_list = cfg.dataset.test_split_path
                for test_list_file in test_list:
                    with open(os.path.join(self.dataset_folder, "splits", test_list_file), "r") as f:
                        # Read lines and extend them to the sequence keys
                        self._seq_keys.extend([line.strip() for line in f if line.strip() not in bad_scenes])

            logger.info(f"Removing scenes that have frames with no valid depths: {bad_scenes}")
            self._seq_keys = [s for s in self._seq_keys if s not in bad_scenes]

        self._missing_scans = []
        with open(os.path.join(self.dataset_folder, "missing_scans.txt"), "r") as f:
            # Read lines and extend them to the sequence keys
            self._missing_scans.extend([line.strip() for line in f])

        if self.is_train:
            self._seq_key_src_idx_pairs = self._full_index(
                self._left_offset,                    # 0 when sampling dilation randomly
                (self.frame_count-1) * fixed_dilation # 0 when sampling dilation randomly
            )
            if self.cfg.dataset.subset != -1: # use cfg.subset source frames, they might come from the same sequence
                self._seq_key_src_idx_pairs = self._seq_key_src_idx_pairs[:self.cfg['subset']] * (len(self._seq_key_src_idx_pairs) // self.cfg['subset']) 
        else:
            #generate indices based on the strategy
            if self.cfg.video_mode:
                self._seq_key_src_idx_pairs = self._generate_video_indices()
            elif not self.cfg.dataset.random_selection:
                self._seq_key_src_idx_pairs = self._generate_defined_indices()
            else:
                self._seq_key_src_idx_pairs = self._generate_random_indices()

    def __len__(self):
        # Return the total number of frame sets in the dataset
        return len(self._seq_key_src_idx_pairs)
    
    def _load_pose_data(self):
        print(f"{self.split} Dataset loading data...")
        file_path = self.dataset_folder  / self.mode / "all_data.pickle.gz"
        with gzip.open(file_path, "rb") as f:
            pose_data = pickle.load(f)
        return pose_data['train']
    
    def _full_index(self, left_offset, extra_frames):
        key_id_pairs = []
        for key in self._seq_keys:
            for period_index, period in enumerate(self._pose_data[key]):
                seq_len = len(period["poses"])
                frame_ids = [i + left_offset for i in range(seq_len - extra_frames)]
                seq_key_id_pairs = [(key, period_index, f_id) for f_id in frame_ids]
                key_id_pairs += seq_key_id_pairs
        return key_id_pairs

    def _generate_random_indices(self):
        """Generate random frame indices for testing."""
        key_id_pairs = []
        for key in self._seq_keys:
            for period_index, period in enumerate(self._pose_data[key]):
                seq_len = len(period["poses"])
                if seq_len < 4:
                    continue  # Skip if there are not enough frames
                # Randomly select four frame indices
                frame_indices = random.sample(range(seq_len), 4)
                frame_indices.sort()  # Sort indices to maintain a consistent order
                key_id_pairs.append((key, period_index, frame_indices))
        return key_id_pairs

    def _generate_defined_indices(self, gap1=-1, gap2=0, gap3=1):
        """Generate default frame indices for varied gaps, including zero or negative."""
        key_id_pairs = []
        
        for key in self._seq_keys:
            for period_index, period in enumerate(self._pose_data[key]):
                seq_len = len(period["poses"])

                # Calculate the minimum and maximum index for the source frame to allow all gaps
                min_gap = min(gap1, gap2, gap3)
                max_gap = max(gap1, gap2, gap3)

                if seq_len <= max(abs(min_gap), abs(max_gap)):
                    print("No enough frames to accommodate the largest gap magnitude.")
                    continue

                min_index = max(0, -min_gap)  # Ensure the smallest negative gap doesn't go below 0
                max_index = seq_len - 1 - max_gap  # Ensure the largest positive gap doesn't exceed seq_len-1
                
                if min_index > max_index:
                    print("Adjusted index range invalid, skipping.")
                    continue

                # src_idx = random.randint(min_index, max_index)
                src_idx = (min_index + max_index) // 2
                
                # Calculate indices for target frames
                tgt_1_idx = src_idx + gap1
                tgt_2_idx = src_idx + gap2
                tgt_3_idx = src_idx + gap3

                key_id_pairs.append((key, period_index, [src_idx, tgt_1_idx, tgt_2_idx, tgt_3_idx]))
            
        return key_id_pairs

    def _generate_video_indices(self):
        """Generate indices such that the middle frame is the source, and all other frames are targets."""
        key_id_pairs = []
        for key in self._seq_keys:
            for period_index, period in enumerate(self._pose_data[key]):
                seq_len = len(period["poses"])
                if seq_len < 2:
                    continue  # Skip if there are not enough frames

                # Set the middle frame as the source frame
                src_idx = seq_len // 2
                # All other frames are target frames
                tgt_indices = [i for i in range(seq_len) if i != src_idx]

                key_id_pairs.append((key, period_index, [src_idx] + tgt_indices))
        return key_id_pairs

    def process_image(self, img_path):
        # Set up color augmentation function
        do_color_aug = self.is_train and random.random() > 0.5 and self.color_aug
        if do_color_aug:
            self.color_aug_fn = T.ColorJitter(
                brightness=self.brightness, 
                contrast=self.contrast, 
                saturation=self.saturation, 
                hue=self.hue
            )
        else:
            self.color_aug_fn = (lambda x: x)

        options=cv2.IMREAD_COLOR
        if str(img_path).endswith(('.exr', 'EXR')):
            options = cv2.IMREAD_ANYDEPTH
        img = cv2.imread(img_path, options)
        if img is None:
            raise IOError(f'Could not load image={img_path} with {options=}')
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        # Resize the image according to the first scale
        img_scale = self.resize[0](img)
        # Convert the resized image to a tensor
        inputs_color = self.to_tensor(img_scale)

        # Apply padding and color augmentation if needed
        if self.cfg.dataset.pad_border_aug != 0:
            inputs_color_aug = self.to_tensor(self.color_aug_fn(self.pad_border_fn(img_scale)))
        else:
            inputs_color_aug = self.to_tensor(self.color_aug_fn(img_scale))

        return inputs_color, inputs_color_aug, img.size
    
    def process_depth(self, depth_path):
        options=cv2.IMREAD_UNCHANGED
        if str(depth_path).endswith(('.exr', 'EXR')):
            options = cv2.IMREAD_ANYDEPTH
        depthmap = cv2.imread(depth_path, options)
        if depthmap is None:
            return None
        if depthmap.ndim == 3:
            depthmap = cv2.cvtColor(depthmap, cv2.COLOR_BGR2RGB)
        depthmap = depthmap.astype(np.float32) / 1000
        
        depthmap_scale = self.to_tensor(depthmap)
        depthmap_scale = self.depth_resize[0](depthmap_scale)
       
        return depthmap_scale.squeeze()
    
    def process_pose(self, c2w):
        inputs_T_c2w = torch.from_numpy(c2w)
        return inputs_T_c2w
    
    def process_intrinsics(self, K, original_height, original_width):
        # Scale the intrinsic matrix for the target image size
        K_scale_target = K.copy()
        K_scale_target[0, :] *= self.image_size[1] / original_width
        K_scale_target[1, :] *= self.image_size[0] / original_height

        # Scale the intrinsic matrix for the source image size (considering padding)
        K_scale_source = K.copy()
        K_scale_source[0, 0] *= self.image_size[1] / original_width
        K_scale_source[1, 1] *= self.image_size[0] / original_height
        K_scale_source[0, 2] *= (self.image_size[1] + self.cfg.dataset.pad_border_aug * 2) / original_width
        K_scale_source[1, 2] *= (self.image_size[0] + self.cfg.dataset.pad_border_aug * 2) / original_height
        

        # Compute the inverse of the scaled source intrinsic matrix
        inv_K_source = np.linalg.pinv(K_scale_source)

        # Convert to PyTorch tensors
        inputs_K_scale_target = torch.from_numpy(K_scale_target)
        inputs_K_scale_source = torch.from_numpy(K_scale_source)
        inputs_inv_K_source = torch.from_numpy(inv_K_source)

        return inputs_K_scale_target, inputs_K_scale_source, inputs_inv_K_source
    
    def __getitem__(self, index):
        # Get the sequence key and frame indices
        if self.is_train:
            seq_key, period_idx, src_idx = self._seq_key_src_idx_pairs[index]
            seq_len = len(self._pose_data[seq_key][period_idx]["poses"])
            pose_data = self._pose_data[seq_key][period_idx]

            if self.cfg.dataset.frame_sampling_method == "two_forward_one_back":
                if self.dilation == "random":
                    dilation = torch.randint(1, self.max_dilation, (1,)).item()
                    left_offset = dilation 
                else:
                    dilation = self.dilation
                    left_offset = self._left_offset
                src_and_tgt_frame_idxs = [src_idx - left_offset + i * dilation for i in range(self.frame_count)]
                src_and_tgt_frame_idxs = [src_idx] + [max(min(i, seq_len-1), 0) for i in src_and_tgt_frame_idxs if i != src_idx]
            elif self.cfg.dataset.frame_sampling_method  == "random":
                target_frame_idxs = torch.randperm( 4 * self.max_dilation + 1 )[:self.frame_count] - 2 * self.max_dilation
                src_and_tgt_frame_idxs = [src_idx] + [max(min(i + src_idx, seq_len-1), 0) for i in target_frame_idxs.tolist() if i != 0][:self.frame_count - 1]                
            frame_names = [0] + self.novel_frames  
        else:
            seq_key, period_idx, src_and_tgt_frame_idxs = self._seq_key_src_idx_pairs[index]
            pose_data = self._pose_data[seq_key][period_idx]
            total_frame_num = len(src_and_tgt_frame_idxs)
            frame_names = list(range(total_frame_num))

        # have_depth = seq_key not in self._missing_scans
        have_depth = True
        inputs = {}

        # Iterate over the frames and process each frame
        for frame_name, frame_idx in zip(frame_names, src_and_tgt_frame_idxs):
            # Process the intrinsic and pose for the current frame
            c2w = pose_data['poses'][frame_idx]
            inputs_T_c2w = self.process_pose(c2w)

            # Process the image for the current frame
            img_name = pose_data['images'][frame_idx][:-4] + '.jpg'
            img_path =  self.dataset_folder / self.mode / seq_key / 'images' / img_name
            inputs_color, inputs_color_aug, orig_size = self.process_image(img_path)

            intrinsic = pose_data['intrinsics'][frame_idx]
            inputs_K_tgt, inputs_K_src, inputs_inv_K_src = self.process_intrinsics(intrinsic, orig_size[1], orig_size[0])

            if have_depth:
                depth_name = pose_data['images'][frame_idx][:-4] + '.png'
                depth_path = self.dataset_folder / self.mode / seq_key / 'depth' / depth_name
                depth = self.process_depth(depth_path)

            # Additional metadata
            first_img_name = pose_data['images'][0][:-4]  # The source frame
            inputs[("frame_id", 0)] = f"{seq_key}+{first_img_name}+{self.split}"
            
            # Prepare the inputs dictionary
            inputs[("K_tgt", frame_name)] = inputs_K_tgt
            inputs[("K_src", frame_name)] = inputs_K_src
            inputs[("inv_K_src", frame_name)] = inputs_inv_K_src
            inputs[("color", frame_name, 0)] = inputs_color
            inputs[("color_aug", frame_name, 0)] = inputs_color_aug
            inputs[("T_c2w", frame_name)] = inputs_T_c2w
            inputs[("T_w2c", frame_name)] = torch.linalg.inv(inputs_T_c2w)
            if have_depth:
                inputs[("depth_sparse", frame_name)] = depth

        if not self.is_train:
            inputs[("total_frame_num", 0)] = total_frame_num

        return inputs


    # def get_view(self, sequence, view_idx, resolution):

    #     # RGB Image
    #     rgb_path = self.color_paths[sequence][view_idx]
    #     rgb_image = imread_cv2(rgb_path)

    #     # Depthmap
    #     depth_path = self.depth_paths[sequence][view_idx]
    #     depthmap = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED)
    #     depthmap = depthmap.astype(np.float32)
    #     depthmap = depthmap / self.png_depth_scale

    #     # C2W Pose
    #     c2w = self.c2ws[sequence][view_idx]

    #     # Camera Intrinsics
    #     intrinsics = self.intrinsics[sequence]

    #     # Resize
    #     rgb_image, depthmap, intrinsics = crop_resize_if_necessary(
    #         rgb_image, depthmap, intrinsics, resolution
    #     )

    #     view = {
    #         'original_img': rgb_image,
    #         'depthmap': depthmap,
    #         'camera_pose': c2w,
    #         'camera_intrinsics': intrinsics,
    #         'dataset': 'scannet++',
    #         'label': f"scannet++/{sequence}",
    #         'instance': f'{view_idx}',
    #         'is_metric_scale': True,
    #         'sky_mask': depthmap <= 0.0,
    #     }
    #     return view
    
if __name__ == "__main__":
    
    # from .loading_utils import read_extrinsics_binary, read_h5py, read_img
    
    dataset = ScannetppDataset(data_path="/scratch/bcdm/ymao3/scannetpp")
    
    for idx in np.random.permutation(len(dataset)):
        data = dataset[idx]