import torch
from PIL import Image
import numpy as np
import io
import torchvision.transforms as T
from torch.utils.data import Dataset
from pathlib import Path
import random
import yaml  
import os
import logging
import json
import cv2

logger = logging.getLogger(__name__)

# Load configuration from GAT_config.yaml
def load_config():
    config_path = Path(__file__).resolve().parent.parent.parent / 'configs' / 'scannetpp.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class scannetppDataset(Dataset):
    def __init__(self, GAT_cfg, stage):
        self.cfg = load_config()  

        self.stage = stage
        if stage == "train":
            self.is_train = True
        else:
            self.is_train = False

        self.video_mode = GAT_cfg['video_mode']
        #Scannet dataset folder root
        self.root = Path(self.cfg['file_path'])
        self.root = self.root.resolve()
        specific_files = self.cfg.get('specific_files', []) 

        self.image_processor = ImageProcessor(self.cfg, self.is_train)

        self.frame_count = len(self.cfg['novel_frames']) + 1
        # load dilation file
        self.dilation = self.cfg['dilation']
        self.max_dilation = self.cfg['max_dilation']
        if isinstance(self.dilation, int):
            self._left_offset = ((self.frame_count - 1) // 2) * self.dilation
            fixed_dilation = self.dilation
        else: # enters here when cfg.dataset.dilation = random
            self._left_offset = 0
            fixed_dilation = 0
        self.png_depth_scale = 1000.0

        # Dictionaries to store the data for each scene
        self.color_paths = {}
        self.depth_paths = {}
        self.intrinsics = {}
        self.c2ws = {}

        # Fetch the seq_keys to use
        if specific_files:
            self.seq_keys = specific_files
        else:
            if self.stage == "train":
                sequence_file = os.path.join(self.root, "raw", "splits", "nvs_sem_train.txt")
                bad_scenes = ['303745abc7']
            elif self.stage == "val" or self.stage == "test":
                sequence_file = os.path.join(self.root, "raw", "splits", "nvs_sem_val.txt")
                bad_scenes = ['cc5237fd77']
            with open(sequence_file, "r") as f:
                self.seq_keys = f.read().splitlines()
            logger.info(f"Removing scenes that have frames with no valid depths: {bad_scenes}")
            self.seq_keys = [s for s in self.seq_keys if s not in bad_scenes]

        # Collect information for every sequence
        scenes_with_no_good_frames = []
        for key in self.seq_keys:

            input_raw_folder = os.path.join(self.root, 'raw', 'data', key)
            input_processed_folder = os.path.join(self.root, 'processed', 'data', key)

            # Load Train & Test Splits
            frame_file = os.path.join(input_raw_folder, "dslr", "train_test_lists.json")
            with open(frame_file, "r") as f:
                train_test_list = json.load(f)

            # Camera Metadata
            cams_metadata_path = f"{input_processed_folder}/dslr/nerfstudio/transforms_undistorted.json"
            with open(cams_metadata_path, "r") as f:
                cams_metadata = json.load(f)

            # Load the nerfstudio/transforms.json file to check whether each image is blurry
            nerfstudio_transforms_path = f"{input_raw_folder}/dslr/nerfstudio/transforms.json"
            with open(nerfstudio_transforms_path, "r") as f:
                nerfstudio_transforms = json.load(f)

            # Create a reverse mapping from image name to the frame information and nerfstudio transform
            # (as transforms_undistorted.json does not store the frames in the same order as train_test_lists.json)
            file_path_to_frame_metadata = {}
            file_path_to_nerfstudio_transform = {}
            for frame in cams_metadata["frames"]:
                file_path_to_frame_metadata[frame["file_path"]] = frame
            for frame in nerfstudio_transforms["frames"]:
                file_path_to_nerfstudio_transform[frame["file_path"]] = frame

            # Fetch the pose for every frame
            sequence_color_paths = []
            sequence_depth_paths = []
            sequence_c2ws = []
            for train_file_name in train_test_list["train"]:
                is_bad = file_path_to_nerfstudio_transform[train_file_name]["is_bad"]
                if is_bad:
                    continue
                sequence_color_paths.append(f"{input_processed_folder}/dslr/undistorted_images/{train_file_name}")
                sequence_depth_paths.append(f"{input_processed_folder}/dslr/undistorted_depths/{train_file_name.replace('.JPG', '.png')}")
                frame_metadata = file_path_to_frame_metadata[train_file_name]
                c2w = np.array(frame_metadata["transform_matrix"], dtype=np.float32)
                P = np.array([
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]]
                ).astype(np.float32)
                c2w = P @ c2w @ P.T
                sequence_c2ws.append(c2w)

            if len(sequence_color_paths) == 0:
                logger.info(f"No good frames for sequence: {key}")
                scenes_with_no_good_frames.append(key)
                continue

            # Get the intrinsics data for the frame
            K = np.eye(3, dtype=np.float32)
            K[0, 0] = cams_metadata["fl_x"]
            K[1, 1] = cams_metadata["fl_y"]
            K[0, 2] = cams_metadata["cx"]
            K[1, 2] = cams_metadata["cy"]

            self.color_paths[key] = sequence_color_paths
            self.depth_paths[key] = sequence_depth_paths
            self.c2ws[key] = sequence_c2ws
            self.intrinsics[key] = K

        # Remove scenes with no good frames
        self.seq_keys = [s for s in self.seq_keys if s not in scenes_with_no_good_frames]

        if self.is_train and not self.video_mode:
            self._seq_key_src_idx_pairs = self._full_index(
                self._left_offset,                    # 0 when sampling dilation randomly
                (self.frame_count-1) * fixed_dilation # 0 when sampling dilation randomly
            )
            if self.cfg['subset'] != -1: # use cfg.subset source frames, they might come from the same sequence
                self._seq_key_src_idx_pairs = self._seq_key_src_idx_pairs[:self.cfg['subset']] * (len(self._seq_key_src_idx_pairs) // self.cfg['subset']) 
        else:
            #generate indices based on the strategy
            if self.video_mode:
                self._seq_key_src_idx_pairs = self._generate_video_indices()
            elif self.cfg['random_selection']:
                self._seq_key_src_idx_pairs = self._generate_random_indices()
            else:
                self._seq_key_src_idx_pairs = self._generate_defined_indices(-30, 30)
   
    def _full_index(self, left_offset, extra_frames):
        # skip_bad = self.cfg['skip_bad_shape']
        # if skip_bad:
            # fn = self.data_path / "valid_seq_ids.train.pickle.gz"
            # valid_seq_ids = pickle.load(gzip.open(fn, "rb"))
        key_id_pairs = []
        for key in self.seq_keys:
            seq_len = len(self.color_paths[key])
            frame_ids = [i + left_offset for i in range(seq_len - extra_frames)]
            # if skip_bad:
            #     good_frames = valid_seq_ids[seq_key]
            #     frame_ids = [f_id for f_id in frame_ids if f_id in good_frames]
            seq_key_id_pairs = [(key, f_id) for f_id in frame_ids]
            key_id_pairs += seq_key_id_pairs
        return key_id_pairs

    def _generate_random_indices(self):
        """Generate random frame indices for testing."""
        key_id_pairs = []
        for key in self.seq_keys:
            seq_len = len(self.color_paths[key])
            if seq_len < 4:
                continue  # Skip if there are not enough frames
            # Randomly select four frame indices
            frame_indices = random.sample(range(seq_len), 4)
            frame_indices.sort()  # Sort indices to maintain a consistent order
            key_id_pairs.append((key, frame_indices))
        return key_id_pairs

    def _generate_defined_indices(self, gap1=5, gap2=10):
        """Generate default frame indices: (random, random+gap1, random+gap2, random)."""
        key_id_pairs = []
        
        for key in self.seq_keys:
            seq_len = len(self.color_paths[key])

            # Ensure gaps are not zero
            if gap1 == 0 or gap2 == 0 or gap1 >= gap2:
                print("Gaps cannot be zero.")
                continue
            
            if gap1 * gap2 > 0:
                # Both gaps are positive
                if max(abs(gap1), abs(gap2)) >= seq_len:
                    print("No enough frames.")
                    continue
                if gap1 > 0 and gap2 > 0:
                    src_idx = random.randint(0, seq_len - 1 - max(gap1, gap2))
                
                # Both gaps are negative
                elif gap1 < 0 and gap2 < 0:
                    src_idx = random.randint(-min(gap1, gap2), seq_len - 1)

            else:
                if gap2 - gap1 >= seq_len:
                    print("No enough frames.")
                    continue
                src_idx = random.randint(-gap1, seq_len - 1 - gap2)
            
            # Determine the next two frames at +gap1 and +gap2 offsets
            tgt_1_idx = src_idx + gap1
            tgt_2_idx = src_idx + gap2

            # Ensure the indices are within valid bounds
            tgt_1_idx = max(0, min(tgt_1_idx, seq_len - 1))
            tgt_2_idx = max(0, min(tgt_2_idx, seq_len - 1)) 

            # Randomly select a target frame different from the others
            tgt_random_idx = random.choice(
                [i for i in range(seq_len) if i not in {src_idx, tgt_1_idx, tgt_2_idx}]
            )

            key_id_pairs.append((key, [src_idx, tgt_1_idx, tgt_2_idx, tgt_random_idx]))
        
        return key_id_pairs

    def _generate_video_indices(self):
        """Generate indices such that the middle frame is the source, and all other frames are targets."""
        key_id_pairs = []
        for key in self.seq_keys:
            seq_len = len(self.color_paths[key])
            if seq_len < 2:
                continue  # Skip if there are not enough frames

            # Set the middle frame as the source frame
            src_idx = seq_len // 2

            # All other frames are target frames
            tgt_indices = [i for i in range(seq_len) if i != src_idx]
            
            key_id_pairs.append((key, [src_idx] + tgt_indices))
        return key_id_pairs
    

    def __len__(self):
        # Return the total number of frame sets in the dataset
        return len(self._seq_key_src_idx_pairs)

    def __getitem__(self, index):
        # Get the sequence key and frame indices
        # print(len(self._seq_key_src_idx_pairs))
        if self.is_train:
            seq_key, src_idx = self._seq_key_src_idx_pairs[index]
            seq_len = len(self.color_paths[seq_key])

            if self.cfg['frame_sampling_method'] == "two_forward_one_back":
                if self.dilation == "random":
                    dilation = torch.randint(1, self.max_dilation, (1,)).item()
                    left_offset = dilation 
                else:
                    dilation = self.dilation
                    left_offset = self._left_offset
                src_and_tgt_frame_idxs = [src_idx - left_offset + i * dilation for i in range(self.frame_count)]
                src_and_tgt_frame_idxs = [src_idx] + [max(min(i, seq_len-1), 0) for i in src_and_tgt_frame_idxs if i != src_idx]
            elif self.cfg['frame_sampling_method'] == "random":
                target_frame_idxs = torch.randperm( 4 * self.max_dilation + 1 )[:self.frame_count] - 2 * self.max_dilation
                src_and_tgt_frame_idxs = [src_idx] + [max(min(i + src_idx, seq_len-1), 0) for i in target_frame_idxs.tolist() if i != 0][:self.frame_count - 1]                
        else:
            seq_key, src_and_tgt_frame_idxs = self._seq_key_src_idx_pairs[index]

        total_frame_num = len(src_and_tgt_frame_idxs)
        frame_names = list(range(total_frame_num))

        inputs = {}

        # Iterate over the frames and process each frame
        for frame_name, frame_idx in zip(frame_names, src_and_tgt_frame_idxs):
            # Process the intrinsic and pose for the current frame
            c2w = self.c2ws[seq_key][frame_idx]
            inputs_T_c2w = self.image_processor.process_pose(c2w)

            intrinsic = self.intrinsics[seq_key]
            inputs_K_tgt, inputs_K_src, inputs_inv_K_src = self.image_processor.process_intrinsics(intrinsic)
            
            # Process the image for the current frame
            img_path = self.color_paths[seq_key][frame_idx]
            inputs_color, inputs_color_aug = self.image_processor.process_image(img_path)

            depth_path = self.depth_paths[seq_key][frame_idx]
            inputs_depth = self.image_processor.process_depth(depth_path)

            # Prepare the inputs dictionary
            inputs[("K_tgt", frame_name)] = inputs_K_tgt
            inputs[("K_src", frame_name)] = inputs_K_src
            inputs[("inv_K_src", frame_name)] = inputs_inv_K_src
            inputs[("color", frame_name, 0)] = inputs_color
            inputs[("color_aug", frame_name, 0)] = inputs_color_aug
            inputs[("depth", frame_name, 0)] = inputs_depth
            inputs[("T_c2w", frame_name)] = inputs_T_c2w
            inputs[("T_w2c", frame_name)] = torch.linalg.inv(inputs_T_c2w)

        # Additional metadata
        input_frame_idx = src_and_tgt_frame_idxs[0]  # The source frame
        input_img_name = self.color_paths[seq_key][input_frame_idx]
        inputs[("frame_id", 0)] = f"{self.stage}+{seq_key}+{input_img_name}"
        inputs[("total_frame_num", 0)] = total_frame_num

        return inputs

class ImageProcessor:
    def __init__(self, cfg, is_train):
        # Configuration with defaults
        self.cfg = cfg

        self.is_train = is_train
        # Set image size from config
        self.image_size = (self.cfg['height'], self.cfg['width'])
        # Set the number of scales from config
        self.num_scales = len(self.cfg['scales'])
        # Define resize transformations based on scales
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            new_size = (self.image_size[0] // s, self.image_size[1] // s)
            self.resize[i] = T.Resize(new_size, interpolation=T.InterpolationMode.BILINEAR)

        # To tensor transformation
        self.to_tensor = T.ToTensor()

        # Padding function if border augmentation is required
        if self.cfg['pad_border_aug'] != 0:
            self.pad_border_fn = T.Pad((self.cfg['pad_border_aug'], self.cfg['pad_border_aug']))
        else:
            self.pad_border_fn = lambda x: x  # No padding if augmentation is 0

        # Set up brightness, contrast, saturation, and hue for color augmentation
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

        # Set up color augmentation function
        do_color_aug = self.is_train and random.random() > 0.5 and self.cfg['color_aug']
        if do_color_aug:
            self.color_aug_fn = T.ColorJitter(
                brightness=self.brightness, 
                contrast=self.contrast, 
                saturation=self.saturation, 
                hue=self.hue
            )
        else:
            self.color_aug_fn = (lambda x: x)

    def process_image(self, img_path):
        options=cv2.IMREAD_COLOR
        if img_path.endswith(('.exr', 'EXR')):
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
        if self.cfg['pad_border_aug'] != 0:
            inputs_color_aug = self.to_tensor(self.color_aug_fn(self.pad_border_fn(img_scale)))
        else:
            inputs_color_aug = self.to_tensor(self.color_aug_fn(img_scale))

        return inputs_color, inputs_color_aug
    
    def process_depth(self, depth_path):
        options=cv2.IMREAD_UNCHANGED
        if depth_path.endswith(('.exr', 'EXR')):
            options = cv2.IMREAD_ANYDEPTH
        depthmap = cv2.imread(depth_path, options)
        if depthmap is None:
            raise IOError(f'Could not load image={depth_path} with {options=}')
        if depthmap.ndim == 3:
            depthmap = cv2.cvtColor(depthmap, cv2.COLOR_BGR2RGB)
        depthmap = depthmap.astype(np.float32)
        
        # Resize the image according to the first scale
        # inputs_depth = self.resize[0](depthmap)

        return depthmap
    
    def process_pose(self, c2w):
        inputs_T_c2w = torch.from_numpy(c2w)
        return inputs_T_c2w
    
    def process_intrinsics(self, K):
        # Scale the intrinsic matrix for the target image size
        K_scale_target = K.copy()
        K_scale_target[0, :] *= self.image_size[1]
        K_scale_target[1, :] *= self.image_size[0]

        # Scale the intrinsic matrix for the source image size (considering padding)
        K_scale_source = K.copy()
        K_scale_source[0, 0] *= self.image_size[1]
        K_scale_source[1, 1] *= self.image_size[0]
        K_scale_source[0, 2] *= (self.image_size[1] + self.cfg['pad_border_aug'] * 2)
        K_scale_source[1, 2] *= (self.image_size[0] + self.cfg['pad_border_aug'] * 2)

        # Compute the inverse of the scaled source intrinsic matrix
        inv_K_source = np.linalg.pinv(K_scale_source)

        # Convert to PyTorch tensors
        inputs_K_scale_target = torch.from_numpy(K_scale_target)
        inputs_K_scale_source = torch.from_numpy(K_scale_source)
        inputs_inv_K_source = torch.from_numpy(inv_K_source)

        return inputs_K_scale_target, inputs_K_scale_source, inputs_inv_K_source
    
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