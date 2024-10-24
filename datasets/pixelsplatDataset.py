import torch
from PIL import Image
import numpy as np
import io
import torchvision.transforms as T
from torch.utils.data import Dataset
from pathlib import Path
import random 
import os
import pickle
import gzip
from tqdm import tqdm

from datasets.data import process_projs, data_to_c2w, pil_loader, get_sparse_depth
from datasets.tardataset import TarDataset
from misc.localstorage import copy_to_local_storage, extract_tar, get_local_dir

class pixelsplatDataset(Dataset):
    def __init__(self, cfg, split):
        # Load the configuration from YAML file
        self.cfg = cfg     

        self.split = split
        self.data_folder = Path(self.cfg.dataset.data_path)
        self.is_train = self.split == "train"
        self.split_name_for_loading = "train" if self.is_train else "test"
        self.data_folder = self.data_folder / self.split_name_for_loading
        # if this is a relative path to the code dir, make it absolute
        if not self.data_folder.is_absolute(): 
            code_dir = Path(__file__).parents[1]
            relative_path = self.data_folder
            self.data_folder = code_dir / relative_path
            if not self.data_folder.exists():
                raise FileNotFoundError(f"Relative path {relative_path} does not exist")
        elif not self.data_folder.exists():
            raise fileNotFoundError(f"Absolute path {self.data_folder} does not exist")
        self.data_folder = self.data_folder.resolve()

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

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            new_size = (self.image_size[0] // s, self.image_size[1] // s)
            self.resize[i] = T.Resize(new_size, interpolation=self.interp)
        
        # load dilation file
        self.dilation = cfg.dataset.dilation
        self.max_dilation = cfg.dataset.max_dilation
        if isinstance(self.dilation, int):
            self._left_offset = ((self.frame_count - 1) // 2) * self.dilation ######
            fixed_dilation = self.dilation
        else: # enters here when cfg.dataset.dilation = random
            self._left_offset = 0
            fixed_dilation = 0

        self._pose_data = self._load_pose_data()

        self._seq_keys = list(self._pose_data.keys())
        missing_keys_path = self.data_folder / f"{self.split_name_for_loading}_missing_keys.txt"
        with open(missing_keys_path, 'r') as f:
            missing_keys = f.read().splitlines() 
        self._seq_keys = [key for key in self._seq_keys if key not in missing_keys]

        if self.is_train:
            self._seq_key_src_idx_pairs = self._full_index(self._seq_keys, 
                self._pose_data, 
                self._left_offset,                    # 0 when sampling dilation randomly
                (self.frame_count-1) * fixed_dilation # 0 when sampling dilation randomly
            )
            if self.cfg.dataset.subset != -1: # use cfg.dataset.subset source frames, they might come from the same sequence
                self._seq_key_src_idx_pairs = self._seq_key_src_idx_pairs[:cfg.dataset.subset] * (len(self._seq_key_src_idx_pairs) // cfg.dataset.subset)
        else:
            # Load the test split indices or generate indices based on the strategy
            if self.cfg.dataset.test_split_path:
                print(f"test using split file {self.cfg.dataset.test_split_path}")
                test_split_path = Path(__file__).resolve().parent / ".." / cfg.dataset.test_split_path 
                if self.cfg.video_mode:
                    self._seq_key_src_idx_pairs = self._generate_video_indices(test_split_path)
                else:
                    self._seq_key_src_idx_pairs = self._load_split_indices(test_split_path)
            else:
                if self.cfg.dataset.random_selection:
                    self._seq_key_src_idx_pairs = self._generate_random_indices()
                else:
                    self._seq_key_src_idx_pairs = self._generate_defined_indices(-30, 30)
        
        self.length = len(self._seq_key_src_idx_pairs)

        if cfg.dataset.from_tar and self.is_train:
            fn = self.data_folder / "all.train.tar"
            self.images_dataset = TarDataset(archive=fn, extensions=(".jpg", ".pickle"))
            self.pcl_dataset = self.images_dataset
        else:
            fn = self.data_folder / f"pcl.{self.split_name_for_loading}.tar"
            if cfg.dataset.copy_to_local:
                self.pcl_dir = self.data_folder / f"pcl_{self.split_name_for_loading}"
            else:
                self.pcl_dataset = TarDataset(archive=fn, extensions=(".jpg", ".pickle"))

    def __len__(self) -> int:
        return self.length

    def _load_pose_data(self):
        print(f"{self.split} Dataset loading data...")
        file_path = self.data_folder  / f"{self.split_name_for_loading}.pickle.gz"
        with gzip.open(file_path, "rb") as f:
            pose_data = pickle.load(f)
        # # Initialize an empty dictionary to store pose data
        # pose_data = {}

        # # Determine the list of files to process
        # if self.specific_files:
        #     file_list = [self.data_folder / (file_name + '.torch') for file_name in self.specific_files]
        # else:
        #     file_list = [self.data_folder / file_name for file_name in os.listdir(self.data_folder) if file_name.endswith('.torch')]

        # # Iterate over the files using tqdm for a progress bar
        # for file_path in tqdm(file_list, desc="Loading Pose Data", unit="file"):
        #     if file_path.exists():
        #         loaded_data = torch.load(file_path)
        #         for element in loaded_data:
        #             key = element['key']
        #             pose_data[key] = element['cameras']
        #     else:
        #         print(f"File {file_path} not found.")
        
        return pose_data

    
    def _full_index(self, seq_keys, pose_data, left_offset, extra_frames):
        skip_bad = self.cfg.dataset.skip_bad_shape
        if skip_bad:
            fn = self.data_folder / "valid_seq_ids.train.pickle.gz"
            valid_seq_ids = pickle.load(gzip.open(fn, "rb"))
        key_id_pairs = []
        for seq_key in seq_keys:
            seq_len = len(pose_data[seq_key]["timestamps"])
            frame_ids = [i + left_offset for i in range(seq_len - extra_frames)]
            if skip_bad:
                good_frames = valid_seq_ids[seq_key]
                frame_ids = [f_id for f_id in frame_ids if f_id in good_frames]
            seq_key_id_pairs = [(seq_key, f_id) for f_id in frame_ids]
            key_id_pairs += seq_key_id_pairs
        return key_id_pairs
    
    def _load_sparse_pcl(self, seq_key):
        fn = f"pcl.{self.split_name_for_loading}/{seq_key}.pickle.gz"
        if self.cfg.dataset.from_tar:
            f = self.pcl_dataset.get_file(fn)
            data = gzip.decompress(f.read())
            return pickle.loads(data)
        else:
            fn = self.pcl_dir / fn
            with gzip.open(fn, "rb") as f:
                data = pickle.load(f)
            return data 
        
    def _load_split_indices(self, index_path):
        """Load the testing split from a text file."""
        def get_key_id(s):
            parts = s.split(" ")
            key = parts[0]
            src_idx = int(parts[1])
            tgt_5_idx = int(parts[2])
            tgt_10_idx = int(parts[3])
            tgt_random_idx = int(parts[4])
            return key, [src_idx, tgt_5_idx, tgt_10_idx, tgt_random_idx]

        with open(index_path, "r") as f:
            lines = f.readlines()
        key_id_pairs = list(map(get_key_id, lines))
        return key_id_pairs

    def _generate_random_indices(self):
        """Generate random frame indices for testing."""
        key_id_pairs = []
        for key in self._seq_keys:
            num_frames = len(self._pose_data[key]["timestamps"])
            if num_frames < 4:
                continue  # Skip if there are not enough frames

            # Randomly select four frame indices
            frame_indices = random.sample(range(num_frames), 4)
            frame_indices.sort()  # Sort indices to maintain a consistent order
            key_id_pairs.append((key, frame_indices))
        return key_id_pairs

    def _generate_defined_indices(self, gap1=5, gap2=10):
        """Generate default frame indices: (random, random+gap1, random+gap2, random)."""
        key_id_pairs = []
        
        for key in self._seq_keys:
            num_frames = len(self._pose_data[key]["timestamps"])

            # Ensure gaps are not zero
            if gap1 == 0 or gap2 == 0 or gap1 >= gap2:
                print("Gaps cannot be zero.")
                continue
            
            if gap1 * gap2 > 0:
                # Both gaps are positive
                if max(abs(gap1), abs(gap2)) >= num_frames:
                    print("No enough frames.")
                    continue
                if gap1 > 0 and gap2 > 0:
                    src_idx = random.randint(0, num_frames - 1 - max(gap1, gap2))
                
                # Both gaps are negative
                elif gap1 < 0 and gap2 < 0:
                    src_idx = random.randint(-min(gap1, gap2), num_frames - 1)

            else:
                if gap2 - gap1 >= num_frames:
                    print("No enough frames.")
                    continue
                src_idx = random.randint(-gap1, num_frames - 1 - gap2)
            
            # Determine the next two frames at +gap1 and +gap2 offsets
            tgt_1_idx = src_idx + gap1
            tgt_2_idx = src_idx + gap2

            # Ensure the indices are within valid bounds
            tgt_1_idx = max(0, min(tgt_1_idx, num_frames - 1))
            tgt_2_idx = max(0, min(tgt_2_idx, num_frames - 1)) 

            # Randomly select a target frame different from the others
            tgt_random_idx = random.choice(
                [i for i in range(num_frames) if i not in {src_idx, tgt_1_idx, tgt_2_idx}]
            )

            key_id_pairs.append((key, [src_idx, tgt_1_idx, tgt_2_idx, tgt_random_idx]))
        
        return key_id_pairs

    def _generate_video_indices(self, index_path):
        """Generate indices such that the middle frame is the source, and all other frames are targets."""
        def get_key(s):
            parts = s.split(" ")
            key = parts[0]
            return key
        
        with open(index_path, "r") as f:
            lines = f.readlines()
        keys = list(map(get_key, lines))
        keys = set(keys)

        key_id_pairs = []
        for key in keys:
            num_frames = len(self._pose_data[key]["timestamps"])
            if num_frames < 2:
                continue  # Skip if there are not enough frames

            # Set the middle frame as the source frame
            src_idx = num_frames // 2

            # All other frames are target frames
            tgt_indices = [i for i in range(num_frames) if i != src_idx]
            
            key_id_pairs.append((key, [src_idx] + tgt_indices))
        return key_id_pairs
    
    def _load_depth(self, key, id):
        path = self.data_folder / f"{self.split_name_for_loading}"
        depth_file = f"{key}/{id}.png"
        if os.path.exists(path / depth_file):
            depth = Image.open(path / depth_file)
            # Scale the saved image using the metadata
            max_value = float(depth.info["max_value"])
            min_value = float(depth.info["min_value"])
            # Scale from uint16 range
            depth = (np.array(depth).astype(np.float32) / (2 ** 16 - 1)) * (max_value - min_value) + min_value
        else:
            # print("Depth file {} is not exist", path / depth_file)
            depth = None
        return depth
    
    def get_data(self, frame_idx, pose_data, image_path, color_aug_fn):
        # Process the intrinsic and pose for the current frame
        # intrinsic_and_pose = pose_data[frame_idx]
        # intrinsic = intrinsic_and_pose[0:6]  # Extract intrinsic parameters
        # pose = intrinsic_and_pose[6:18]
        # pose = np.array(pose).reshape(3, 4)

        # if self.data_folder is not None:
            # depth = self._load_depth(seq_key, frame_idx)
        #     if depth is not None:
        #         depth = self.to_tensor(depth)
        #         depth = F.interpolate(depth[None,...], size=self.image_size, mode="nearest")[0]
        # else:
        #     depth = None

        intrinsic = pose_data["intrinsics"][frame_idx]
        pose = pose_data["poses"][frame_idx]
        # Process the intrinsic matrix
        K = process_projs(intrinsic)

        # Scale the intrinsic matrix for the target image size
        K_scale_target = K.copy()
        K_scale_target[0, :] *= self.image_size[1]
        K_scale_target[1, :] *= self.image_size[0]

        # Scale the intrinsic matrix for the source image size (considering padding)
        K_scale_source = K.copy()
        K_scale_source[0, 0] *= self.image_size[1]
        K_scale_source[1, 1] *= self.image_size[0]
        K_scale_source[0, 2] *= (self.image_size[1] + self.cfg.dataset.pad_border_aug * 2)
        K_scale_source[1, 2] *= (self.image_size[0] + self.cfg.dataset.pad_border_aug * 2)

        # Compute the inverse of the scaled source intrinsic matrix
        inv_K_source = np.linalg.pinv(K_scale_source)

        # Convert to PyTorch tensors
        inputs_K_scale_target = torch.from_numpy(K_scale_target)
        inputs_K_scale_source = torch.from_numpy(K_scale_source)
        inputs_inv_K_source = torch.from_numpy(inv_K_source)

        image_path = os.path.join(image_path , f'{frame_idx}.jpg')
        img = Image.open(image_path).convert('RGB')

        # Resize the image according to the first scale
        img_scale = self.resize[0](img)

        # Convert the resized image to a tensor
        inputs_color = self.to_tensor(img_scale)

        # Apply padding and color augmentation if needed
        if self.cfg.dataset.pad_border_aug != 0:
            inputs_color_aug = self.to_tensor(color_aug_fn(self.pad_border_fn(img_scale)))
        else:
            inputs_color_aug = self.to_tensor(color_aug_fn(img_scale))
        
        # Process the extrinsic matrix (pose)
        c2w = data_to_c2w(pose)
        # original world-to-camera matrix in row-major order and transfer to column-major order
        inputs_T_c2w = torch.from_numpy(c2w)

        return inputs_K_scale_target, inputs_K_scale_source, inputs_inv_K_source, inputs_color, inputs_color_aug, inputs_T_c2w, img.size
  
    def __getitem__(self, index):
        # Get the sequence key and frame indices
        if self.is_train:
            seq_key, src_idx = self._seq_key_src_idx_pairs[index]
            pose_data = self._pose_data[seq_key]
            seq_len = len(pose_data["timestamps"])

            if self.cfg.dataset.frame_sampling_method == "two_forward_one_back":
                if self.dilation == "random":
                    dilation = torch.randint(1, self.max_dilation, (1,)).item()
                    left_offset = dilation  # one frame in the past
                else:
                     # self.dilation and self._left_offsets can be fixed if cfg.dataset.dilation is an int
                    dilation = self.dilation
                    left_offset = self._left_offset
                # frame count is num_novel_frames + 1 for source view
                # sample one frame in backwards time and self.frame_count - 2 into the future  
                # self.frame_count = len(self.novel_frames) + 1, gauss_novel_frames: [1, 2] #
                src_and_tgt_frame_idxs = [src_idx - left_offset + i * dilation for i in range(self.frame_count)]
                src_and_tgt_frame_idxs = [src_idx] + [max(min(i, seq_len-1), 0) for i in src_and_tgt_frame_idxs if i != src_idx]
            elif self.cfg.dataset.frame_sampling_method == "random":
                target_frame_idxs = torch.randperm( 4 * self.max_dilation + 1 )[:self.frame_count] - 2 * self.max_dilation
                src_and_tgt_frame_idxs = [src_idx] + [max(min(i + src_idx, seq_len-1), 0) for i in target_frame_idxs.tolist() if i != 0][:self.frame_count - 1]    
            frame_names = [0] + self.novel_frames            
        else:
            seq_key, src_and_tgt_frame_idxs = self._seq_key_src_idx_pairs[index]
            pose_data = self._pose_data[seq_key]
            total_frame_num = len(src_and_tgt_frame_idxs)
            frame_names = list(range(total_frame_num))

        do_color_aug = self.is_train and random.random() > 0.5 and self.color_aug
        if do_color_aug:
            color_aug = T.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        image_path = os.path.join(self.data_folder , 'image', str(seq_key))

        inputs = {}
        # Iterate over the frames and process each frame
        for frame_name, frame_idx in zip(frame_names, src_and_tgt_frame_idxs):

            inputs_K_tgt, inputs_K_src, inputs_inv_K_src, inputs_color, inputs_color_aug, \
            inputs_T_c2w, orig_size = self.get_data(
                                                frame_idx=frame_idx, 
                                                pose_data = pose_data,
                                                image_path = image_path,
                                                color_aug_fn=color_aug
            )
            if self.cfg.dataset.scale_pose_by_depth:
                sparse_pcl = self._load_sparse_pcl(seq_key)
            if self.cfg.dataset.scale_pose_by_depth:
                # get colmap_image_id
                xyd = get_sparse_depth(pose_data, orig_size, sparse_pcl, frame_idx)
            else:
                xyd = None

            # Additional metadata
            input_frame_idx = src_and_tgt_frame_idxs[0]  # The source frame
            timestamp = pose_data["timestamps"][input_frame_idx]
            inputs[("frame_id", 0)] = f"{self.split}+{seq_key}+{timestamp}"

            # Prepare the inputs dictionary
            inputs[("K_tgt", frame_name)] = inputs_K_tgt
            inputs[("K_src", frame_name)] = inputs_K_src
            inputs[("inv_K_src", frame_name)] = inputs_inv_K_src
            inputs[("color", frame_name, 0)] = inputs_color
            inputs[("color_aug", frame_name, 0)] = inputs_color_aug  
            # original world-to-camera matrix in row-major order and transfer to column-major order
            inputs[("T_c2w", frame_name)] = inputs_T_c2w
            inputs[("T_w2c", frame_name)] = torch.linalg.inv(inputs_T_c2w)
            if xyd is not None and frame_name == 0:
                inputs[("depth_sparse", frame_name)] = xyd

        if not self.is_train:
            inputs[("total_frame_num", 0)] = total_frame_num

        return inputs
