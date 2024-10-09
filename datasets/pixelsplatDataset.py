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

# Load configuration from GAT_config.yaml
def load_config():
    config_path = Path(__file__).resolve().parent.parent / 'configs' / 'GAT_config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class pixelsplatDataset(Dataset):
    def __init__(self):
        # Load the configuration from YAML file
        self.cfg = load_config()['dataset']        
        # Load the .torch file

        self.base_dir = Path(__file__).resolve().parent.parent
        dataset_folder = Path(self.cfg['file_path'])
        dataset_folder = dataset_folder.resolve()
        specific_files = self.cfg.get('specific_files', []) 

        self.data = []  
        # Load specific files if provided
        if specific_files:
            for file_name in specific_files:
                file_path = dataset_folder / (file_name + '.torch')  
                if file_path.exists():  # Ensure the file exists
                    loaded_data = torch.load(file_path)
                    self.data.extend(loaded_data)  
                else:
                    print(f"File {file_path} not found.")
        else: 
            for file_name in os.listdir(dataset_folder):
                if file_name.endswith('.torch'): 
                    file_path = dataset_folder / file_name
                    loaded_data = torch.load(file_path)
                    self.data.extend(loaded_data) 

        self.image_processor = ImageProcessor(self.cfg)

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
        if self.cfg['is_train']:
            self.mode = 'train'
        else:
            self.mode = 'test'

        # Prepare the sequence data
        self._seq_data = {}
        for element in self.data:
            key = element['key']
            self._seq_data[key] = element
        self._seq_keys = list(self._seq_data.keys())

        if self.cfg['is_train']:
            self._seq_key_src_idx_pairs = self._full_index(self._seq_keys, 
                self._seq_data, 
                self._left_offset,                    # 0 when sampling dilation randomly
                (self.frame_count-1) * fixed_dilation # 0 when sampling dilation randomly
            )
            if self.cfg['subset'] != -1: # use cfg.subset source frames, they might come from the same sequence
                self._seq_key_src_idx_pairs = self._seq_key_src_idx_pairs[:self.cfg['subset']] * (len(self._seq_key_src_idx_pairs) // self.cfg['subset']) 
        else:
            # Load the test split indices or generate indices based on the strategy
            if self.cfg['test_split_path']:
                test_split_path = Path(__file__).resolve().parent / self.cfg['test_split_path']
                self._seq_key_src_idx_pairs = self._load_split_indices(test_split_path)
            else:
                if load_config()['eval']['video_mode']:
                    self._seq_key_src_idx_pairs = self._generate_video_indices()
                elif self.cfg['random_selection']:
                    self._seq_key_src_idx_pairs = self._generate_random_indices()
                else:
                    self._seq_key_src_idx_pairs = self._generate_defined_indices(-30, 30)
   
    def _full_index(self, seq_keys, seq_data, left_offset, extra_frames):
        # skip_bad = self.cfg['skip_bad_shape']
        # if skip_bad:
            # fn = self.data_path / "valid_seq_ids.train.pickle.gz"
            # valid_seq_ids = pickle.load(gzip.open(fn, "rb"))
        key_id_pairs = []
        for seq_key in seq_keys:
            seq_len = len(seq_data[seq_key]['timestamps'])
            frame_ids = [i + left_offset for i in range(seq_len - extra_frames)]
            # if skip_bad:
            #     good_frames = valid_seq_ids[seq_key]
            #     frame_ids = [f_id for f_id in frame_ids if f_id in good_frames]
            seq_key_id_pairs = [(seq_key, f_id) for f_id in frame_ids]
            key_id_pairs += seq_key_id_pairs
        return key_id_pairs
    
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
        for key, element in self._seq_data.items():
            num_frames = len(element['timestamps'])
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
        
        for key, element in self._seq_data.items():
            num_frames = len(element['timestamps'])

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

    def _generate_video_indices(self):
        """Generate indices such that the middle frame is the source, and all other frames are targets."""
        key_id_pairs = []
        for key, element in self._seq_data.items():
            num_frames = len(element['timestamps'])
            if num_frames < 2:
                continue  # Skip if there are not enough frames

            # Set the middle frame as the source frame
            src_idx = num_frames // 2

            # All other frames are target frames
            tgt_indices = [i for i in range(num_frames) if i != src_idx]
            
            key_id_pairs.append((key, [src_idx] + tgt_indices))
        return key_id_pairs
    
    def __len__(self):
        # Return the total number of frame sets in the dataset
        return len(self._seq_key_src_idx_pairs)

    def __getitem__(self, index):
        # Get the sequence key and frame indices
        # print(len(self._seq_key_src_idx_pairs))
        if self.cfg['is_train']:
            seq_key, src_idx = self._seq_key_src_idx_pairs[index]
            element_data = self._seq_data[seq_key]
            seq_len = len(element_data["timestamps"])

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
            element_data = self._seq_data[seq_key]

        total_frame_num = len(src_and_tgt_frame_idxs)
        frame_names = list(range(total_frame_num))

        inputs = {}

        # Iterate over the frames and process each frame
        for frame_name, frame_idx in zip(frame_names, src_and_tgt_frame_idxs):
            # Process the intrinsic and pose for the current frame
            intrinsic_and_pose = element_data['cameras'][frame_idx]
            intrinsic = intrinsic_and_pose[0:6]  # Extract intrinsic parameters
            pose = intrinsic_and_pose[6:18]
            pose = np.array(pose).reshape(3, 4)  # Reshape pose to 3x4

            inputs_K_tgt, inputs_K_src, inputs_inv_K_src, inputs_T_c2w = self.image_processor.process_intrinsics_and_pose(intrinsic, pose)
            # Process the image for the current frame
            img_tensor = element_data['images'][frame_idx]
            inputs_color, inputs_color_aug = self.image_processor.process_image(img_tensor)

            # Prepare the inputs dictionary
            inputs[("K_tgt", frame_name)] = inputs_K_tgt
            inputs[("K_src", frame_name)] = inputs_K_src
            inputs[("inv_K_src", frame_name)] = inputs_inv_K_src
            inputs[("color", frame_name, 0)] = inputs_color
            inputs[("color_aug", frame_name, 0)] = inputs_color_aug
            inputs[("T_c2w", frame_name)] = inputs_T_c2w
            inputs[("T_w2c", frame_name)] = torch.linalg.inv(inputs_T_c2w)

        # Additional metadata
        input_frame_idx = src_and_tgt_frame_idxs[0]  # The source frame
        timestamp = self._seq_data[seq_key]["timestamps"][input_frame_idx]
        inputs[("frame_id", 0)] = f"{self.mode}+{seq_key}+{timestamp}"
        inputs[("total_frame_num", 0)] = total_frame_num

        return inputs

class ImageProcessor:
    def __init__(self, cfg=None):
        # Configuration with defaults
        self.cfg = cfg

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
        do_color_aug = self.cfg['is_train'] and random.random() > 0.5 and self.cfg['color_aug']
        if do_color_aug:
            self.color_aug_fn = T.ColorJitter(
                brightness=self.brightness, 
                contrast=self.contrast, 
                saturation=self.saturation, 
                hue=self.hue
            )
        else:
            self.color_aug_fn = (lambda x: x)

    def load_image_from_tensor(self, img_tensor):
        # Convert the tensor to bytes
        img_bytes = img_tensor.numpy().tobytes()

        # Open the image using PIL
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        return img

    def process_image(self, img_tensor):
        # Load image from tensor
        img = self.load_image_from_tensor(img_tensor)
        
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
    
    def process_intrinsics_and_pose(self, intrinsic, pose):
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
        K_scale_source[0, 2] *= (self.image_size[1] + self.cfg['pad_border_aug'] * 2)
        K_scale_source[1, 2] *= (self.image_size[0] + self.cfg['pad_border_aug'] * 2)

        # Compute the inverse of the scaled source intrinsic matrix
        inv_K_source = np.linalg.pinv(K_scale_source)

        # Convert to PyTorch tensors
        inputs_K_scale_target = torch.from_numpy(K_scale_target)
        inputs_K_scale_source = torch.from_numpy(K_scale_source)
        inputs_inv_K_source = torch.from_numpy(inv_K_source)

        # Process the extrinsic matrix (pose)
        c2w = data_to_c2w(pose)
        inputs_T_c2w = torch.from_numpy(c2w)

        return inputs_K_scale_target, inputs_K_scale_source, inputs_inv_K_source, inputs_T_c2w
    
def process_projs(proj):
    # Pose in dataset is normalized by resolution
    # Need to unnormalize it for metric projection
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = proj[0]
    K[1, 1] = proj[1]
    K[0, 2] = proj[2]
    K[1, 2] = proj[3]
    return K

def pose_to_4x4(w2c):
    if w2c.shape[0] == 3:
        w2c = np.concatenate((w2c.astype(np.float32),
                             np.array([[0, 0, 0, 1]], dtype=np.float32)), axis=0)
    return w2c

def data_to_c2w(w2c):
    w2c = pose_to_4x4(w2c)
    c2w = np.linalg.inv(w2c)
    return c2w