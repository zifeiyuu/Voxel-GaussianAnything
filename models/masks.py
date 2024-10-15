
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

from GaussianAnything.src.splatt3r_src.loss_mask import calculate_in_frustum_mask
from GaussianAnything.datasets.util import create_datasets

class mask:
    def __init__(self) -> None:
        pass
    def calculate_mask(self, inputs, frame_names):

        self.inputs = inputs
        for frame_name in frame_names:
            if frame_name == 0:
                context_depth = self.inputs[("depth", frame_name)]
                context_intrinsics = self.inputs[("K_tgt", frame_name)]
                context_c2w = self.inputs[("T_c2w", frame_name)]   
            else:             
                target_depth = self.inputs[("depth", frame_name)]
                target_intrinsics = self.inputs[("K_tgt", frame_name)]
                target_c2w = self.inputs[("T_c2w", frame_name)]

        mask = calculate_in_frustum_mask(target_depth, target_intrinsics, target_c2w, context_depth, context_intrinsics, context_c2w)

        return mask