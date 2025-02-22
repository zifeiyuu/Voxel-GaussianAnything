#!/bin/bash

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=8 train_torch.py \
  +experiment=flash3d_train
#--master_port=29501