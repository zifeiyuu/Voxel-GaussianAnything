#!/bin/bash

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=4 --master_port=29501 train_torch.py --config-name=gat_config #--master_port=29501
# CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 train_torch.py --config-name=gat_config #--master_port=29501