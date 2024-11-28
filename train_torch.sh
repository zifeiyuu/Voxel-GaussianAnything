#!/bin/bash

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=4 train_torch.py --config-name=gat_config
