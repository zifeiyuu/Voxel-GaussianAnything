#!/bin/bash

torchrun --nproc_per_node=1 train_torch.py --config-name=gat_config
