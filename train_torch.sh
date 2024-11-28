#!/bin/bash

torchrun --nproc_per_node=4 train_torch.py --config-name=gat_config
