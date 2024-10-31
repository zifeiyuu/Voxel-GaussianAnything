  python train.py \
  hydra=cluster \
  hydra/launcher=submitit_slurm \
  +hydra.job.tag=gaussian2_unidepthv1 \
  +experiment=layered_re10k \
  model.depth.version=v1 \
  train.logging=false \
  # # -m


# python train.py \
#   +experiment=GAT \
#   train.logging=false \
#   --config-name=gat_config
#   # -m

