import torch
import logging
from torch.utils.data import DataLoader, DistributedSampler

from packaging.version import Version
from datasets.re10k import Re10KDataset
from datasets.nyu.dataset import NYUv2Dataset
from datasets.kitti import KITTIDataset
from datasets.pixelsplatDataset import pixelsplatDataset
from datasets.scannetppDataset import scannetppDataset
from datasets.arkitscenesDataset import arkitscenesDataset

def create_datasets(cfg, split="val", distributed=False, rank=0):

    datasets_dict = {
        "pixelsplat": pixelsplatDataset,
        "scannetpp": scannetppDataset,
        "arkitscenes": arkitscenesDataset
    }[cfg.dataset.name]

    dataset = datasets_dict(cfg, split)
    logging.info("There are {:d} {} items. Using {}\n".format(len(dataset), split, cfg.dataset)
    )
    shuffle = True if split == "train" else False
    if distributed:
        train_sampler = DistributedSampler(dataset, num_replicas=cfg.train.num_gpus, rank=rank)
        data_loader = DataLoader(dataset, 
            batch_size=cfg.data_loader.batch_size, 
            num_workers=cfg.data_loader.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            collate_fn=custom_collate,
            sampler=train_sampler)
    else:
        data_loader = DataLoader(
            dataset,
            cfg.data_loader.batch_size,
            shuffle=shuffle,
            num_workers=cfg.data_loader.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            collate_fn=custom_collate,
        )

    return dataset, data_loader

if Version(torch.__version__) < Version("1.11"):
    from torch.utils.data._utils.collate import default_collate
else:
    from torch.utils.data import default_collate


def custom_collate(batch):
    all_keys = batch[0].keys()
    dense_keys = [k for k in all_keys if "sparse" not in k[0]]
    sparse_keys = [k for k in all_keys if "sparse" in k[0]]
    dense_batch = [{k: b[k] for k in dense_keys} for b in batch]
    sparse_batch = {k: [b[k] for b in batch] for k in sparse_keys}
    dense_batch = default_collate(dense_batch)
    batch = sparse_batch | dense_batch
    return batch