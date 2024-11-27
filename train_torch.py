import os
import time
import logging
import hydra
import torch
import torch.distributed as dist
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from ema_pytorch import EMA
from omegaconf import DictConfig
from hydra import main
from hydra.core.hydra_config import HydraConfig

from evaluation.evaluator import Evaluator
from datasets.util import create_datasets
from trainer import Trainer

from IPython import embed

def run_epoch(trainer: Trainer, ema, train_loader, val_loader, optimiser, lr_scheduler, evaluator):
    """Run a single epoch of training and validation"""
    cfg = trainer.cfg
    trainer.model.train()  # Set model to training mode
    local_rank = dist.get_rank()

    logging.info("Training on epoch {}".format(trainer.epoch))

    for batch_idx, inputs in enumerate(tqdm(train_loader, desc="Training", total=len(train_loader), dynamic_ncols=True)):
        # Instruct the model which novel frames to render
        inputs["target_frame_ids"] = cfg.model.gauss_novel_frames
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(local_rank, non_blocking=True)
        
        losses, outputs = trainer(inputs)

        optimiser.zero_grad(set_to_none=True)
        losses["loss/total"].backward()  # Backpropagate the total loss
        optimiser.step()

        if ema is not None:
            ema.update()
        
        step = trainer.step

        early_phase = batch_idx % trainer.cfg.run.log_frequency == 0 and step < 10000
        learning_rate = lr_scheduler.get_lr()
        if isinstance(learning_rate, list):
            learning_rate = max(learning_rate)

        # Log training metrics
        if local_rank == 0:  # Only log from the main process
            trainer.log_scalars("train", outputs, losses, learning_rate)

            # Log model outputs and save the model at defined intervals
            if early_phase or step % 1000 == 0:
                trainer.log("train", inputs, outputs)

            if step % cfg.run.save_frequency == 0 and step != 0:
                if isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel):
                    trainer.model.module.save_model(optimiser, step, ema)
                else:
                    trainer.model.save_model(optimiser, step, ema)

            if step > 0 and (
                early_phase or step % cfg.run.val_frequency == 0 and evaluator is not None):
                with torch.no_grad():
                    model_eval = ema if ema is not None else trainer.model
                    trainer.validate(model_eval, evaluator, val_loader, device='cuda')

        # Clean up and free GPU memory
        if early_phase or step % cfg.run.val_frequency == 0:
            torch.cuda.empty_cache()

        trainer.step += 1
        lr_scheduler.step()

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None
)
def main(cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    os.chdir(output_dir)
    logging.info(f"Working dir: {output_dir}")

    # Initialize DDP
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    set_seed_everywhere(cfg.run.random_seed + local_rank)

    # Set up model
    trainer = Trainer(cfg)
    # trainer.set_logger()
    model = trainer.model

    # Wrap model in DDP
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model.to(local_rank), device_ids=[local_rank], find_unused_parameters=True)
    
    # set up optimiser
    # optimiser = optim.AdamW(model.parameters_to_train, lr=cfg.optimiser.learning_rate, weight_decay=cfg.optimiser.weight_decay) 
    optimiser = optim.Adam(model.parameters_to_train, cfg.optimiser.learning_rate)
    num_warmup_steps = cfg.optimiser.num_warmup_steps
    max_training_steps = cfg.optimiser.max_training_steps
    min_lr_ratio = cfg.optimiser.min_lr_ratio

    if cfg.optimiser.mode == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            T_max=max_training_steps,  # sets the period of the cosine cycle
            eta_min= min_lr_ratio * cfg.optimiser.learning_rate  # sets the minimum learning rate
        )
    elif cfg.optimiser.mode == 'linear':
        def lr_linear(num_warmup_steps, num_training_steps, min_lr_ratio=0):
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return cfg.optimiser.warmup_ratio
                # Linear decrease from 1 to min_lr_ratio
                return max(min_lr_ratio, (num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
            
            return lr_lambda
        
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimiser, 
            lr_linear(num_warmup_steps, max_training_steps, min_lr_ratio)
        )
    else:
        def lr_lambda(*args):
            threshold = cfg.optimiser.scheduler_lambda_step_size
            if trainer.step < threshold:
                return 1.0
            else:
                return 0.1
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimiser, lr_lambda
        )
    
    # Set up Exponential Moving Average (EMA)
    if cfg.train.ema.use: 
        ema = EMA(
            model, 
            beta=cfg.train.ema.beta,
            update_every=cfg.train.ema.update_every,
            update_after_step=cfg.train.ema.update_after_step
        ).to(local_rank)
    else:
        ema = None

    # Load model from checkpoint
    if (ckpt_dir := model.checkpoint_dir()).exists():
        model.load_model(ckpt_dir, optimiser=optimiser)
        print(f"Resume training using checkpoint from {ckpt_dir}")
    elif cfg.train.load_weights_folder:
        model.load_model(cfg.ckpt_path)
        print(f"Train using existing checkpoint from {cfg.ckpt_path}")

    trainer.model = ddp_model

    # Set up dataset
    train_dataset, train_loader = create_datasets(cfg, split="train")

    val_dataset, val_loader, evaluator = None, None, None
    if local_rank == 0:
        val_dataset, val_loader = create_datasets(cfg, split="val")
        evaluator = Evaluator()

    # Launch training
    trainer.epoch = 0
    trainer.start_time = time.time()
    for trainer.epoch in range(cfg.optimiser.num_epochs):
        run_epoch(trainer, ema, train_loader, val_loader, optimiser, lr_scheduler, evaluator)

    # Clean up DDP
    dist.destroy_process_group()

if __name__ == "__main__":
    # Use torchrun to launch the script, set the number of processes
    main()