import os
import time
import logging
import torch
import hydra
import torch.optim as optim
import yaml  
from tqdm import tqdm

from ema_pytorch import EMA
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning import seed_everything
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy

# from misc.logger import setup_logger
from evaluation.evaluator import Evaluator
from datasets.util import create_datasets
from trainer import Trainer
from pathlib import Path

def run_epoch(fabric,
              trainer,
              ema,
              train_loader,
              val_loader,
              optimiser,
              lr_scheduler,
              evaluator):
    """Run a single epoch of training and validation
    """
    cfg = trainer.cfg
    trainer.model.set_train()

    if fabric.is_global_zero:
        logging.info("Training on epoch {}".format(trainer.epoch))

    # max_iterations = 25002
    for batch_idx, inputs in enumerate(tqdm(train_loader, desc="Training", 
                                            total=len(train_loader), dynamic_ncols=True)):
        step = trainer.step
        # if step >= max_iterations:
        #     break
        # instruct the model which novel frames to render
        inputs["target_frame_ids"] = cfg.model.gauss_novel_frames
        losses, outputs = trainer(inputs)

        # Log losses
        total_loss = losses["loss/total"]
        trainer.writer.add_scalar('Loss/total', total_loss, trainer.step)
        if cfg.model.gaussian_rendering:
            trainer.writer.add_scalar('Loss/gaussian_regularization', losses["loss/big_gauss_reg_loss"], trainer.step)
            if cfg.model.predict_offset:
                trainer.writer.add_scalar('Loss/offset', losses["loss/gauss_offset_reg"], trainer.step)  
            trainer.writer.add_scalar('Loss/reconstruction', losses["loss/rec"], trainer.step) 
            # trainer.writer.add_scalar('Loss/rec/mse', losses["loss/rec"]["loss/mse"], trainer.step) 
            # trainer.writer.add_scalar('Loss/rec/ssim', losses["loss/rec"]["loss/ssim"], trainer.step) 
            # trainer.writer.add_scalar('Loss/rec/lpips', losses["loss/rec"]["loss/lpips"], trainer.step)
        
        optimiser.zero_grad(set_to_none=True)
        fabric.backward(losses["loss/total"])
        optimiser.step()

        if ema is not None:
            ema.update()

        early_phase = batch_idx % trainer.cfg.run.log_frequency == 0 and step < 6000
        if fabric.is_global_zero:
            learning_rate = lr_scheduler.get_lr()
            if isinstance(learning_rate, list):
                learning_rate = max(learning_rate)
            
            # save the loss and scales
            trainer.log_scalars("train", outputs, losses, learning_rate)

            # log less frequently after the first 2000 steps to save time & disk space
            late_phase = step % 2000 == 0
            # save the visual results
            if early_phase or late_phase:
                trainer.log("train", inputs, outputs)
            # save the model
            if step % cfg.run.save_frequency == 0 and step != 0:
                trainer.model.save_model(optimiser, step, ema)
            # save the validation results
            early_phase = (step < 20000) and (step % 500 == 0) #500
            if early_phase or step % cfg.run.val_frequency == 0:
                with torch.no_grad():
                    model_eval = ema if ema is not None else trainer.model
                    trainer.validate(model_eval, evaluator, val_loader, device=fabric.device)

        # Clean up and free GPU memory
        if early_phase or step % cfg.run.val_frequency == 0 or step % 500 == 0: #################@@@@@@@@@@@@@@@@
            torch.cuda.empty_cache()

        # # Clear up loss and outputs to free memory
        # del losses, outputs
        # torch.cuda.empty_cache()

        trainer.step += 1
        lr_scheduler.step()



@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None
)
def main(cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    # set up the output directory
    output_dir = hydra_cfg['runtime']['output_dir']
    os.chdir(output_dir)
    logging.info(f"Working dir: {output_dir}")

    # set up random set
    torch.set_float32_matmul_precision('high')
    seed_everything(cfg.run.random_seed)

    # set up training precision
    fabric = Fabric(
        accelerator="cuda",
        devices = cfg.train.num_gpus,
        strategy = DDPStrategy(find_unused_parameters=True),
        precision = cfg.train.mixed_precision
    )
    fabric.launch()
    fabric.barrier()

    # set up model
    trainer = Trainer(cfg)
    model = trainer.model

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

    if cfg.train.ema.use and fabric.is_global_zero: #Exponential Moving Average (EMA)
        ema = EMA(  
            model, 
            beta=cfg.train.ema.beta,
            update_every=cfg.train.ema.update_every,
            update_after_step=cfg.train.ema.update_after_step
        )
        ema = fabric.to_device(ema)
    else:
        ema = None

    # set up checkpointing
    if (ckpt_dir := model.checkpoint_dir()).exists():
        # resume training
        model.load_model(ckpt_dir, optimiser=optimiser)
        print(f"resume training using checkpoint from {ckpt_dir}")
    elif cfg.train.load_weights_folder:
        model.load_model(cfg.ckpt_path)
        print(f"train using existing checkpoint from {cfg.ckpt_path}")

    trainer, optimiser = fabric.setup(trainer, optimiser)
    # set up dataset
    train_dataset, train_loader = create_datasets(cfg, split="train")
    train_loader = fabric.setup_dataloaders(train_loader)
    if fabric.is_global_zero:
        # if cfg.train.logging:
        #     trainer.set_logger(setup_logger(cfg))
        val_dataset, val_loader = create_datasets(cfg, split="val") 
        evaluator = Evaluator()
        evaluator = fabric.to_device(evaluator)
    else:
        val_loader = None
        evaluator = None
    # launch training
    trainer.epoch = 0
    trainer.start_time = time.time()
    for trainer.epoch in range(cfg.optimiser.num_epochs):
        run_epoch(fabric, trainer, ema, train_loader, val_loader, optimiser, lr_scheduler, evaluator)


if __name__ == "__main__":
    main()
