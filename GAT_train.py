import os
import time
import logging
import torch
import hydra
import torch.optim as optim
import yaml  

from ema_pytorch import EMA
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning import seed_everything
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy

# from misc.logger import setup_logger
from evaluation.evaluator import Evaluator
from datasets.util import create_datasets_GAT
from GAT_trainer import Trainer
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

    for batch_idx, inputs in enumerate(train_loader):
        # instruct the model which novel frames to render
        inputs["target_frame_ids"] = cfg.model.gauss_novel_frames
        losses, outputs = trainer(inputs)

        optimiser.zero_grad(set_to_none=True)
        fabric.backward(losses["loss/total"])
        optimiser.step()
        if ema is not None:
            ema.update()
        
        step = trainer.step

        early_phase = batch_idx % trainer.cfg.run.log_frequency == 0 and step < 6000
        if fabric.is_global_zero:
            learning_rate = lr_scheduler.get_lr()
            if type(learning_rate) is list:
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
            early_phase = (step < 6000) and (step % 500 == 0)
            if (early_phase or step % cfg.run.val_frequency == 0): # and step != 0:
                model_eval = ema if ema is not None else trainer.model
                trainer.validate(model_eval, evaluator, val_loader, device=fabric.device)

        if (early_phase or step % cfg.run.val_frequency == 0): # and step != 0:
            torch.cuda.empty_cache()
            
        trainer.step += 1
        lr_scheduler.step()

def load_config():
    config_path = Path(__file__).resolve().parent / 'configs' / 'GAT_config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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

    GAT_cfg = load_config()['train']

    # set up random set
    torch.set_float32_matmul_precision('high')
    seed_everything(GAT_cfg['seed'])

    # set up training precision
    fabric = Fabric(
        accelerator="cuda",
        devices = GAT_cfg['num_gpus'],
        strategy = DDPStrategy(find_unused_parameters=True),
        precision = GAT_cfg['mixed_precision']
    )
    fabric.launch()
    fabric.barrier()
    print("Loaded datasets")

    # set up model
    trainer = Trainer(cfg, GAT_cfg)
    model = trainer.model

    # set up optimiser
    optimiser = optim.Adam(model.parameters_to_train, float(GAT_cfg['optimiser']['learning_rate']))
    def lr_lambda(*args):
        threshold = GAT_cfg['optimiser']['scheduler_lambda_step_size']
        if trainer.step < threshold:
            return 1.0
        else:
            return 0.1
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimiser, lr_lambda
    )

    if GAT_cfg['use_ema'] and fabric.is_global_zero: #Exponential Moving Average (EMA)
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
    elif GAT_cfg['ckpt_path']:
        model.load_model(GAT_cfg['ckpt_path'])

    trainer, optimiser = fabric.setup(trainer, optimiser)
    # set up dataset
    train_dataset, train_loader = create_datasets_GAT(cfg, split="train")
    train_loader = fabric.setup_dataloaders(train_loader)
    if fabric.is_global_zero:
        # if cfg.train.logging:
        #     trainer.set_logger(setup_logger(cfg))
        val_dataset, val_loader = create_datasets_GAT(cfg, split="val")
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
