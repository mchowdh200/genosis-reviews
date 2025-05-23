import argparse
import os
import random
import sys
from datetime import datetime
from functools import partial
from glob import glob
from types import SimpleNamespace

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         StochasticWeightAveraging)
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import nn, optim
from torch.utils import data
from torchmetrics import MeanSquaredLogError
from torchvision import ops

from data_utils.gt_datasets import GTDataset, pad_data
from models.encoder import Conv1DEncoder, SiameseModule


def NestedNamespace(x: dict):
    """Convert a nested dict to a nested namespace"""
    return SimpleNamespace(
        **{k: NestedNamespace(v) if isinstance(v, dict) else v for k, v in x.items()}
    )


def compute_steps_per_epoch(
    train_dataset, batch_size, accumulate_grad_batches, n_devices
):
    """
    Compute the total number of optimizer steps for the learning rate scheduler.
    """
    effective_batch_size = batch_size * accumulate_grad_batches * n_devices
    n_batches = int(np.ceil(len(train_dataset) / effective_batch_size))
    return n_batches


def get_dataloaders(
    *,
    P1_train,
    P2_train,
    D_train,
    P1_val,
    P2_val,
    D_val,
    model_type,
    batch_size,
    n_workers,
):
    # make main dataset
    train_ds = GTDataset(
        pos_files=(P1_train, P2_train),
        dist_file=D_train,
    )
    val_ds = GTDataset(
        pos_files=(P1_val, P2_val),
        dist_file=D_val,
    )

    collate_fn = partial(pad_data, model_type=model_type)

    train_dataloader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2,
    )
    val_dataloader = data.DataLoader(
        val_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2,
    )

    return {
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
    }


def copy_training_set(
    P1_train: str,  ## Source file locations; some of which could be directories themselves
    P2_train: str,
    D_train: str,
    P1_val: str,
    P2_val: str,
    D_val: str,
    dest_dir: str,  ## Destination directory on host machine
    prefix: str,  ## copy over to dest_dir/prefix/
):
    """
    We need to copy data over to the host machine to eliminate the IO bottleneck.
    Additionally, we return the new pathnames of the files for later use.
    """
    print("Copying training set to host machine...", file=sys.stderr)
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(f"{dest_dir}/{prefix}", exist_ok=True)

    # Copy over the training set
    basenames = []
    for f in [P1_train, P2_train, D_train, P1_val, P2_val, D_val]:
        errcode = os.system(f"cp -r {f} {dest_dir}/{prefix}/")
        if errcode != 0:
            print(f"Error copying {f} to {dest_dir}/{prefix}/", file=sys.stderr)
            sys.exit(1)
        basename = os.path.basename(f)
        basenames.append(basename)
    return [os.path.join(dest_dir, prefix, basename) for basename in basenames]


def train_model(args):
    # TODO load model config from yaml
    model_config = NestedNamespace(
        yaml.load(
            open(args.model_config, "r"),
            Loader=yaml.FullLoader,
        )
    )
    dataloader_params = model_config.dataloader_params
    training_params = model_config.training_params
    optimizer_params = model_config.optimizer_params
    encoder_params = model_config.encoder_params

    P1_train, P2_train, D_train, P1_val, P2_val, D_val = copy_training_set(
        P1_train=args.P1_train,
        P2_train=args.P2_train,
        D_train=args.D_train,
        P1_val=args.P1_val,
        P2_val=args.P2_val,
        D_val=args.D_val,
        dest_dir="/cache/much8161",
        prefix=args.prefix,
    )

    loss_functions = {
        "mse": nn.MSELoss,
        "msle": MeanSquaredLogError,
        "huber": nn.SmoothL1Loss,
    }

    loss_fn = loss_functions[training_params.loss_fn]

    data = get_dataloaders(
        P1_train=P1_train,
        P2_train=P2_train,
        D_train=D_train,
        P1_val=P1_val,
        P2_val=P2_val,
        D_val=D_val,
        model_type=model_config.model_type,
        batch_size=dataloader_params.batch_size,
        n_workers=dataloader_params.num_workers,
    )

    steps_per_epoch = compute_steps_per_epoch(
        train_dataset=data["train_dataloader"].dataset,
        batch_size=dataloader_params.batch_size,
        accumulate_grad_batches=training_params.accumulate_grad_batches,
        n_devices=training_params.devices,
    )
    print(f"steps_per_epoch: {steps_per_epoch}", file=sys.stderr)

    siamese_model = SiameseModule(
        encoder_type=model_config.model_type,
        encoder_params=vars(encoder_params),
        optimizer=optim.AdamW,
        optimizer_params=vars(optimizer_params),
        # scheduler=optim.lr_scheduler.CosineAnnealingWarmRestarts,
        scheduler=optim.lr_scheduler.CosineAnnealingLR,
        scheduler_params={
            "T_max": steps_per_epoch * 200, # regular cosine annealing
            # "T_0": 17_328,  # from methods, for sgdr
            # "T_mult": 1,
            "eta_min": 1e-6,
            "verbose": True,
        },
        loss_fn=loss_fn,
    )
    # if training_params.compile:
    #     siamese_model = torch.compile(siamese_model)

    callbacks = [
        StochasticWeightAveraging(swa_lrs=optimizer_params.lr),
        EarlyStopping(
            monitor="val_loss",
            patience=training_params.early_stop_patience,
            verbose=True,
            mode="min",
        ),
        # save the last 10 checkpoints
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=f"{args.outdir}/{model_config.model_type}-{args.model_prefix}.checkpoints",
            filename=f"{model_config.model_type}-{args.model_prefix}-{{epoch:03d}}-{{val_loss:.5f}}",
            # save_top_k=5, we'll just use the wandb checkpoints for other points in training
            save_last=True,
            mode="min",
        ),
    ]

    logger = WandbLogger(
        project="gt-similarity-search",
        name=f"{model_config.model_type}-{args.model_prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        log_model="all",
    )

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=training_params.accumulate_grad_batches,
        precision=training_params.precision,
        gradient_clip_val=training_params.gradient_clip_val,
        gradient_clip_algorithm=training_params.gradient_clip_algorithm,
        max_epochs=training_params.max_epochs,
        accelerator=training_params.accelerator,
        devices=training_params.devices,
        fast_dev_run=training_params.fast_dev_run,
        enable_progress_bar=training_params.enable_progress_bar,
    )

    logger.watch(siamese_model)

    # torch.set_float32_matmul_precision("high") # if you want to use tensor cores
    trainer.fit(
        model=siamese_model,
        train_dataloaders=data["train_dataloader"],
        val_dataloaders=data["val_dataloader"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save checkpoints to",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="prefix to add to run name and checkpoint directory name",
    )
    parser.add_argument(
        "--P1_train",
        type=str,
        required=True,
        help="Path to training P1 memmap",
    )
    parser.add_argument(
        "--P2_train",
        type=str,
        required=True,
        help="Path to training P2 memmap",
    )
    parser.add_argument(
        "--P1_val",
        type=str,
        required=True,
        help="Path to validation P1 memmap",
    )
    parser.add_argument(
        "--P2_val",
        type=str,
        required=True,
        help="Path to validation P2 memmap",
    )
    parser.add_argument(
        "--D_train",
        type=str,
        required=True,
        help="Path to training distance file",
    )
    parser.add_argument(
        "--D_val",
        type=str,
        required=True,
        help="Path to validation distance file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="path to model config file",
    )
    parser.add_argument(
        "--model_prefix",
        type=str,
        required=True,
        help="name of model training run",
    )
    args = parser.parse_args()

    train_model(args)


if __name__ == "__main__":
    main()
