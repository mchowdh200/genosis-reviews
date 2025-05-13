import argparse
import os
from functools import partial
from pprint import pprint
from types import SimpleNamespace
from typing import Any

import pytorch_lightning as pl
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from data_utils.gt_datasets import (GTInferenceDataset, GTInferenceWriter,
                                    pad_data)
from models.encoder import SiameseModule


def NestedNamespace(x: dict):
    """Convert a nested dict to a nested namespace"""
    return SimpleNamespace(
        **{k: NestedNamespace(v) if isinstance(v, dict) else v for k, v in x.items()}
    )


def get_encoder_params(model_config: str | None) -> tuple[SimpleNamespace, ...]:
    if model_config is None:
        raise ValueError("No model config provided")
    else:
        config = NestedNamespace(
            yaml.load(
                open(args.model_config, "r"),
                Loader=yaml.FullLoader,
            )
        )
    dataloader_params = config.dataloader_params
    training_params = config.training_params
    optimizer_params = config.optimizer_params
    encoder_params = config.encoder_params
    return dataloader_params, training_params, optimizer_params, encoder_params


def load_siamese_encoder(
    path: str | None = None, model_config: str | None = None
) -> pl.LightningModule:
    """
    Load the encoder from checkpoint.  If no path is provided, we will return an untrained encoder.
    """
    if path is None:
        _, _, optimizer_params, encoder_params = get_encoder_params(model_config)
        model = SiameseModule(
            encoder_type="conv1d",
            encoder_params=vars(encoder_params),
            # we wont be using this stuff, to test the untrained encoder, but
            # we need to pass it in to the constructor
            optimizer=optim.AdamW,
            optimizer_params=vars(optimizer_params),
            scheduler=optim.lr_scheduler.CosineAnnealingWarmRestarts,
            scheduler_params={
                "T_0": 17_328,  # from methods
                "T_mult": 1,
                "eta_min": 1e-6,
                "verbose": True,
            },
            loss_fn=torch.nn.MSELoss,
        )  # TODO
        encoder = model.encoder
    else:
        model = SiameseModule.load_from_checkpoint(
            path, map_location=torch.device("cpu")  # is this needed?
        )
        # model = SiameseModule.load_from_checkpoint(path)
        encoder = model.encoder
    return encoder


def encode_samples(
    *,
    encoder: pl.LightningModule,
    batch_size: int,
    output: str,
    files: list[str],
    gpu: bool = False,
    num_workers: int = 0,
):
    """
    Encode a dataset of samples using the provided encoder.
    :param encoder: The encoder model for inference
    :param batch_size: The batch size for inference
    :param output: The output file to write the encoded samples to
    :param files: The files to encode
    :param gpu: Whether to use the GPU
    :param num_workers: The number of workers to use for data loading
    """
    pprint(files)

    dataset = GTInferenceDataset(files=files)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=partial(pad_data, model_type="conv1d_inference"),
    )
    writer = GTInferenceWriter(
        output=output,
        write_interval="batch",
    )
    trainer = pl.Trainer(
        inference_mode=True,
        callbacks=[writer],
        accelerator="cuda" if gpu else "cpu",
        devices=1 if gpu else 1,
    )
    trainer.predict(encoder, dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        type=str,
        required=False,
        help="Path to the encoder checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="output file",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        required=True,
        help="Files to encode",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Use GPU",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to the model config file",
    )

    args = parser.parse_args()
    encoder = load_siamese_encoder(args.encoder, args.model_config)
    encode_samples(
        encoder=encoder,
        batch_size=args.batch_size,
        output=args.output,
        files=args.files,
        gpu=args.gpu,
        num_workers=args.num_workers,
    )
