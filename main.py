import argparse

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
# import pandas as pd

from data_modules import CIFAR10DataModule, MNISTDataModule

# Datamodules
from multiple_size_networks import SubspaceFCN, SubspaceLeNet, SubspaceResNet20

PATH_DATASETS = "./data/"
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


def parse_args():
    '''
    Parse the terminal arguments.
    '''
    parser = argparse.ArgumentParser(description='Set the hyperparameters for the training.')
    parser.add_argument('--dataset', type=str, default="mnist",
                    help='The dataset to use. Choose between: "mnist" or "cifar10". (default: mnist)')
    parser.add_argument('--network_type', type=str, default="fc",
                    help='The neural network to use. Choose between: "fc" (fully connected), "lenet", "resnet20". (default: "fc")')
    parser.add_argument('--subspace_dim', type=int, default=None,
                    help='Number parameters to use in the subspace training. If None do not use subspace trainig. (default: None)')
    parser.add_argument('--proj_type', type=str, default="dense",
                    help='The projection matrix to use. Choose between: "dense", "sparse" or "fastfood". If subspace dimension is None, ignore this parameter. (default: "dense")')
    parser.add_argument('--deterministic', type=int, default=1,
                    help='Choose if we want the training to act deterministically. (default: 1)')
    parser.add_argument('--shuffle_pixels', type=int, default=0,
                    help='If 1 shuffle the pixels in the input images. (default: 0)')
    parser.add_argument('--lr', type=float, default=3e-3,
                    help='Learning rate. (default: 3e-3)')
    parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train. (default: 10)')
    parser.add_argument('--hidden_width', type=int, default=100,
                    help='Size of hidden layers in fc network. (default: 100)')
    parser.add_argument('--hidden_depth', type=int, default=1,
                    help='How many hidden layer in the fc network. (default: 1)')
    parser.add_argument('--n_feature', type=int, default=6,
                    help='Number of features for CNNs. (default: 6)')
    parser.add_argument('--logs_dir', type=str, default="logs/",
                help='Path to the directory in which store the training logs. (default: "logs/")')
    return parser.parse_args()


def setup_model(args):
    """ Set-up the model using the terminal inputs."""

    _deterministic = True if args.deterministic==1 else False
    _shuffle_pixels = True if args.shuffle_pixels==1 else False

    # Reproducibility
    if _deterministic:
        seed_everything(42, workers=True)

    # Get the datamodule
    if args.dataset == "mnist":
        data_module = MNISTDataModule(
            data_dir=PATH_DATASETS,
            batch_size=BATCH_SIZE,
            shuffle_pixels=_shuffle_pixels,
            deterministic=_deterministic,
            seed=42)
        input_size = 28*28
        input_channels = 1
        output_size = 10

    if args.dataset == "cifar10":
        data_module = CIFAR10DataModule(
            data_dir=PATH_DATASETS,
            batch_size=BATCH_SIZE,
            shuffle_pixels=_shuffle_pixels)
        input_size = 32*32
        input_channels = 1
        output_size = 10

    # Init the model
    if args.network_type == "fc":
        model = SubspaceFCN(
            input_size=input_size,
            input_channels=input_channels,
            hidden_width=args.hidden_width,
            output_size=output_size,
            hidden_depth=args.hidden_depth,
            subspace_dim=args.subspace_dim,
            proj_type=args.proj_type)

    if args.network_type == "lenet":
        model = SubspaceLeNet(
            input_size=input_size,
            input_channels=input_channels,
            n_feature=args.n_feature,
            output_size=output_size,
            subspace_dim=args.subspace_dim,
            proj_type=args.proj_type)

    if args.network_type == "resnet20":
        model = SubspaceResNet20(
            input_size=input_size,
            input_channels=input_channels,
            output_size=output_size,
            subspace_dim=args.subspace_dim,
            proj_type=args.proj_type)
    
    return data_module, model


def setup_trainer(args):
    """Set-up the trainer using the terminal inputs."""

    _deterministic = True if args.deterministic==1 else False
    _shuffle_pixels = True if args.shuffle_pixels==1 else False

    # Setup trainer
    trainer = Trainer(
        accelerator = "auto",
        devices = 1 if torch.cuda.is_available() else None,
        max_epochs = args.epochs,
        callbacks = [TQDMProgressBar(refresh_rate=20)],
        logger = CSVLogger(save_dir=args.logs_dir),
        deterministic = _deterministic  # Reproducibility
    )

    return trainer


if __name__ == "__main__":
    
    # Read input
    args = parse_args()

    # Setup model
    data_module, model = setup_model(args)

    # Setup trainer
    trainer = setup_trainer(args)

    # Train and log
    trainer.fit(model, data_module)

    # Test and log
    test_metrics = trainer.test(model, data_module)