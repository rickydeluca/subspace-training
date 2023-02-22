import argparse

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
import pandas as pd

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
    parser.add_argument('--deterministic', type=bool, default=True,
                    help='Choose if we want the training to act deterministically. (default: True)')
    parser.add_argument('--lr', type=float, default=3e-3,
                    help='Learning rate. (default: 3e-3)')
    parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train. (default: 10)')
    parser.add_argument('--n_hidden_layers', type=int, default=1,
                    help='How many times repeat the hidden layer in the fc network. (default: 1)')
    parser.add_argument('--logs_dir', type=str, default="logs/",
                help='Path to the directory in which store the training logs. (default: "logs/")')
    return parser.parse_args()


def setup_model(args):
    """ Set-up the model using the terminal inputs."""

    # Reproducibility
    if args.deterministic:
        seed_everything(42, workers=True)   # https://en.wikipedia.org/wiki/Phrases_from_The_Hitchhiker%27s_Guide_to_the_Galaxy#The_Answer_to_the_Ultimate_Question_of_Life,_the_Universe,_and_Everything_is_42

    # Get the datamodule
    if args.dataset == "mnist":
        data_module = MNISTDataModule(data_dir=PATH_DATASETS, batch_size=BATCH_SIZE)
        input_size = 28*28
        input_channels = 1
        output_size = 10
    if args.dataset == "cifar10":
        data_module = CIFAR10DataModule(data_dir=PATH_DATASETS, batch_size=BATCH_SIZE)
        input_size = 32*32
        input_channels = 1
        output_size = 10

    # Init the model
    if args.network_type == "fc":
        model = SubspaceFCN(
            input_size=input_size,
            input_channels=input_channels,
            hidden_size=100,
            output_size=output_size,
            n_hidden_layers=2,
            subspace_dim=args.subspace_dim,
            proj_type=args.proj_type)

    if args.network_type == "lenet":
        model = SubspaceLeNet(
            input_size=input_size,
            input_channels=input_channels,
            n_feature=6,
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

    # Setup trainer
    trainer = Trainer(
        accelerator = "auto",
        devices = 1 if torch.cuda.is_available() else None,
        max_epochs = 10,
        callbacks = [TQDMProgressBar(refresh_rate=20)],
        logger = CSVLogger(save_dir=args.logs_dir),
        deterministic = args.deterministic  # Reproducibility
    )

    return trainer


if __name__ == "__main__":
    
    # Read input
    args = parse_args()

    # Setup model
    data_module, model = setup_model(args)

    # Setup trainer
    trainer = setup_trainer(args)

    # Train
    trainer.fit(model, data_module)

    # Test
    test_metrics = trainer.test(model, data_module)

    # Log test metrics
    pd.DataFrame(test_metrics[0]).to_csv(f'{args.logs_dir}/test_logs/out.csv', index=False)