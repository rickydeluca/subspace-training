import argparse
import time
from datetime import timedelta

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar

# Datamodules
from data_modules import CIFAR10DataModule, MNISTDataModule
# Log utilitis
from log_utils import (CustomCSVLogger, ForwardBackwardTimingCallback,
                       save_results)
# Subspace networks
from subspace_networks import SubspaceFCN, SubspaceLeNet, SubspaceResNet20

PATH_DATASETS = "./data/"
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


def parse_args():
    '''
    Parse the terminal arguments.
    '''
    parser = argparse.ArgumentParser(description='Set the hyperparameters for the training.')
    parser.add_argument('--dataset', type=str, default="mnist",
                    help='The dataset to use. Choose between: "mnist" or "cifar10". (default: mnist)')
    parser.add_argument('--network', type=str, default="fc",
                    help='The neural network to use. Choose between: "fc", "lenet", "resnet20". (default: "fc")')
    parser.add_argument('--subspace_dim', type=int, default=None,
                    help='Number parameters to use in the subspace training. If None do not use subspace trainig. (default: None)')
    parser.add_argument('--proj', type=str, default="dense",
                    help='The projection matrix to use. Choose between: "dense", "sparse" or "fastfood". If subspace dimension is None, ignore this parameter. (default: "dense")')
    parser.add_argument('--deterministic', type=int, default=1,
                    help='Choose if we want the training to act deterministically. (default: 1)')
    parser.add_argument('--shuffle_pixels', type=int, default=0,
                    help='If 1 shuffle the pixels in the input images. (default: 0)')
    parser.add_argument('--shuffle_labels', type=int, default=0,
                    help='If 1 shuffle the labels in the training dataset. (default: 0)')
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
    parser.add_argument('--res_dir', type=str, default="results/",
                help='Path to the directory in which store the test metrics. (default: "results/")')
    parser.add_argument('--test', type=str, default=None,
                    help='Which type of data we are collecting: "subspace" for subspace dim vs test accuracy; "baseline" for num of params VS baseline value; "time" for forward+backword pass time. (default: None)')
   
    
    return parser.parse_args()

def read_input(args):
    """
    Given the terminal inputs, create a dictionary of hyperparameters.
    """

    hyperparams = {
        "dataset":          args.dataset,
        "network_type":     args.network,
        "subspace_dim":     args.subspace_dim,
        "proj_type":        args.proj,
        "deterministic":    True if args.deterministic==1 else False,
        "shuffle_pixels":   True if args.shuffle_pixels==1 else False,
        "shuffle_labels":   True if args.shuffle_labels==1 else False,
        "lr":               args.lr,
        "epochs":           args.epochs,
        "logs_dir":         args.logs_dir,
        "res_dir":          args.res_dir,
        "test":             args.test,
        "hidden_width":     args.hidden_width,
        "hidden_depth":     args.hidden_depth,
        "n_feature":        args.n_feature
    }

    return hyperparams

    
def setup_model(hyperparams):
    """ Set-up the model using the given hyperparameters."""

    # Reproducibility
    if hyperparams["deterministic"]:
        seed_everything(42, workers=True)

    # Get the datamodule
    if hyperparams["dataset"] == "mnist":
        data_module = MNISTDataModule(
            data_dir=PATH_DATASETS,
            batch_size=BATCH_SIZE,
            shuffle_pixels=hyperparams["shuffle_pixels"],
            shuffle_labels=hyperparams["shuffle_labels"],
            deterministic=hyperparams["deterministic"],
            seed=42)
        input_size = 28*28
        input_channels = 1
        output_size = 10

    if hyperparams["dataset"] == "cifar10":
        data_module = CIFAR10DataModule(
            data_dir=PATH_DATASETS,
            batch_size=BATCH_SIZE,
            shuffle_pixels=hyperparams["shuffle_pixels"],
            shuffle_labels=hyperparams["shuffle_labels"],
            deterministic= hyperparams["deterministic"],
            seed=42)
        input_size = 32*32
        input_channels = 1
        output_size = 10

    # Init the model

    if hyperparams["network_type"] == "fc":
        model = SubspaceFCN(
            input_size=input_size,
            input_channels=input_channels,
            hidden_width=hyperparams["hidden_width"],
            output_size=output_size,
            hidden_depth=hyperparams["hidden_depth"],
            subspace_dim=hyperparams["subspace_dim"],
            proj_type=hyperparams["proj_type"])

    if hyperparams["network_type"] == "lenet":
        model = SubspaceLeNet(
            input_size=input_size,
            input_channels=input_channels,
            n_feature=hyperparams["n_feature"],
            output_size=output_size,
            subspace_dim=hyperparams["subspace_dim"],
            proj_type=hyperparams["proj_type"])

    if hyperparams["network_type"]  == "resnet20":
        model = SubspaceResNet20(
            input_size=input_size,
            input_channels=input_channels,
            output_size=output_size,
            subspace_dim=hyperparams["subspace_dim"],
            proj_type=hyperparams["proj_type"])
    
    return data_module, model


def setup_trainer(hyperparams):
    """Set-up the trainer using the given hyperparameters."""
    
    # Instantiate the custom CSV logger to explicit the settings of the experiment
    custom_logger = CustomCSVLogger(hyperparams, save_dir=hyperparams["logs_dir"])
    
    timing_callback = ForwardBackwardTimingCallback()
    
    # Time average forward + bacward pass if requested
    # computed over 3 batches
    if hyperparams['test'] == 'time':
        # Setup trainer
        trainer = Trainer(
            accelerator = "auto",
            devices = 1 if torch.cuda.is_available() else None,
            max_epochs = 1,
            limit_train_batches=3,
            callbacks = [TQDMProgressBar(refresh_rate=20), timing_callback],
            logger = custom_logger,
            deterministic = hyperparams["deterministic"],  # Reproducibility
            # Reduce number of steps if we are timing the forward/backward pass
            # max_steps=1 if hyperparams['test'] == 'time' else None,
            # max_time=timedelta(minutes=1) if hyperparams['test'] == 'time' else None, 
        )
    
    else:
        trainer = Trainer(
            accelerator = "auto",
            devices = 1 if torch.cuda.is_available() else None,
            max_epochs = hyperparams["epochs"],
            callbacks = [TQDMProgressBar(refresh_rate=20)],
            logger = custom_logger,
            deterministic = hyperparams["deterministic"],  # Reproducibility
        )

    return trainer, timing_callback


if __name__ == "__main__":
    
    # Read input
    args = parse_args()
    hyperparams = read_input(args)

    # # Setup model (with time limit)
    # model = None
    # time_limit_minutes = 1
    # start_time = time.time()
    # while model is None and (time.time() - start_time) < time_limit_minutes*60:
    #     try:
    #         data_module, model = setup_model(hyperparams)
    #     except:
    #         raise Exception("Error in model initialization")
    # if model is None:
    #     raise Exception("Could not initialize model in time limit")

    # Setup model
    data_module, model = setup_model(hyperparams)
    
    # Setup trainer
    trainer, timing_callback = setup_trainer(hyperparams)

    # Train and log
    train_results = trainer.fit(model, data_module)

    # Test and log
    test_metrics = trainer.test(model, data_module)

    # Save test metrics
    if hyperparams['test'] is not None:
        output = save_results(model, test_metrics[0], timing_callback, hyperparams)
        print(output)