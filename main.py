import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from subspace_networks import SubspaceLeNetCIFAR10, SubspaceLeNetMNIST, SubspaceFcCifar10, SubspaceFcMnist
from data_modules import MNISTDataModule, CIFAR10DataModule

PATH_DATASETS = "./data/"
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


# Check the datasete we are using
dataset         = "cifar10"
network_type    = "lenet"
subspace_dim    = None
proj_type       = "sparse"
deterministic   = True

# Reproducibility
if deterministic:
    seed_everything(42, workers=True)   # https://en.wikipedia.org/wiki/Phrases_from_The_Hitchhiker%27s_Guide_to_the_Galaxy#The_Answer_to_the_Ultimate_Question_of_Life,_the_Universe,_and_Everything_is_42


# Data module
if dataset == "mnist":
    data_module = MNISTDataModule(data_dir=PATH_DATASETS, batch_size=BATCH_SIZE)
if dataset == "cifar10":
    data_module = CIFAR10DataModule(data_dir=PATH_DATASETS, batch_size=BATCH_SIZE)


# Init the model
if dataset == "mnist":
    if network_type == "fc":
        model = SubspaceFcMnist(subspace_dim=subspace_dim, proj_type=proj_type)
    if network_type == "lenet":
        model = SubspaceLeNetMNIST(subspace_dim=subspace_dim, proj_type=proj_type)

if dataset == "cifar10":
    if network_type == "fc":
        model = SubspaceFcCifar10(subspace_dim=subspace_dim, proj_type=proj_type)
    if network_type == "lenet":
        model = SubspaceLeNetCIFAR10(subspace_dim=subspace_dim, proj_type=proj_type)

# Setup trainer
trainer = Trainer(
    accelerator = "auto",
    devices = 1 if torch.cuda.is_available() else None,
    max_epochs = 10,
    callbacks = [TQDMProgressBar(refresh_rate=20)],
    logger = CSVLogger(save_dir="logs/"),
    deterministic = True if deterministic else False  # Reproducibility
)

# Train the model
trainer.fit(model, data_module)

# Test
trainer.test(model, data_module)