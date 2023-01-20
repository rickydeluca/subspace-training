import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

# Datamodules
from subspace_networks import SubspaceFCN, SubspaceLeNet, SubspaceResNet20
from data_modules import MNISTDataModule, CIFAR10DataModule

PATH_DATASETS = "./data/"
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


# Check the datasete we are using
dataset         = "mnist"
network_type    = "fc"
subspace_dim    = 1000
proj_type       = "sparse"
deterministic   = True

# Reproducibility
if deterministic:
    seed_everything(42, workers=True)   # https://en.wikipedia.org/wiki/Phrases_from_The_Hitchhiker%27s_Guide_to_the_Galaxy#The_Answer_to_the_Ultimate_Question_of_Life,_the_Universe,_and_Everything_is_42


# Get the datamodule
if dataset == "mnist":
    data_module = MNISTDataModule(data_dir=PATH_DATASETS, batch_size=BATCH_SIZE)
    input_size = 28*28
    input_channels = 1
    output_size = 10
if dataset == "cifar10":
    data_module = CIFAR10DataModule(data_dir=PATH_DATASETS, batch_size=BATCH_SIZE)
    input_size = 32*32
    input_channels = 1
    output_size = 10
# if dataset == "imagenet":
#     data_module = ImagenetDataModule(PATH_DATASETS+"/ILSVRC2017_DET_test_new/ILSVRC")
#     input_size = 224*224
#     input_channels = 3
#     output_size = 1000


# Init the model
if network_type == "fc":
    model = SubspaceFCN(    input_size=input_size,
                            input_channels=input_size,
                            n_hidden=100,
                            output_size=output_size,
                            subspace_dim=subspace_dim,
                            proj_type=proj_type)

if network_type == "lenet":
    model = SubspaceLeNet(  input_size=input_size,
                            input_channels=input_channels,
                            n_feature=6,
                            output_size=output_size,
                            subspace_dim=subspace_dim,
                            proj_type=proj_type)

if network_type == "resnet20":
    model = SubspaceResNet20(   input_size=input_size,
                                input_channels=input_channels,
                                output_size=output_size,
                                subspace_dim=subspace_dim,
                                proj_type=proj_type)


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