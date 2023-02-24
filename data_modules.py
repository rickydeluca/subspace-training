import multiprocessing
import sys

import numpy as np
import plotly.express as px
import torch
import torchvision.transforms.functional as F
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST

# from pl_bolts.datamodules import ImagenetDataModule

PATH_DATASETS = "./data/"
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


# =========
#   UTILS
# =========

class PixelShuffleTransform(object):
    """
    Define a permutation of the pixel in an image.
    This class will be used as a transformations in the DataModule.
    It applies the permutation only if the 'shuffle' variable is set
    to True, otherwise it returns the original image.
    """
    def __init__(self, perm=None):
        self.perm = perm

    def __call__(self, data):
        
        if self.perm is not None:
            # print("Shuffling pixels!!!")
            data_new=torch.zeros((data.shape))
            for i, img in enumerate(data):
                data_new[i] = img.flatten()[self.perm].reshape((1,28,28))
            
            return data_new
        
        else:
            return data
            
        


# =========
#   MNIST
# =========

class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = PATH_DATASETS, batch_size: int = BATCH_SIZE, shuffle_pixels=False, deterministic=True, seed=42):
        
        super().__init__()

        # Reproducibility
        if deterministic == True:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Class attributes
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # Define pixel permutation to use eventually
        if shuffle_pixels==True:
            perm = torch.randperm(28*28*1)
        else:
            perm = None

        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            PixelShuffleTransform(perm=perm)    # only if perm is not None
        ])

    def prepare_data(self):
        # Download (if not already in the root folder)
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            # Download and transform
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            
            # Split train and validation
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count(), shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count())


# ===========
#   CIFAR10
# ===========

class CIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir: str = PATH_DATASETS, batch_size: int = BATCH_SIZE, shuffle_pixels=False):
        
        super().__init__()
        
        # Class attributes
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Define pixel permutation to use eventually
        if shuffle_pixels==True:
            perm = torch.randperm(32*32*1)
        else:
            perm = None

        # Define transform
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
            PixelShuffleTransform(perm=perm)    # only if perm is not None
        ])

    def prepare_data(self):
        # Download (if not already in the root folder)
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar10_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count(), shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count())