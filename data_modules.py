import multiprocessing
import random

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST

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
            data_new=torch.zeros((data.shape))
            for i, img in enumerate(data):
                data_new[i] = img.flatten()[self.perm].reshape((data.shape))
            
            return data_new
        
        else:
            return data


# =========
#   MNIST
# =========

class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = PATH_DATASETS, 
        batch_size: int = BATCH_SIZE, 
        shuffle_pixels=False, 
        shuffle_labels=False, 
        deterministic=True, seed=42
        ):
        
        super().__init__()

        # Reproducibility
        if deterministic == True:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Class attributes
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle_labels = shuffle_labels
        
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
            
            if self.shuffle_labels:
                # Download dataset twice, one time for train and one for validation.
                # In this way we can shuffle the train labels without altering
                # the validation ones.
                mnist_train_full = MNIST(self.data_dir, train=True, transform=self.transform)
                mnist_val_full = MNIST(self.data_dir, train=True, transform=self.transform)
               
                # Get original labels
                original_train_labels = mnist_train_full.targets

                # Generate random labels for train
                random_train_labels = torch.randint(0, 9, (len(original_train_labels),))

                # Substitute original train labels with the random ones
                mnist_train_full.targets = random_train_labels

                # Get the right size for train and valdation
                shuffled_indices = torch.randperm(len(mnist_train_full))
                train_indices, val_indices = shuffled_indices[:55000], shuffled_indices[55000:]

                # Build final train and validation dataset
                self.mnist_train = Subset(mnist_train_full, train_indices)
                self.mnist_val = Subset(mnist_val_full, val_indices)
            
            else:
                # Download and transform
                mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)

                # Split in train and validation
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
    def __init__(
            self, 
            data_dir: str = PATH_DATASETS, 
            batch_size: int = BATCH_SIZE, 
            shuffle_pixels=False,
            shuffle_labels=False,
            deterministic=True,
            seed=42
        ):
        
        super().__init__()

        # Reproducibility
        if deterministic == True:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Class attributes
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle_labels = shuffle_labels

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
            
            if self.shuffle_labels:
                # Download dataset twice, one time for train and one for validation.
                # In this way we can shuffle the train labels without altering
                # the validation ones.
                cifar10_train_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
                cifar10_val_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
               
                # Get original labels
                original_train_labels = cifar10_train_full.targets

                # Generate random labels for train
                random_train_labels = torch.randint(0, 9, (len(original_train_labels),))

                # Substitute original train labels with the random ones
                cifar10_train_full.targets = random_train_labels

                # Get the right size for train and valdation
                shuffled_indices = torch.randperm(len(cifar10_train_full))
                train_indices, val_indices = shuffled_indices[:45000], shuffled_indices[5000:]

                # Build final train and validation dataset
                self.mnist_train = Subset(cifar10_train_full, train_indices)
                self.mnist_val = Subset(cifar10_val_full, val_indices)
            
            else:
                # Download and transform
                cifar10_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
                
                # Split train and validation
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