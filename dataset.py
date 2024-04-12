from typing import Optional
import glob

import torch
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CACHED_DIR = "cached_data"


def load_notmnist_data(root: Optional[str] = "downloaded_data/notMNIST_small/*/*"):
    fnames = glob.glob(root)
    data = []
    for fname in fnames:
        splits = fname.split("/")
        label = splits[-2]
        label = ord(label.lower()) - ord('a')
        data.append((fname, label))
    train, test = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)
    return train, test


class BaseDataset(Dataset):
    def __init__(self, is_train: bool) -> None:
        super().__init__()
        self.is_train = is_train
        self.dataset = self.load_dataset()

    def load_dataset(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset)

    def get_image(self, idx):
        image, _ = self.dataset[idx]
        return image

    def get_label(self, idx):
        _, label = self.dataset[idx]
        return label

    def __getitem__(self, idx):
        image = self.get_image(idx)
        label = self.get_label(idx)
        return image, label


class CustomDataset(BaseDataset):
    def __init__(self, dataset_name: str, is_train: bool, offset: Optional[int] = 0) -> None:
        self.dataset_name = dataset_name
        self.offset = offset
        super().__init__(is_train=is_train)

    def load_dataset(self):
        train = DATASETS[self.dataset_name](
            root = CACHED_DIR,
            train = self.is_train,
            download = True
        )
        return train

    def get_label(self, idx):
        _, label = self.dataset[idx]
        return label + self.offset


class NOTMINST(Dataset):
    def __init__(self, root: str, train: bool, download: Optional[bool]=True):
        self.root = root
        self.is_train = train
        self.download = download

        self.dataset = self.load_dataset()
        
    def load_dataset(self):
        notmnist_train, notmnist_test = load_notmnist_data()
        if self.is_train:
            return notmnist_train
        else:
            return notmnist_test

    def __len__(self):
        return len(self.dataset)

    def get_image(self, idx):
        image_fname, _ = self.dataset[idx]
        image = Image.open(image_fname)
        return image

    def get_label(self, idx):
        _, label = self.dataset[idx]
        return label

    def __getitem__(self, idx):
        image = self.get_image(idx)
        label = self.get_label(idx)
        return image, label


DATASETS = {
    "cifar10": torchvision.datasets.CIFAR10,
    "cifar100": torchvision.datasets.CIFAR100,
    "mnist": torchvision.datasets.MNIST,
    "fashionmnist": torchvision.datasets.FashionMNIST,
    "notmnist": NOTMINST,
}


class CIFAR10_NEAR_ODD_Dataset(BaseDataset):
    """
    Labels 0-9 are in-distribution data from CIFAR10
    Labels 10-109 are near out-of-distribution data from CIFAR100
    """
    def __init__(self, is_train: bool, transform: Optional[bool]=None) -> None:
        self.is_train = is_train
        self.cifar10_nlabels = 10
        self.cifar100_nlabels = 100
        self.transform = transform
        super().__init__(is_train=is_train)

    def load_dataset(self):
        offset = 0
        cifar10_train = CustomDataset(
            dataset_name="cifar10",
            is_train=self.is_train,
            offset=offset
        )
        offset += self.cifar10_nlabels

        cifar100_train = CustomDataset(
            dataset_name="cifar100",
            is_train=self.is_train,
            offset=offset,
        )
        offset += self.cifar100_nlabels

        return torch.utils.data.ConcatDataset([cifar10_train, cifar100_train])

    def get_image(self, idx):
        image, _ = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = np.uint8(image)
        return image
    

class CIFAR10_FAR_ODD_Dataset(BaseDataset):
    """
    Labels 0-9 are in-distribution data from CIFAR10
    Labels 10-19 are far out-of-distribution data from MNIST
    Labels 20-29 are far out-of-distribution data from FashionMNIST
    Labels 30-39 are far out-of-distribution data from NOTMNIST
    """
    def __init__(self, is_train: bool, transform: Optional[bool]=None) -> None:
        self.is_train = is_train
        self.cifar10_nlabels = 10
        self.mnist_nlabels = 10
        self.fashion_nlabels = 10
        self.notmnist_nlabels = 10
        self.transform = transform
        super().__init__(is_train=is_train)

    def load_dataset(self):
        offset = 0

        cifar10_train = CustomDataset(
            dataset_name="cifar10",
            is_train=self.is_train,
            offset=offset,
        )
        offset += self.cifar10_nlabels

        mnist_train = CustomDataset(
            dataset_name="mnist",
            is_train=self.is_train,
            offset=offset,
        )
        offset += self.mnist_nlabels

        fashionmnist_train = CustomDataset(
            dataset_name="fashionmnist",
            is_train=self.is_train,
            offset=offset,
        )
        offset += self.fashion_nlabels

        notmnist_train = CustomDataset(
            dataset_name="notmnist",
            is_train=self.is_train,
            offset=offset,
        )
        offset += self.notmnist_nlabels

        return torch.utils.data.ConcatDataset([cifar10_train, mnist_train, fashionmnist_train, notmnist_train])

    def get_image(self, idx):
        image, _ = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = np.uint8(image)
        return image