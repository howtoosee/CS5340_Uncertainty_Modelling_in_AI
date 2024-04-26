import glob
from einops import rearrange
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
import torch.utils
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision.datasets import mnist, CIFAR10, CIFAR100
from torchvision.transforms import Compose, Pad
from torchvision.transforms.functional import rgb_to_grayscale

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def sample(dataset, n):
    """Randomly take n samples from the dataset"""
    assert n < len(dataset)
    indices = np.random.choice(len(dataset), n, replace=False)
    return Subset(dataset, indices)


def pad_by_2(images: torch.Tensor):
    """Pads a batch of images by 2 pixels on all sides"""
    transformation = Compose(
        [
            Pad(padding=(2, 2), fill=0, padding_mode="constant"),
        ]
    )
    images = torch.stack([transformation(image) for image in images])
    return images


class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.images = None
        self.class_labels = None
        self.ood_labels = None

    def __len__(self):
        return self.images.size()[0]

    def __getitem__(self, idx):
        # print("Images size:", self.images.size())
        # print("Class labels size:", self.class_labels.size())
        # print("Ood labels size:", self.ood_labels.size())
        image = self.images[idx].to(dtype=torch.float32)
        label = self.class_labels[idx].to(dtype=torch.int64)
        ood_label = self.ood_labels[idx].to(dtype=torch.int64)

        return image, label, ood_label

    def sample(self, n):
        assert n < len(self)
        indices = np.random.choice(len(self), n, replace=False)
        return Subset(self, indices)
    
    def to(self, device):
        self.images = self.images.to(device)
        self.class_labels = self.class_labels.to(device)
        self.ood_labels = self.ood_labels.to(device)
        return self


class Cifar10(MyDataset):
    def __init__(self, is_train, label_offset, ood_label, download_path="../downloaded_data"):
        super().__init__()
        self.is_train = is_train
        self.ood_label = ood_label
        self.images, self.class_labels, self.ood_labels = self.get_dataset(download_path)
        self.class_labels += label_offset
        self.num_classes = 10

    def get_dataset(self, download_path):
        data = CIFAR10(download_path, train=self.is_train, download=True)
        images = rearrange(torch.tensor(data.data), "b h w c -> b c h w")
        images = rgb_to_grayscale(images)
        image_labels = torch.tensor(data.targets)
        ood_labels = torch.zeros(len(images)) + self.ood_label

        return images, image_labels, ood_labels


class Cifar100(MyDataset):
    def __init__(self, is_train, label_offset, ood_label, download_path="../downloaded_data"):
        super().__init__()
        self.is_train = is_train
        self.ood_label = ood_label
        self.images, self.class_labels, self.ood_labels = self.get_dataset(download_path)
        self.class_labels += label_offset
        self.num_classes = 100

    def get_dataset(self, download_path):
        data = CIFAR100(download_path, train=self.is_train, download=True)
        images = rearrange(torch.tensor(data.data), "b h w c -> b c h w")
        images = rgb_to_grayscale(images)
        image_labels = torch.tensor(data.targets)
        ood_labels = torch.zeros(len(images)) + self.ood_label

        return images, image_labels, ood_labels


class MnistDataset(MyDataset):
    def __init__(self, is_train, label_offset, ood_label, download_path="../downloaded_data"):
        super().__init__()
        self.is_train = is_train
        self.ood_label = ood_label
        self.images, self.class_labels, self.ood_labels = self.get_dataset(download_path)
        self.class_labels += label_offset
        self.num_classes = 10

    def get_dataset(self, download_path):
        data = mnist.MNIST(download_path, train=self.is_train, download=True)
        images = rearrange(data.data, "b h w -> b 1 h w")
        assert images.shape[1:] == (1, 28, 28), f"Expected shape (b, 1, 28, 28), got {images.shape} instead"
        images = pad_by_2(images)
        image_labels = data.targets
        ood_labels = torch.zeros(len(images)) + self.ood_label

        return images, image_labels, ood_labels


class NotMnistDataset(MyDataset):
    def __init__(self, is_train, label_offset, ood_label, download_path="../downloaded_data"):
        super().__init__()
        self.is_train = is_train
        self.ood_label = ood_label
        self.images, self.class_labels, self.ood_labels = self.get_dataset(download_path)
        self.class_labels += label_offset
        self.num_classes = 10

    def get_dataset(self, download_path):
        images, labels = self.load_data()
        assert images.shape[1:] == (1, 28, 28), f"Expected shape (b, 1, 28, 28), got {images.shape} instead"
        images = pad_by_2(images)
        ood_labels = torch.zeros(len(images)) + self.ood_label
        return images, labels, ood_labels

    def load_data(self, root="/home/xihao/repository/cs5340/downloaded_data/notMNIST_small/*/*"):
        fnames = glob.glob(root)
        images = []
        targets = []
        for fname in fnames:
            splits = fname.split("/")
            label = splits[-2]
            label = ord(label.lower()) - ord("a")
            try:
                with Image.open(fname) as image:
                    image = Image.open(fname).convert("L")
            except UnidentifiedImageError as e:
                continue
            image = rearrange(torch.tensor(np.array(image)), "h w -> 1 1 h w")
            images.append(image)
            targets.append(label)
        return torch.vstack(images), torch.tensor(targets)


class FashionMnistDataset(MyDataset):
    def __init__(self, is_train, label_offset, ood_label, download_path="../downloaded_data"):
        super().__init__()
        self.is_train = is_train
        self.ood_label = ood_label
        self.images, self.class_labels, self.ood_labels = self.get_dataset(download_path)
        self.class_labels += label_offset
        self.num_classes = 10

    def get_dataset(self, download_path):
        data = mnist.FashionMNIST(download_path, train=self.is_train, download=True)
        images = rearrange(data.data, "b h w -> b 1 h w")
        assert images.shape[1:] == (1, 28, 28), f"Expected shape (b, 1, 28, 28), got {images.shape} instead"
        images = pad_by_2(images)
        image_labels = torch.tensor(data.targets)
        ood_labels = torch.zeros(len(images)) + self.ood_label

        return images, image_labels, ood_labels


"""
Functions to generate datasets
"""

PROPORTION = 0.01


def get_cifar10_train():
    return {
        "dataset": Cifar10(is_train=True, label_offset=0, ood_label=0),
        "num_classes_id": 10,
        "num_classes_ood": 0,
    }


def get_cifar10_near(is_train=False):
    id_data = Cifar10(is_train=is_train, label_offset=0, ood_label=0)
    n = int(len(id_data) * PROPORTION)
    ood_data = Cifar100(is_train=is_train, label_offset=10, ood_label=1).sample(n)
    return {
        "dataset": ConcatDataset([id_data, ood_data]),
        "num_classes_id": 10,
        "num_classes_ood": 100,
    }


def get_cifar10_far(is_train=False):
    id_data = Cifar10(is_train=is_train, label_offset=0, ood_label=0)
    n = int(len(id_data) * PROPORTION / 3)

    ood_data = ConcatDataset(
        [
            MnistDataset(is_train=is_train, label_offset=20, ood_label=1).sample(n),
            FashionMnistDataset(is_train=is_train, label_offset=30, ood_label=1).sample(n),
            NotMnistDataset(is_train=is_train, label_offset=40, ood_label=1).sample(n),
        ]
    )
    return {
        "dataset": ConcatDataset([id_data, ood_data]),
        "num_classes_id": 10,
        "num_classes_ood": 30,
    }


def get_mnist_train():
    return {
        "dataset": MnistDataset(is_train=True, label_offset=0, ood_label=0),
        "num_classes_id": 10,
        "num_classes_ood": 0,
    }


def get_mnist_near(is_train=False):
    id_data = MnistDataset(is_train=is_train, label_offset=0, ood_label=0)
    n = int(len(id_data) * PROPORTION / 2)
    ood_data = ConcatDataset(
        [
            FashionMnistDataset(is_train=is_train, label_offset=10, ood_label=1).sample(n),
            NotMnistDataset(is_train=is_train, label_offset=20, ood_label=1).sample(n),
        ]
    )

    return {
        "dataset": ConcatDataset([id_data, ood_data]),
        "num_classes_id": 10,
        "num_classes_ood": 20,
    }


def get_mnist_far1(is_train=False):
    id_data = MnistDataset(is_train=is_train, label_offset=0, ood_label=0)
    n = int(len(id_data) * PROPORTION / 2)
    ood_data = ConcatDataset(
        [
            Cifar10(is_train=is_train, label_offset=10, ood_label=1).sample(n),
            Cifar100(is_train=is_train, label_offset=20, ood_label=1).sample(n),
        ]
    )
    return {
        "dataset": ConcatDataset([id_data, ood_data]),
        "num_classes_id": 10,
        "num_classes_ood": 110,
    }


def get_mnist_far2(is_train=False):
    id_data = MnistDataset(is_train=is_train, label_offset=0, ood_label=0)
    n = int(len(id_data) * PROPORTION)
    ood_data = Cifar10(is_train=is_train, label_offset=10, ood_label=1).sample(n)
    return {
        "dataset": ConcatDataset([id_data, ood_data]),
        "num_classes_id": 10,
        "num_classes_ood": 10,
    }
