import glob
from einops import rearrange
import numpy as np
import pandas as pd
from PIL import Image
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
    return torch.stack([transformation(image) for image in images])


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
        label = self.class_labels[idx].to(dtype=torch.int32)
        ood_label = self.ood_labels[idx].to(dtype=torch.int32)

        return image, label, ood_label

    def sample(self, n):
        assert n < len(self)
        indices = np.random.choice(len(self), n, replace=False)
        return Subset(self, indices)


class Cifar10(MyDataset):
    def __init__(self, label_offset, is_train, ood_label=0, download_path="../downloaded_data"):
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
        ood_labels = torch.zeros(len(images), dtype=torch.int) + self.ood_label

        return images, image_labels, ood_labels


class Cifar100(MyDataset):
    def __init__(self, label_offset, is_train=False, ood_label=1, download_path="../downloaded_data"):
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
        ood_labels = torch.zeros(len(images), dtype=torch.int) + self.ood_label

        return images, image_labels, ood_labels


class MnistDataset(MyDataset):
    def __init__(self, label_offset, is_train=False, ood_label=0, download_path="../downloaded_data"):
        super().__init__()
        self.is_train = is_train
        self.ood_label = ood_label
        self.images, self.class_labels, self.ood_labels = self.get_dataset(download_path)
        self.class_labels += label_offset
        self.num_classes = 10

    def get_dataset(self, download_path):
        data = mnist.MNIST(download_path, train=self.is_train, download=True)
        images = torch.tensor(data.data)
        assert images.shape[1:] == (28, 28)
        images = pad_by_2(images)
        image_labels = torch.tensor(data.targets)
        ood_labels = torch.zeros(len(images), dtype=torch.int) + self.ood_label

        return images, image_labels, ood_labels


class NotMnistDataset(MyDataset):
    def __init__(self, label_offset, is_train=False, ood_label=1, download_path="../downloaded_data"):
        super().__init__()
        self.is_train = is_train
        self.ood_label = ood_label
        self.images, self.class_labels, self.ood_labels = self.get_dataset(download_path)
        self.class_labels += label_offset
        self.num_classes = 10

    def get_dataset(self, download_path):
        data = self.load_notmnist_data()
        images = torch.tensor(data.data)
        assert images.shape[1:] == (28, 28)
        images = pad_by_2(images)
        image_labels = torch.tensor(data.targets)
        ood_labels = torch.zeros(len(images), dtype=torch.int) + self.ood_label

        return images, image_labels, ood_labels

    def load_notmnist_data(root="../downloaded_data/notMNIST_small/*/*"):
        fnames = glob.glob(root)
        images = []
        targets = []
        for fname in fnames:
            splits = fname.split("/")
            label = splits[-2]
            label = ord(label.lower()) - ord("a")
            image = Image.open(fname)
            images.append(image)
            targets.append(label)
        return {"images": images, "labels": targets}


class FashionMnistDataset(MyDataset):
    def __init__(self, label_offset, is_train=False, ood_label=1, download_path="../downloaded_data"):
        super().__init__()
        self.is_train = is_train
        self.ood_label = ood_label
        self.images, self.class_labels, self.ood_labels = self.get_dataset(download_path)
        self.class_labels += label_offset
        self.num_classes = 10

    def get_dataset(self, download_path):
        data = mnist.FashionMNIST(download_path, train=self.is_train, download=True)
        images = torch.tensor(data.data)
        assert images.shape[1:] == (28, 28)
        images = pad_by_2(images)
        image_labels = torch.tensor(data.targets)
        ood_labels = torch.zeros(len(images), dtype=torch.int) + self.ood_label

        return images, image_labels, ood_labels


"""
Functions to generate datasets
"""


def get_cifar10_train():
    return {
        "dataset": Cifar10(0, is_train=True),
        "num_classes_id": 10,
        "num_classes_ood": 0,
    }


def get_cifar10_near(is_train=False):
    id_data = Cifar10(0, is_train=is_train)
    n = int(len(id_data) / 10)
    ood_data = Cifar100(10, is_train=is_train).sample(n)
    return {
        "dataset": torch.utils.data.ConcatDataset([id_data, ood_data]),
        "num_classes_id": 10,
        "num_classes_ood": 100,
    }


def get_cifar10_far(is_train=False):
    id_data = (Cifar10(0, is_train=is_train),)
    n = int(len(id_data) / 10 / 3)

    ood_data = ConcatDataset(
        [
            MnistDataset(20, is_train=is_train).sample(n),
            FashionMnistDataset(30, is_train=is_train).sample(n),
            NotMnistDataset(40, is_train=is_train).sample(n),
        ]
    )
    return {
        "dataset": ConcatDataset(id_data, ood_data),
        "num_classes_id": 10,
        "num_classes_ood": 30,
    }


def get_mnist_train():
    return {
        "dataset": MnistDataset(0, is_train=True),
        "num_classes_id": 10,
        "num_classes_ood": 0,
    }


def get_mnist_near(is_train=False):
    id_data = (MnistDataset(0, is_train),)
    n = int(len(id_data) / 10 / 2)
    ood_data = ConcatDataset(
        [
            FashionMnistDataset(10, is_train).sample(n),
            NotMnistDataset(20, is_train).sample(n),
        ]
    )

    return {
        "dataset": torch.utils.data.ConcatDataset([id_data, ood_data]),
        "num_classes_id": 10,
        "num_classes_ood": 20,
    }


def get_mnist_far1(is_train=False):
    id_data = (MnistDataset(0, is_train=is_train),)
    n = int(len(id_data) / 10 / 2)
    ood_data = ConcatDataset(
        [
            Cifar10(10, is_train=is_train).sample(n),
            Cifar100(20, is_train=is_train).sample(n),
        ]
    )
    return {
        "dataset": torch.utils.data.ConcatDataset([id_data, ood_data]),
        "num_classes_id": 10,
        "num_classes_ood": 110,
    }


def get_mnist_far2(is_train=False):
    id_data = (MnistDataset(0, is_train=is_train),)
    n = int(len(id_data) / 10)
    ood_data = Cifar10(10, is_train=is_train).sample(n)
    return {
        "dataset": torch.utils.data.ConcatDataset([id_data, ood_data]),
        "num_classes_id": 10,
        "num_classes_ood": 10,
    }
