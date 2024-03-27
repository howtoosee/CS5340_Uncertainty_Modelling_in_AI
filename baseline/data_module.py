import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import mnist


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

        return self.images[idx], self.class_labels[idx], self.ood_labels[idx]


    @classmethod
    def get_ood_labels(cls, size, class_label=-1, ood_label=1):
        class_labels = torch.zeros(size, dtype=torch.long) + class_label
        ood_labels = torch.zeros(size, dtype=torch.long) + ood_label
        return class_labels, ood_labels


    @classmethod
    def concat_datasets(cls, images, class_labels, ood_labels):
        images = torch.cat(images, dim=0)
        class_labels = torch.cat(class_labels, dim=0)
        ood_labels = torch.cat(ood_labels, dim=0)
        return images, class_labels, ood_labels


    @classmethod
    def shuffle(cls, images, class_labels, ood_labels):
        indices = torch.randperm(images.size()[0])
        images = images[indices]
        class_labels = class_labels[indices]
        ood_labels = ood_labels[indices]
        return images, class_labels, ood_labels


class MnistOodRandomDataset(MyDataset):
    def __init__(self):
        super(MnistOodRandomDataset, self).__init__()
        self.images, self.class_labels, self.ood_labels = self.get_dataset(60000)


    def get_dataset(self, num_random_ood=100):
        mnist_train = mnist.MNIST('data', train=True, download=True)
        mnist_images = mnist_train.data
        mnist_image_labels = mnist_train.targets
        mnist_ood_labels = torch.zeros(len(mnist_image_labels), dtype=torch.long)

        rand_images = torch.rand(num_random_ood, 28, 28)
        random_image_labels, random_ood_labels = self.get_ood_labels(num_random_ood)

        images, class_labels, ood_labels = self.concat_datasets(
            [mnist_images, rand_images],
            [mnist_image_labels, random_image_labels],
            [mnist_ood_labels, random_ood_labels]
        )

        images, class_labels, ood_labels = self.shuffle(images, class_labels, ood_labels)

        return images, class_labels, ood_labels


class MnistOodFashionMnistDataset(MyDataset):
    def __init__(self):
        super(MnistOodFashionMnistDataset, self).__init__()
        self.images, self.class_labels, self.ood_labels = self.get_dataset(100)


    def filter_target_class(self, images, class_labels, target_class=-1):
        if target_class < 0:
            return images, class_labels

        indices = torch.nonzero(class_labels == target_class).squeeze()
        return images[indices], class_labels[indices]


    def get_dataset(self, num_random_ood=100):
        mnist_data = mnist.MNIST('data', train=True, download=True)
        mnist_images = mnist_data.data
        mnist_image_labels = mnist_data.targets
        mnist_ood_labels = torch.zeros(len(mnist_image_labels), dtype=torch.long)

        mnist_fashion_data = mnist.FashionMNIST('data', train=True, download=True)
        ood_images, _ = self.filter_target_class(mnist_fashion_data.data, mnist_fashion_data.targets, target_class=1)
        ood_images = ood_images

        indices = torch.randomperm(len(mnist_fashion_data))[:num_random_ood]
        ood_images = mnist_fashion_data.data[indices]
        # ood_image_labels = mnist_fashion_data.targets[indices]
        ood_image_labels, fake_ood_labels = self.get_ood_labels(num_random_ood)

        images, class_labels, ood_labels = self.concat_datasets(
            [mnist_images, ood_images],
            [mnist_image_labels, ood_image_labels],
            [mnist_ood_labels, fake_ood_labels]
        )

        images, class_labels, ood_labels = self.shuffle(images, class_labels, ood_labels)

        return images, class_labels, ood_labels
