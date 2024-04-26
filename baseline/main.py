import argparse
import torch
from torch.utils import data
import lightning as L
from lightning.pytorch import loggers as pl_loggers

import data_module
from model import ImageClassification, OodClassification, MyModel

def run_cifar10_near():
    train_dict = data_module.get_cifar10_near(is_train=True, proportion=0.01)
    test_dict = data_module.get_cifar10_near(is_train=False, proportion=0.5)
    return train_dict, test_dict

def run_cifar10_far():
    train_dict = data_module.get_cifar10_far(is_train=True, proportion=0.01)
    test_dict = data_module.get_cifar10_far(is_train=False, proportion=0.5)
    return train_dict, test_dict

def run_mnist_near():
    train_dict = data_module.get_mnist_near(is_train=True, proportion=0.01)
    test_dict = data_module.get_mnist_near(is_train=False, proportion=0.5)
    return train_dict, test_dict

def run_mnist_far1():
    train_dict = data_module.get_mnist_far1(is_train=True, proportion=0.01)
    test_dict = data_module.get_mnist_far1(is_train=False, proportion=0.5)
    return train_dict, test_dict

def run_mnist_far2():
    train_dict = data_module.get_mnist_far2(is_train=True, proportion=0.01)
    test_dict = data_module.get_mnist_far2(is_train=False, proportion=0.5)
    return train_dict, test_dict

run_configs = dict(
    cifar10_near=run_cifar10_near,
    cifar10_far=run_cifar10_far,
    mnist_near=run_mnist_near,
    mnist_far1=run_mnist_far1,
    mnist_far2=run_mnist_far2,
)

def main(config_name, devices):
    train_dict, test_dict = run_configs[config_name]()

    train_data = train_dict["dataset"]
    test_data = test_dict["dataset"]
    num_classes = train_dict["num_classes_id"]

    train_loader = data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=8)  # TODO change to train_data
    test_loader = data.DataLoader(test_data, batch_size=16, shuffle=True, num_workers=8)

    # Create model
    image_model = ImageClassification(num_classes=num_classes)
    ood_model = OodClassification()
    model = MyModel(image_model, ood_model)

    # Train model
    tb_logger = pl_loggers.TensorBoardLogger(save_dir='lightning_logs/', name=config_name)
    trainer = L.Trainer(max_epochs=100, logger=tb_logger, devices=devices)
    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()
    

    main(args.name, args.device)

