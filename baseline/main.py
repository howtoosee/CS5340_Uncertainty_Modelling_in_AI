import torch
from torch.utils import data
import lightning as L
from lightning import Fabric
from lightning.pytorch import loggers as pl_loggers
from einops import rearrange

import data_module
from model import ImageClassification, OodClassification, MyModel


def main():

    train_dict = data_module.get_cifar10_near(is_train=True)
    test_dict = data_module.get_cifar10_near(is_train=False)

    train_data = train_dict["dataset"]
    test_data = test_dict["dataset"]
    num_classes = train_dict["num_classes_id"]

    train_loader = data.DataLoader(test_data, batch_size=16, shuffle=True, num_workers=2)  # TODO change to train_data
    test_loader = data.DataLoader(test_data, batch_size=16, shuffle=True, num_workers=2)

    # Create model
    image_model = ImageClassification(num_classes=num_classes)
    ood_model = OodClassification()
    model = MyModel(image_model, ood_model, cutoff_epoch=40)

    # Train model
    trainer = L.Trainer(max_epochs=100, logger=True)
    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    main()
