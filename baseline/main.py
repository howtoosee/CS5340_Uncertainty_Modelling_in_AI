from torch.utils import data
import lightning as L
from lightning import Fabric
from model import ImageClassification, OodClassification, MyModel
import data_module


def main():
    # fabric = Fabric(devices='mps' if torch.backends.mps.is_available() else 'auto')
    # fabric = Fabric(devices='auto')

    # Load dataset
    dataset = data_module.MnistOodRandomDataset()
    n = len(dataset)
    train_size = int(0.8 * n)
    train_set, test_set = data.random_split(dataset, [train_size, n - train_size])
    train_loader = data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=64, shuffle=True)

    # Create model
    image_model = ImageClassification(num_classes=10)
    ood_model = OodClassification()
    model = MyModel(image_model, ood_model)

    # Train model
    trainer = L.Trainer(devices='auto')
    trainer.fit(model, train_loader, test_loader)


if __name__ == '__main__':
    main()
