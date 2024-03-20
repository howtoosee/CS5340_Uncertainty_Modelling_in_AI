import torch
from torchmetrics import classification as C
from torch import nn
import lightning as L
from einops import rearrange, reduce, repeat


class ImageClassification(L.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        # mnist images are (28, 28) (channels, width, height)
        self.layers = nn.Sequential(*[
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1),
        ])

        self.metrics = nn.ModuleDict({
            'acc': C.MulticlassAccuracy(num_classes),
            'f1': C.MulticlassF1Score(10, average='micro'),
        })


    def forward(self, x):
        batch_size, width, height = x.size()

        # (b, 28, 28) -> (b, 28*28)
        x = x.view(batch_size, -1)
        return self.layers(x)


    #
    #
    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


    # def test_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'test_loss': avg_loss}
    #     return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}


    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate)
    #     lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
    #                     'name': 'expo_lr'}
    #     return [optimizer], [lr_scheduler]


class OodClassification(L.LightningModule):
    def __init__(self, hidden_dim=10):
        super().__init__()

        self.layers = nn.Sequential(*[
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1),
        ])

        self.metrics = nn.ModuleDict({
            'acc': C.BinaryAccuracy(),
            'f1': C.BinaryF1Score(),
        })


    def forward(self, x):
        return self.layers(x)


class MyModel(L.LightningModule):
    def __init__(self, image_classifier, ood_classifier):
        super().__init__()
        self.image_classifier = image_classifier
        self.ood_classifier = ood_classifier

        self.loss_fn = nn.CrossEntropyLoss()
        self.test_preds = {
            'class': list(),
            'ood': list(),
        }


    def training_step(self, batch, batch_idx):
        images, image_class, ood_class = batch
        image_class_logits = self.image_classifier(images)
        ood_class_logits = self.ood_classifier(image_class_logits)

        ood_classification_loss = self.loss_fn(ood_class_logits, ood_class)

        # remove ood samples from image classification loss
        id_indices = torch.nonzero(image_class >= 0).squeeze()
        image_class_logits = image_class_logits[id_indices]
        image_class = image_class[id_indices]

        image_classification_loss = self.loss_fn(image_class_logits, image_class)

        total_loss = image_classification_loss + ood_classification_loss

        self.log('train/loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/image_class_loss', image_classification_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/ood_class_loss', ood_classification_loss, on_step=False, on_epoch=True, prog_bar=False)

        return {
            'loss': total_loss,
            'image_class_loss': image_classification_loss,
            'ood_class_loss': ood_classification_loss
        }


    def validation_step(self, batch, batch_idx):
        images, image_class, ood_class = batch

        image_class_logits = self.image_classifier(images)
        ood_class_logits = self.ood_classifier(image_class_logits)

        # print(image_class_logits.size(), image_class.size())
        # print(ood_class_logits.size(), ood_class.size())
        image_classification_loss = self.loss_fn(image_class_logits, image_class)
        ood_classification_loss = self.loss_fn(ood_class_logits, ood_class)
        total_loss = image_classification_loss + ood_classification_loss

        ## Accumulate losses automatically
        self.log('test/loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/image_class_loss', image_classification_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/ood_class_loss', ood_classification_loss, on_step=False, on_epoch=True, prog_bar=False)

        # class_metrics = {k: compute_metric(image_class_logits.cpu(), image_class.cpu()) for k, compute_metric in self.image_classifier.metrics.items()}
        # ood_metrics = {k: compute_metric(ood_class_logits.cpu(), ood_class.cpu()) for k, compute_metric in self.ood_classifier.metrics.items()}
        ood_class_preds = ood_class_logits.argmax(dim=1)
        self.test_preds['class'].extend(image_class_logits.cpu().numpy().tolist())
        # from sklearn.metrics import f1_score
        # print(f1_score(ood_class.cpu(), ood_class_preds.cpu()))

        for name, metric in self.image_classifier.metrics.items():
            metric.update(image_class_logits, image_class)
        for name, metric in self.ood_classifier.metrics.items():
            metric.update(ood_class_preds, ood_class)


    def on_validation_epoch_end(self):
        class_metrics = {k: metric.compute() for k, metric in self.image_classifier.metrics.items()}
        ood_metrics = {k: metric.compute() for k, metric in self.ood_classifier.metrics.items()}

        self.log('test/image_class_acc', class_metrics['acc'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/image_class_f1', class_metrics['f1'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/ood_class_acc', ood_metrics['acc'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/ood_class_f1', ood_metrics['f1'], on_step=False, on_epoch=True, prog_bar=False)

        ood_preds = torch.tensor(self.test_preds['ood'])
        print("Predicted OOD:", torch.nonzero(ood_preds > 1).tolist())

        return {
            # 'test_loss': total_loss,
            # 'test_image_class_loss': image_classification_loss,
            # 'test_ood_class_loss': ood_classification_loss,
            'test_metrics': {
                'class': class_metrics,
                'ood': ood_metrics
            }
        }


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
