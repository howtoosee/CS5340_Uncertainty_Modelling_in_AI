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
            nn.Conv2d(1, 8, 3, 1), # (b, 8, 26, 26)
            nn.ReLU(),
            nn.Conv2d(8, 16, 5, 1), # (b, 16, 22, 22)
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # (b, 16, 11, 11)
            nn.Flatten(), # (b, 16*11*11)
            nn.Linear(16*11*11, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        ])

        self.metrics = nn.ModuleDict({
            'acc': C.MulticlassAccuracy(num_classes),
            'f1': C.MulticlassF1Score(10, average='micro'),
        })


    def forward(self, x):
        # batch_size, width, height = x.size()

        # # (b, 28, 28) -> (b, 28*28)
        # x = x.view(batch_size, -1)
        # x: (batch_size, 28, 28)
        return self.layers(x.unsqueeze(1))


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
    def __init__(self, hidden_dim=16):
        super().__init__()

        self.layers = nn.Sequential(*[
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        ])

        self.metrics = nn.ModuleDict({
            'acc': C.BinaryAccuracy(),
            'f1': C.BinaryF1Score(),
        })


    def forward(self, x):
        # x: (batch_size, 10)
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
        ood_class_logits = self.ood_classifier(image_class_logits.detach())

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
        ood_class_logits = self.ood_classifier(image_class_logits.detach())

        # print(image_class_logits.size(), image_class.size())
        # print(ood_class_logits.size(), ood_class.size())
        
        ood_classification_loss = self.loss_fn(ood_class_logits, ood_class)

        # remove ood samples from image classification loss
        id_indices = torch.nonzero(image_class >= 0).squeeze()
        image_class_logits = image_class_logits[id_indices]
        image_class = image_class[id_indices]

        image_classification_loss = self.loss_fn(image_class_logits, image_class)
        
        total_loss = image_classification_loss + ood_classification_loss

        ## Accumulate losses automatically
        self.log('test/loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/image_class_loss', image_classification_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/ood_class_loss', ood_classification_loss, on_step=False, on_epoch=True, prog_bar=False)

        # class_metrics = {k: compute_metric(image_class_logits.cpu(), image_class.cpu()) for k, compute_metric in self.image_classifier.metrics.items()}
        # ood_metrics = {k: compute_metric(ood_class_logits.cpu(), ood_class.cpu()) for k, compute_metric in self.ood_classifier.metrics.items()}
        
        ood_class_preds = ood_class_logits.argmax(dim=1)
        self.test_preds['class'].extend(image_class_logits.cpu().numpy().tolist())
        self.test_preds['ood'].extend(ood_class_preds.cpu().numpy().tolist())
        # from sklearn.metrics import f1_score
        # print(f1_score(ood_class.cpu(), ood_class_preds.cpu()))

        for name, metric in self.image_classifier.metrics.items():
            metric.update(image_class_logits.argmax(dim=1), image_class)
        for name, metric in self.ood_classifier.metrics.items():
            metric.update(ood_class_preds, ood_class)

    def on_validation_epoch_end(self):
        class_metrics = {k: metric.compute() for k, metric in self.image_classifier.metrics.items()}
        ood_metrics = {k: metric.compute() for k, metric in self.ood_classifier.metrics.items()}

        self.log('test/image_class_acc', class_metrics['acc'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/image_class_f1', class_metrics['f1'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/ood_class_acc', ood_metrics['acc'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/ood_class_f1', ood_metrics['f1'], on_step=False, on_epoch=True, prog_bar=False)

        print(class_metrics)
        print(ood_metrics)

        ood_preds = torch.tensor(self.test_preds['ood'])

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
