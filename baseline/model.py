import torch
from torch import nn
from torchmetrics import classification as C
import lightning as L


class ImageClassification(L.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        # images size (b, 1, 32, 32)
        self.layers = nn.Sequential(
            *[
                nn.Conv2d(1, 8, 3, 1),  # (b, 8, 30, 30)
                nn.ReLU(),
                nn.Conv2d(8, 16, 5, 1),  # (b, 16, 26, 26)
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # (b, 16, 13, 13)
                nn.Flatten(),  # (b, 16x13x13)
                nn.Linear(16 * 13 * 13, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes),
            ]
        )

        self.metrics = nn.ModuleDict(
            {
                "acc": C.MulticlassAccuracy(num_classes),
                "f1": C.MulticlassF1Score(num_classes, average="micro"),
                "f1_all": C.MulticlassF1Score(num_classes, average=None),
            }
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # batch_size, width, height = x.size()
        return self.layers(x)

    def update_metrics(self, logits, gt):
        if logits.shape[0] == 0 or gt.shape[0] == 0:
            return
        ## Update image classification metrics
        probs = torch.softmax(logits, dim=1)
        for name, metric in self.metrics.items():
            metric.update(probs, gt)


class OodClassification(L.LightningModule):
    def __init__(self, input_dim=10, hidden_dim=16):
        super().__init__()

        self.layers = nn.Sequential(
            *[
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            ]
        )

        self.metrics = nn.ModuleDict(
            {
                "acc": C.BinaryAccuracy(),
                "f1": C.BinaryF1Score(),
                "auroc": C.BinaryAUROC(),
                "auprc": C.BinaryAveragePrecision(),
            }
        )

        # reduction = "mean" if loss_weight is not None else None
        # self.criterion = nn.CrossEntropyLoss(weight=loss_weight, reduction=reduction)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # x: (batch_size, 10)
        return self.layers(x)

    def update_metrics(self, logits, gt):
        ## Update OOD classifcation metrics
        probs = torch.softmax(logits, dim=1)
        for name, metric in self.metrics.items():
            metric.update(probs[:, 1], gt)


class MyModel(L.LightningModule):
    def __init__(self, image_classifier, ood_classifier, cutoff_epoch=None):
        super().__init__()
        self.image_classifier = image_classifier
        self.ood_classifier = ood_classifier
        self.cutoff_epoch = cutoff_epoch  # Epoch at which to stop training the image classifier

    def forward(self, image):
        image_class_logits = self.image_classifier(image)
        ood_class_logits = self.ood_classifier(image_class_logits.detach())
        return image_class_logits, ood_class_logits

    def _remove_ood_data(self, image_class_logits, image_class_gt, ood_class_gt):
        ## Remove OOD samples from batch
        ## ood_class == 0 -> in-distribution
        id_indices = torch.nonzero(ood_class_gt == 0).squeeze()
        image_class_logits = image_class_logits[id_indices]
        image_class_gt = image_class_gt[id_indices]
        assert (
            image_class_gt.shape[0] == 0 or image_class_gt.max() <= self.image_classifier.num_classes
        ), f"{list(zip(image_class_gt.tolist(), ood_class_gt.tolist()))}"

        return image_class_logits, image_class_gt

    def training_step(self, batch, batch_idx):
        images, image_class_gt, ood_class_gt = batch
        # print(f"Training: {ood_class_gt.sum() / len(ood_class_gt):.3f} are OOD")
        image_class_logits, ood_class_logits = self(images)

        ## Compute OOD detection loss
        ood_classification_loss = self.ood_classifier.criterion(ood_class_logits, ood_class_gt)

        ## Compute image classification loss
        image_class_logits, image_class_gt = self._remove_ood_data(image_class_logits, image_class_gt, ood_class_gt)
        image_classification_loss = self.image_classifier.criterion(image_class_logits, image_class_gt)

        total_loss = image_classification_loss + ood_classification_loss

        self.log("loss/train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss/train_image_class_loss", image_classification_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("loss/train_ood_class_loss", ood_classification_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        self.ood_classifier.update_metrics(ood_class_logits, ood_class_gt)
        self.image_classifier.update_metrics(image_class_logits, image_class_gt)

        return {
            "loss": total_loss,
            "image_class_loss": image_classification_loss,
            "ood_class_loss": ood_classification_loss,
        }

    def on_train_epoch_end(self) -> None:
        class_metrics = {k: metric.compute().detach() for k, metric in self.image_classifier.metrics.items()}
        # ood_metrics = {k: metric.compute().detach() for k, metric in {**self.ood_classifier.metrics_pred, **self.ood_classifier.metrics_logit}.items()}
        ood_metrics = {k: metric.compute().detach() for k, metric in self.ood_classifier.metrics.items()}

        self.log("train_metrics/image_class_acc", class_metrics["acc"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train_metrics/image_class_f1", class_metrics["f1"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        for name, val in sorted(ood_metrics.items()):
            self.log(f"train_metrics/ood_class_{name}", val, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return {"train_metrics": {"class": class_metrics, "ood": ood_metrics}}

    def validation_step(self, batch, batch_idx):
        images, image_class_gt, ood_class_gt = batch
        image_class_logits, ood_class_logits = self(images)

        ## Compute OOD detection loss
        ood_classification_loss = self.ood_classifier.criterion(ood_class_logits, ood_class_gt)

        ## Compute image classification loss
        image_class_logits, image_class_gt = self._remove_ood_data(image_class_logits, image_class_gt, ood_class_gt)
        if image_class_logits.shape[0] > 0 and image_class_gt.shape[0] > 0:
            ## batch has ID samples
            image_classification_loss = self.image_classifier.criterion(image_class_logits, image_class_gt)
        else:
            ## batch only contains OOD samples
            image_classification_loss = 0

        total_loss = image_classification_loss + 5 * ood_classification_loss

        ## Accumulate losses automatically
        self.log("loss/test_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss/test_image_class_loss", image_classification_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("loss/test_ood_class_loss", ood_classification_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        self.ood_classifier.update_metrics(ood_class_logits, ood_class_gt)
        self.image_classifier.update_metrics(image_class_logits, image_class_gt)

    def on_validation_epoch_end(self):
        class_metrics = {k: metric.compute().detach() for k, metric in self.image_classifier.metrics.items()}
        ood_metrics = {k: metric.compute().detach() for k, metric in self.ood_classifier.metrics.items()}

        self.log("test_metrics/image_class_acc", class_metrics["acc"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("test_metrics/image_class_f1", class_metrics["f1"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        for name, val in sorted(ood_metrics.items()):
            self.log(f"test_metrics/ood_class_{name}", val, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        print(f"Testing epoch {self.current_epoch}")
        print("Class", class_metrics)
        print("OOD", ood_metrics)

        return {"test_metrics": {"class": class_metrics, "ood": ood_metrics}}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer
