from pytorch_lightning import LightningModule
import torch
from torch import nn

from typing import Dict, Any, List, Optional
from gan_lightning.utils import get_optimizer
from gan_lightning.src.models import model_registration
from gan_lightning.src.models.blocks.classifier_blocks import *


@model_registration("Controllable_Classifier")
class Controllable_Classifier(LightningModule):
    def __init__(
        self,
        losses: Optional[Dict[str, Any]] = None,
        optimizer_dict: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        img_channel: int = 3,
        num_classes: int = 40,
        hidden_dim: int = 128,
        mode: str = "train",
        **kwargs,
    ):
        super(Controllable_Classifier, self).__init__()

        self.img_channel = img_channel
        self.down_sampling = DownSamplingBlock(self.img_channel, hidden_dim, stride=2)

        self.mobilenetv2_feature_extractor = Mobilenetv2_Feature_Extractor(
            in_channels=hidden_dim * 2
        )

        self.num_classes = num_classes
        self.classifier_head = ClassificationHead(hidden_dim * 3, num_classes)

        if mode == "train":
            self.set_attributes(training_config)
            self.weight_init(training_config["weight_init_name"])
            self.optimizer_dict = optimizer_dict
            self.classifier_loss = losses.get("classifier_loss", None)()
            self.metrics = {"multi_label_accuracy": self.multi_label_accuracy}

    def forward(self, x):
        shortcut, downsampled = self.down_sampling(x)
        feature_extracted = self.mobilenetv2_feature_extractor(shortcut)
        concat_features = torch.cat([downsampled, feature_extracted], dim=1)
        class_pred = self.classifier_head(concat_features)

        return class_pred

    def training_step(self, batch: List[torch.Tensor], batch_idx: int):
        X, y = batch
        X = X.to(self.device)
        labels = y.to(self.device).float()
        y_pred = self(X).float()
        train_loss = self.classifier_loss(y_pred, labels)
        self.train_multi_label_accuracy = self.metrics["multi_label_accuracy"](
            labels, y_pred
        )

        self.logger.experiment.log_metric(
            run_id=self.logger.run_id, key="train_loss", value=train_loss
        )
        self.log("train_loss", train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int):
        X, y = batch
        X = X.to(self.device)
        labels = y.to(self.device).float()
        y_pred = self(X).float()
        val_loss = self.classifier_loss(y_pred, labels)

        self.val_multi_label_accuracy = self.metrics["multi_label_accuracy"](
            labels, y_pred
        )

        self.logger.experiment.log_metric(
            run_id=self.logger.run_id, key="val_loss", value=val_loss
        )
        self.log("val_loss", val_loss, prog_bar=True)
        self.log(
            "val_multi_label_accuracy", self.val_multi_label_accuracy, prog_bar=True
        )
        return val_loss

    def validation_epoch_end(self, outputs):
        self.logger.experiment.log_metric(
            run_id=self.logger.run_id,
            key="val_multi_label_accuracy",
            value=self.val_multi_label_accuracy,
        )

    def training_epoch_end(self, outputs):
        self.logger.experiment.log_metric(
            run_id=self.logger.run_id,
            key="train_multi_label_accuracy",
            value=self.train_multi_label_accuracy,
        )

    def configure_optimizers(self):
        classifier_optimizer = get_optimizer(
            self.parameters(), self.optimizer_dict, betas=(0.5, 0.999)
        )

        return [classifier_optimizer[0]], [classifier_optimizer[1]]

    def set_attributes(self, config: Dict[str, Any]):
        for key, value in config.items():
            setattr(self, key, value)

    def weight_init(self, mode):
        if mode == "kaiming":
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
        elif mode == "xavier":
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)

        elif mode == "orthogonal":
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal_(m.weight)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        elif mode == "normal":
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.normal_(m.weight, 0, 0.02)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        else:
            raise ValueError("Unknown weight initialization")

    def multi_label_accuracy(self, y_true, y_pred, threshold=0.5):
        # apply sigmoid preds to get binary predictions
        y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred > threshold).int()
        correct = (y_pred == y_true.int()).all(dim=1).float()
        accuracy = correct.sum() / len(correct)
        return accuracy

    def get_name(self):
        return "Controllable_Classifier"


if __name__ == "__main__":
    import torch.onnx
    from gan_lightning.src.models.generators.deepconv_generator import (
        DeepConv_Generator,
    )

    input_dim = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = DeepConv_Generator(input_dim=input_dim).to(device)
    generator.load_state_dict(
        torch.load("gan_lightning/checkpoints/Celeba_Generator.pth")
    )
    print("Generator loaded successfully")
