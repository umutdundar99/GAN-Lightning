import lightning.pytorch as pl
import torch
from torch import nn
from gan_lightning.src.models.blocks.classifier_blocks import (  # noqa
    controllable_classifier_block,
)

from gan_lightning.src.models import model_registration


@model_registration("Controllable_Classifier")
class Controllable_Classifier(pl.LightningModule):
    def __init__(self, img_channel=3, n_classes=2, hidden_dim=64):
        super().__init__()
        
        self.classifier = nn.Sequential(
            controllable_classifier_block(img_channel, hidden_dim),
            controllable_classifier_block(hidden_dim, hidden_dim*2),
            controllable_classifier_block(hidden_dim*2, hidden_dim*4),
        )
        
        self.final_block = controllable_classifier_block(hidden_dim*4, n_classes, final_layer=True) 
        
    def forward(self, x): 
        class_pred = self.classifier(x)
        class_pred = self.final_block(class_pred, True)
        return class_pred.view(len(class_pred), -1)