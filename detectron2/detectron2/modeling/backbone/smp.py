import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

import segmentation_models_pytorch as smp

import torch.optim as optim
import torch.nn as nn

from .backbone import Backbone
from .build import BACKBONE_REGISTRY

__all__ = [
    "Unet",
    "build_unet_backbone",
]

class Unet(Backbone):
    def __init__(self, **kwargs):
        super().__init__()
        self.net = smp.Unet(**kwargs)

    def forward(self, x):
        return self.net(x)
    
    def output_shape(self):
        return None

    @property
    def size_divisibility(self):
        return 32

@BACKBONE_REGISTRY.register()
def build_unet_backbone(cfg, input_shape):
    """
    Create a Model instance from segmentation_models_pytorch, from config.

    Returns:
        Unet: a :class:`Unet` instance.
    """
    encoder_name        = cfg.MODEL.BACKBONE.ENCODER_NAME
    encoder_weights     = cfg.MODEL.BACKBONE.WEIGHTS
    in_channels         = cfg.MODEL.BACKBONE.IN_CHANNELS
    classes             = cfg.MODEL.BACKBONE.NUM_CLASSES

    # Define the U-Net model with a pre-trained ResNet34 backbone
    return Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,     # Use ImageNet pre-trained weights
        in_channels=in_channels,                  # Input channels (RGB)
        classes=classes,                      # Output classes (background, head, body, flippers)
        activation=None           # No softmax applied here; handled by CrossEntropyLoss
    )