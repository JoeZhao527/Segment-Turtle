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

import torchvision
from torchvision import models, transforms


__all__ = [
    "Deeplab",
    "build_deeplab_backbone",
]

class Deeplab(Backbone):
    def __init__(self, num_categories, **kwargs):
        super().__init__()
        self.net = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
        self.net.classifier[4] = nn.Conv2d(256, num_categories, kernel_size=(1, 1))

    def forward(self, x):
        return self.net(x)['out']
    
    def output_shape(self):
        return None

    @property
    def size_divisibility(self):
        return 32

@BACKBONE_REGISTRY.register()
def build_deeplab_backbone(cfg, input_shape):
    """
    Create a Model instance from segmentation_models_pytorch, from config.

    Returns:
        Unet: a :class:`Unet` instance.
    """
    classes             = cfg.MODEL.BACKBONE.NUM_CLASSES

    # Define the U-Net model with a pre-trained ResNet34 backbone
    return Deeplab(
        num_categories=classes
    )