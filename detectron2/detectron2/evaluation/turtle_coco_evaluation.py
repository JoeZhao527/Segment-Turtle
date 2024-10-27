import contextlib
import io
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from detectron2.utils.file_io import PathManager
from detectron2.structures import Boxes, BoxMode
from detectron2.data import MetadataCatalog

from .evaluator import DatasetEvaluator
from .coco_evaluation import COCOEvaluator

class TurtleCOCOEvaluator(COCOEvaluator):
    """
    Training time evaluation to select best checkpoint

    Only provide bbox evaluation and select best checkpoint based on that, since
    bbox proposal evaluation requires less evaluation time.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print(self._coco_api)
        exit(0)
    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            print("output: ", output)
            prediction = {"image_id": input["image_id"]}

            # Get the unique classes present in pred_classes
            unique_classes = torch.unique(outputs['pred_classes'])

            # Create a dictionary to hold the masks for each unique class
            class_masks = {}

            # Perform logical OR for each class using tensor operations
            for class_id in unique_classes:
                class_masks[class_id.item()] = torch.any(outputs['pred_classes'][outputs['pred_classes'] == class_id], dim=0)

            prediction['pred_masks'] = class_masks

            self._predictions.append(prediction)

        return self._predictions
