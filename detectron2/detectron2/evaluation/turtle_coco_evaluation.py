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
    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)

