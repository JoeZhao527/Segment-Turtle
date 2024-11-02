# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
from typing import Optional, Union
import pycocotools.mask as mask_util
import torch
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from .evaluator import DatasetEvaluator
from .sem_seg_evaluation import SemSegEvaluator


class TurtleSemSegEvaluator(SemSegEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        sem_seg_loading_fn=None,
        num_classes=None,
        ignore_label=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            sem_seg_loading_fn: function to read sem seg file and load into numpy array.
                Default provided, but projects can customize.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn(
                "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = meta.stuff_classes
        self.sem_seg_loading_fn = sem_seg_loading_fn
        self._num_classes = len(meta.stuff_classes)
        if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

        self.turle_mask_predictions = {}

    def process(self, inputs, outputs):
        self.turle_mask_predictions = {}

        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=int)

            gt = input["sem_seg"]

            gt[gt == self._ignore_label] = self._num_classes

            self.turle_mask_predictions = {
                **self.turle_mask_predictions,
                input["image_id"]: {
                    "gt": gt,
                    "pred": pred,
                    "turtle_iou": compute_iou(pred, gt, 1),
                    "flippers_iou": compute_iou(pred, gt, 2),
                    "head_iou": compute_iou(pred, gt, 3),
                }
            }

    def evaluate(self):
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self.turle_mask_predictions))

        # Compute mean IoU for each category based on `self.turle_mask_predictions`
        turtle_ious = [pred["turtle_iou"] for pred in self.turle_mask_predictions.values()]
        flippers_ious = [pred["flippers_iou"] for pred in self.turle_mask_predictions.values()]
        head_ious = [pred["head_iou"] for pred in self.turle_mask_predictions.values()]

        # Calculate mean IoU for each category
        res = {
            "turtle_miou": np.mean(turtle_ious) if turtle_ious else 0,
            "flippers_miou": np.mean(flippers_ious) if flippers_ious else 0,
            "head_miou": np.mean(head_ious) if head_ious else 0,
        }
        res['average_miou'] = (res['turtle_miou'] + res['flippers_miou'] + res['head_miou']) / 3

        # Save the evaluation results if an output directory is specified
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        
        # Log and return results in the required format
        self._logger.info(res)
        return res

    

# Helper function to compute IoU for a specific class
def compute_iou(pred, target, class_id):
    pred_binary = (pred == class_id).astype(np.uint8)
    target_binary = (target == class_id).astype(np.uint8)
    intersection = np.logical_and(pred_binary, target_binary).sum()
    union = np.logical_or(pred_binary, target_binary).sum()
    return intersection / union if union != 0 else 0