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
        num_classes=3,
        ignore_label=None,
    ):
        self._logger = logging.getLogger(__name__)

        self._cpu_device = "cpu"
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self._num_classes = num_classes
        self.turle_mask_predictions = {}

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=int)

            # original resolution ground truth is in numpy.array.
            # Reference to data.dataset_mapper.TurtleSemanticDatasetMapper
            gt = input["sem_seg_gt"]

            # Convert pred and gt to RLE format
            pred_rle = encode_multi_class_mask(pred)
            gt_rle = encode_multi_class_mask(gt)

            self.turle_mask_predictions = {
                **self.turle_mask_predictions,
                input["image_id"]: {
                    "gt": gt_rle,
                    "pred": pred_rle,
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

        self._logger.info(f"Evaluting sample num: {len(turtle_ious)}, {len(flippers_ious)}, {len(head_ious)} (turtle, flippers, head)")
        
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


class TurtleFocusSemSegEvaluator(TurtleSemSegEvaluator):
    """
    This include recover original image size from focus cropped image
    """
    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=int)

            # Recover the original image size
            pad_mask_to_original_size(pred, input['crop_pos'])

            # original resolution ground truth is in numpy.array.
            # Reference to data.dataset_mapper.TurtleSemanticDatasetMapper
            gt = input["sem_seg_gt"]

            # Convert pred and gt to RLE format
            pred_rle = encode_multi_class_mask(pred)
            gt_rle = encode_multi_class_mask(gt)

            self.turle_mask_predictions = {
                **self.turle_mask_predictions,
                input["image_id"]: {
                    "gt": gt_rle,
                    "pred": pred_rle,
                    "turtle_iou": compute_iou(pred, gt, 1),
                    "flippers_iou": compute_iou(pred, gt, 2),
                    "head_iou": compute_iou(pred, gt, 3),
                }
            }


def pad_mask_to_original_size(mask, crop_pos):
    """
    Pad the cropped mask back to original image size using crop position information
    
    Args:
        mask (np.ndarray): Cropped mask of shape (H', W')
        crop_pos (tuple): Tuple of (x_start, y_start, orig_height, orig_width)
        
    Returns:
        np.ndarray: Padded mask of original image size (H, W)
    """
    x_start, y_start, orig_height, orig_width = crop_pos

    # Create empty mask of original size
    padded_mask = np.zeros((orig_height, orig_width), dtype=mask.dtype)
    
    print(y_start+h, x_start+w)
    print(padded_mask.shape)
    print(mask.shape)
    # Insert cropped mask at correct position
    h, w = mask.shape
    padded_mask[y_start:y_start+h, x_start:x_start+w] = mask
    
    return padded_mask


def encode_multi_class_mask(mask):
    """
    Encode a multi-class mask into RLE format for each class.

    Args:
        mask (np.ndarray): The input mask of shape (H, W) with integer class labels.

    Returns:
        dict: A dictionary where keys are class labels and values are RLE-encoded masks.
    """
    rle_dict = {}
    unique_classes = np.unique(mask)

    for _cls in unique_classes:
        class_id = int(_cls)
        binary_mask = (mask == _cls).astype(np.uint8)
        rle = mask_util.encode(np.asfortranarray(binary_mask))
        rle["counts"] = rle["counts"].decode("utf-8")
        rle_dict[class_id] = rle

    return rle_dict

# Helper function to compute IoU for a specific class
def compute_iou(pred, target, class_id):
    pred_binary = (pred == class_id).astype(np.uint8)
    target_binary = (target == class_id).astype(np.uint8)
    intersection = np.logical_and(pred_binary, target_binary).sum()
    union = np.logical_or(pred_binary, target_binary).sum()
    return float(intersection / union) if union != 0 else 0.0