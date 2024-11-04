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
from tqdm import tqdm
from typing import List

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

        self.ground_truth_mask = init_ground_truth(self._coco_api)
        self.turle_mask_predictions = {}

        self.dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
        self.all_contiguous_ids = list(self.dataset_id_to_contiguous_id.values())
        self.num_classes = len(self.all_contiguous_ids)

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            # Extract instances from the output
            instances = output["instances"]

            # Get pred_classes and pred_masks from the instances
            pred_classes = instances.pred_classes
            pred_masks = instances.pred_masks

            # Get the unique classes present in pred_classes
            unique_classes = torch.unique(pred_classes)

            # Create a dictionary to hold the masks for each unique class
            class_masks = {}

            # unmap the category ids for COCO
            
            assert min(self.all_contiguous_ids) == 0 and max(self.all_contiguous_ids) == self.num_classes - 1

            reverse_id_mapping = {v: k for k, v in self.dataset_id_to_contiguous_id.items()}

            # Perform logical OR for each class using tensor operations
            for class_id in unique_classes:
                cat_id = reverse_id_mapping[class_id.item()]
                _mask = torch.any(
                    pred_masks[pred_classes == class_id], dim=0
                ).detach().cpu().numpy()

                # Convert RLE counts from bytes to a UTF-8 string for JSON serialization
                _mask = mask_util.encode(np.asfortranarray(_mask.astype(np.uint8)))
                _mask["counts"] = _mask["counts"].decode("utf-8")

                class_masks[cat_id] = _mask

            # Update the predictions dictionary with the computed masks for the current image
            self.turle_mask_predictions = {**self.turle_mask_predictions, input["image_id"]: class_masks}

        return self.turle_mask_predictions

    def evaluate(self):
        id_map = {v: k for k, v in self.dataset_id_to_contiguous_id.items()}
        print(id_map)
        cat_ids = [id_map[i] for i in range(self.num_classes)]

        eval_result = compute_iou(
            gt=self.ground_truth_mask,
            pred=self.turle_mask_predictions,
            cat_ids=cat_ids
        )
        
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self.turle_mask_predictions))

        turtle_miou = np.array(list(eval_result[1].values())).mean()
        flippers_miou = np.array(list(eval_result[2].values())).mean()
        head_miou = np.array(list(eval_result[3].values())).mean()

        result = {
            "turtle_miou": turtle_miou,
            "flippers_miou": flippers_miou,
            "head_miou": head_miou,
            "average_miou": (turtle_miou + flippers_miou + head_miou) / 3
        }

        if 4 in cat_ids:
            result["whole_turtle_miou"] = np.array(list(eval_result[4].values())).mean()

        return result


def iou(gt_mask, pred_mask):
    """Calculate Intersection over Union (IoU) for two masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union


def get_mask(preds: dict, img_id, cat_id):
    mask = None
    if img_id in preds and cat_id in preds[img_id]:
        mask = preds[img_id][cat_id]

    return mask

def compute_iou(gt, pred, cat_ids=[1, 2, 3]):
    results = {}
    for cat_id in cat_ids:
        results[cat_id] = {}

        for img_id, cats in tqdm(gt.items(), desc=f"Processing cat {cat_id}"):
            gt_mask = cats[cat_id]
            pred_mask = get_mask(pred, img_id, cat_id)
            
            # Convert pred_mask from RLE to a NumPy array if it exists
            if pred_mask is not None:
                pred_mask = mask_util.decode(pred_mask)
            else:
                pred_mask = np.zeros_like(gt_mask)

            # Calculate IoU and store the result
            results[cat_id][img_id] = iou(gt_mask, pred_mask)
            
    return results


def init_ground_truth(gt_coco):
    # provide one ground truth mask for each label in image
    gt = {}
    for img_id in tqdm(gt_coco.imgs, desc="Initializing ground truth images"):
        gt[img_id] = {}
        for cat in gt_coco.getCatIds():
            ann_ids = gt_coco.getAnnIds(imgIds=img_id, catIds=[cat])
            masks = [gt_coco.annToMask(gt_coco.anns[_id]) for _id in ann_ids]
            
            mask = np.logical_or.reduce(masks)

            gt[img_id][cat] = mask

    return gt