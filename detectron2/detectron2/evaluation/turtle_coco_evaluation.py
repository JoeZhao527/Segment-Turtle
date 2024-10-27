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

class TurtleCOCOEvaluator(DatasetEvaluator):
    """
    Evaluator for instance segmentation to compute mIoU (mean Intersection over Union)
    for each class only.
    """

    def __init__(self, dataset_name, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            output_dir (str): optional, an output directory to dump results.
        """
        self._logger = logging.getLogger(__name__)
        self._output_dir = output_dir
        self._metadata = MetadataCatalog.get(dataset_name)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Process the model outputs.
        Args:
            inputs: the inputs to a COCO model.
            outputs: the outputs of a COCO model.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(torch.device("cpu"))
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])

            if len(prediction) > 1:
                self._predictions.append(prediction)

    def evaluate(self):
        """
        Evaluate the model predictions using mIoU.
        """
        if len(self._predictions) == 0:
            self._logger.warning("No valid predictions received.")
            return {}

        # Merge all predictions
        coco_results = list(itertools.chain(*[x["instances"] for x in self._predictions]))
        self._logger.info("Evaluating mIoU for instance segmentation ...")

        # Load the results into COCO format
        coco_dt = self._coco_api.loadRes(coco_results)
        mIoU_results = self._compute_miou(coco_dt)

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            file_path = os.path.join(self._output_dir, "coco_miou_results.json")
            with open(file_path, "w") as f:
                json.dump(mIoU_results, f)

        return mIoU_results

    def _compute_miou(self, coco_dt):
        """
        Compute the mean IoU for each class.

        Args:
            coco_dt (COCO): the COCO object for the detected results.

        Returns:
            A dictionary with mIoU for each class.
        """
        coco_gt = self._coco_api
        iou_per_class = {}
        for cat_id in coco_gt.getCatIds():
            ious = []
            img_ids = coco_gt.getImgIds(catIds=[cat_id])
            for img_id in img_ids:
                gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id], catIds=[cat_id]))
                dt_anns = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=[img_id], catIds=[cat_id]))

                if not gt_anns or not dt_anns:
                    continue

                gt_masks = [mask_util.decode(ann['segmentation']) for ann in gt_anns]
                dt_masks = [mask_util.decode(ann['segmentation']) for ann in dt_anns]

                # Calculate IoU for each pair of gt and dt masks
                for gt_mask in gt_masks:
                    for dt_mask in dt_masks:
                        iou = self._calculate_iou(gt_mask, dt_mask)
                        ious.append(iou)

            # Compute mean IoU for the current class
            if ious:
                iou_per_class[cat_id] = np.mean(ious)
            else:
                iou_per_class[cat_id] = float('nan')

        # Log the mIoU results
        for cat_id, miou in iou_per_class.items():
            cat_name = self._metadata.get("thing_classes")[cat_id - 1]
            self._logger.info(f"Class '{cat_name}' (ID: {cat_id}): mIoU = {miou:.4f}")

        return iou_per_class

    def _calculate_iou(self, mask1, mask2):
        """
        Calculate IoU (Intersection over Union) between two binary masks.

        Args:
            mask1, mask2 (np.ndarray): Binary masks.

        Returns:
            float: IoU value.
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return intersection / union


def instances_to_coco_json(instances, img_id):
    """
    Convert "Instances" object to COCO format.

    Args:
        instances (Instances): predicted instances.
        img_id (int): image ID.

    Returns:
        list[dict]: list of annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k] + 1,  # COCO category IDs start from 1
            "bbox": boxes[k].tolist(),
            "score": scores[k],
            "segmentation": rles[k] if has_mask else None,
        }
        results.append(result)
    return results
