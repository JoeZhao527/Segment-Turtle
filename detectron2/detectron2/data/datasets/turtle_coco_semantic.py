# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
import shutil
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image
import pandas as pd

from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager
from pycocotools.coco import COCO
from tqdm import tqdm
import copy
from typing import List

from .. import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

def load_coco_api_semantic(coco_api, image_root, dataset_name=None, extra_annotation_keys=None):
    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)

    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), "input coco"))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for img_dict, anno_dict_list in tqdm(imgs_anns, desc=f"Preparing semantic segmentation mask"):
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]
        
        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS

            objs.append(obj)

        record["annotations"] = objs

        # Prepare semantic segmentation mask
        record['sem_seg'] = process_semantic_mask(record)

        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts


def split_n_prepare_turtle_semantic_coco(data_dir: str, dev_mode: bool = False, test_only: bool = False):
    """
    Process and prepare the turtle dataset under data_dir.

    It prepares train, valid, and test sets by making a single mask contains 4 categories:
        0) background
        1) carapace
        2) flippers
        3) head

    Args:
        data_dir (str): the turtle data root dir
        dev_mode (bool): if true, only process a small portion of data for development
        
    Returns:
        Dict of dataset:
        {
            `dataset_name`: `dataset <List[dict]>`,
            ...
        }

    Avaliable dataset names:
        - Turtle body parts:
            turtle_parts_train
            turtle_parts_valid
            turtle_parts_test
    """
    # Initialize paths
    annotations_path = os.path.join(data_dir, 'annotations.json')
    split_path = os.path.join(data_dir, 'metadata_splits.csv')

    # Load the split dataframe
    split = pd.read_csv(split_path)

    # Load the COCO annotations
    coco = COCO(annotations_path)

    # Separate the COCO ids into train/valid/test based on the split DataFrame
    train_ids = split[split['split_open'] == 'train']['id'].tolist()
    valid_ids = split[split['split_open'] == 'valid']['id'].tolist()
    test_ids = split[split['split_open'] == 'test']['id'].tolist()

    # Only process a small portion of data for development
    if dev_mode:
        train_ids = train_ids[:80]
        valid_ids = valid_ids[:40]
        test_ids = test_ids[:20]

    datasets = {}
    # Register each dataset split
    for split_name, img_ids in zip(
        ["train", "valid", "test"], [train_ids, valid_ids, test_ids]
    ):
        if test_only:
            if split_name != "test":
                continue

        logger.info(f"Preprocessing and registering for {split_name} data...")

        # Process and register the turtle body parts dataset
        _data_name = f"turtle_parts_{split_name}"
        split_coco_parts = create_split_coco(img_ids, coco, process_body_parts=True)
        datasets[_data_name] = load_coco_api_semantic(split_coco_parts, data_dir, _data_name)

    return datasets


def register_turtle_coco(dataset: List[dict], dataset_name: str, data_dir: str, meta: dict = {}):
    DatasetCatalog.register(
        dataset_name,
        lambda: dataset
    )
    MetadataCatalog.get(dataset_name).set(
        evaluator_type="coco", image_root=data_dir, **meta
    )


def create_split_coco(img_ids, coco, process_body_parts):
    """
    Create a COCO dataset split, processing body parts if specified.

    Args:
        img_ids (list): List of image IDs for the split.
        coco (COCO): Original COCO object.
        process_body_parts (bool): Whether to apply body parts processing.

    Returns:
        COCO: Processed COCO object for the dataset split.
    """
    split_annotations = {}
    for img_id in img_ids:
        anns = coco.imgToAnns[img_id]

        if process_body_parts:
            # Leave annotations for body parts dataset
            processed_anns = anns
        else:
            # For the whole turtle dataset, keep only category 1 annotations
            processed_anns = [ann for ann in anns if ann["category_id"] == 1]

        # Make detectron2 happy
        for _ann in processed_anns:
            _ann["iscrowd"] = 0

        # Add processed annotations to new split dict
        for ann in processed_anns:
            split_annotations[ann["id"]] = ann

    # Create new COCO object with processed split annotations
    split_coco = COCO()
    split_coco.dataset = {
        "images": [coco.imgs[img_id] for img_id in img_ids],
        "annotations": list(split_annotations.values()),
        "categories": coco.dataset["categories"]
    }
    split_coco.createIndex()
    return split_coco


def process_semantic_mask(data: dict):
    """
    Prepare a semantic segmentation mask `sem_seg` for the input image data dictionary.
    
    Args:
        data (dict): A dictionary containing annotations for an image.
        
    Returns:
        np.ndarray: the semantic segmentation mask for the input sample
    """
    # Initialize a blank mask for the entire image
    height, width = data["height"], data["width"]
    sem_seg = np.zeros((height, width), dtype=np.uint8)
    
    # Loop through each annotation and fill the mask
    for ann in data["annotations"]:
        # Get the category_id to fill in the mask
        category_id = ann["category_id"]
        
        # Update the semantic segmentation mask with the category_id
        binary_mask = mask_util.decode(ann["segmentation"]).astype(np.uint8)
        update_mask = (binary_mask == 1) & (sem_seg != 2) & (sem_seg != 3)
        sem_seg[update_mask] = np.uint8(category_id)

    return sem_seg
