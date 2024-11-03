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
from .turtle_coco import load_coco_api, process_ann

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

def split_n_prepare_turtle_coco(data_dir: str, dev_mode: bool = False):
    """
    Process and prepare the turtle dataset under data_dir.

    It prepares train, valid, and test sets for two settings:
    1. the annotations for the whole turtle only
    2. the annotations for flippers, head, and carapace

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
        logger.info(f"Preprocessing and registering for {split_name} data...")

        # Process and register the turtle body parts dataset
        _data_name = f"turtle_parts_{split_name}"
        split_coco_parts = create_split_coco(img_ids, coco)
        
        datasets[_data_name] = load_coco_api(split_coco_parts, data_dir, _data_name)

    return datasets


def register_turtle_coco(dataset: List[dict], dataset_name: str, data_dir: str, meta: dict = {}):
    DatasetCatalog.register(
        dataset_name,
        lambda: dataset
    )
    MetadataCatalog.get(dataset_name).set(
        evaluator_type="coco", image_root=data_dir, **meta
    )


def create_split_coco(img_ids, coco):
    """
    Create a COCO dataset split, processing body parts if specified.

    Args:
        img_ids (list): List of image IDs for the split.
        coco (COCO): Original COCO object.

    Returns:
        COCO: Processed COCO object for the dataset split.
    """
    split_annotations = {}
    for img_id in tqdm(img_ids, desc="Preprocessing bbox and masks"):
        anns = coco.imgToAnns[img_id]

        # Get the original turtle instances, keep them as category 4
        turtle_anns = [copy.deepcopy(ann) for ann in anns if ann["category_id"] == 1]
        for ann in turtle_anns:
            ann["category_id"] = 4  # Set category ID to 4 for "whole_turtle"
        
        # Process annotations for body parts dataset
        processed_anns = process_ann(anns, coco)
        
        # Add the whole turtle annotations to the processed annotations
        processed_anns.extend(turtle_anns)
        
        # Make detectron2 happy
        for _ann in processed_anns:
            _ann["iscrowd"] = 0

        # Add processed annotations to new split dict
        for ann in processed_anns:
            split_annotations[ann["id"]] = ann

    # Update the categories to include "whole_turtle"
    categories = copy.deepcopy(coco.dataset["categories"])
    categories.append({"id": 4, "name": "whole_turtle", "supercategory": ""})

    # Create new COCO object with processed split annotations
    split_coco = COCO()
    split_coco.dataset = {
        "images": [coco.imgs[img_id] for img_id in img_ids],
        "annotations": list(split_annotations.values()),
        "categories": categories
    }
    split_coco.createIndex()
    return split_coco
