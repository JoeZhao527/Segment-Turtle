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

def load_coco_api(coco_api, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a COCO object and convert it to a list of dict in instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        coco_api (COCO): COCO object in instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
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

    for img_dict, anno_dict_list in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
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
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs
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
        - Whole turtle:
            turtle_whole_train
            turtle_whole_valid
            turtle_whole_test

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

        # Process and register the whole turtle dataset
        _data_name = f"turtle_whole_{split_name}"
        split_coco_whole = create_split_coco(img_ids, coco, process_body_parts=False)
        datasets[_data_name] = load_coco_api(split_coco_whole, data_dir, _data_name)

        # Process and register the turtle body parts dataset
        _data_name = f"turtle_parts_{split_name}"
        split_coco_parts = create_split_coco(img_ids, coco, process_body_parts=True)
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
    for img_id in tqdm(img_ids, desc="Preprocessing bbox and masks"):
        anns = coco.imgToAnns[img_id]

        if process_body_parts:
            # Process annotations for body parts dataset
            processed_anns = process_ann(anns, coco)
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


def process_ann(anns, coco):
    """
    Process a list of annotations for one image, modifying each instance mask of category 1
    by excluding areas covered by category 2 and category 3 masks.
    
    Args:
        anns (list): List of annotations for a single image.

    Returns:
        list: Updated list of annotations.
    """
    # Initialize combined masks for categories 2 and 3
    combined_cat_2_mask = None
    combined_cat_3_mask = None

    # Combine all masks for categories 2 and 3 to exclude them from each instance of category 1
    for ann in anns:
        mask = coco.annToMask(ann)
        if ann["category_id"] == 2:
            combined_cat_2_mask = mask if combined_cat_2_mask is None else np.logical_or(combined_cat_2_mask, mask)
        elif ann["category_id"] == 3:
            combined_cat_3_mask = mask if combined_cat_3_mask is None else np.logical_or(combined_cat_3_mask, mask)

    # Process each instance of category 1 separately
    for ann in anns:
        if ann["category_id"] == 1:
            # Decode the current category 1 mask
            cat_1_mask = coco.annToMask(ann)

            # Subtract combined masks of categories 2 and 3 from the current category 1 instance
            if combined_cat_2_mask is not None:
                cat_1_mask = np.logical_and(cat_1_mask, np.logical_not(combined_cat_2_mask))
            if combined_cat_3_mask is not None:
                cat_1_mask = np.logical_and(cat_1_mask, np.logical_not(combined_cat_3_mask))

            # Convert modified mask back to RLE format
            cat_1_rle = mask_util.encode(np.asfortranarray(cat_1_mask.astype(np.uint8)))

            # Update segmentation mask for the current annotation
            ann["segmentation"] = cat_1_rle

            # Update bounding box based on the modified mask
            ys, xs = np.where(cat_1_mask)
            if ys.size > 0 and xs.size > 0:
                x_min, y_min, x_max, y_max = xs.min(), ys.min(), xs.max(), ys.max()
                ann["bbox"] = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                # Set bbox to empty if mask is empty after processing
                ann["bbox"] = [0, 0, 0, 0]

    return anns
