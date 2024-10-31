import sys, os, distutils.core
sys.path.insert(0, os.path.abspath('./detectron2'))

import os
import glob
import json
import pandas as pd 
from pycocotools.coco import COCO
from PIL import Image
import numpy as np 
import skimage.io as io
from matplotlib import pyplot as plt
import random
from pprint import pprint

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import (
    default_setup,
    default_argument_parser
)
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from mask2former.trainer import Trainer
from mask2former import add_maskformer2_config

from detectron2.data.datasets.turtle_coco import split_n_prepare_turtle_coco, register_turtle_coco

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.evaluation.turtle_coco_evaluation import TurtleCOCOEvaluator
from detectron2.data import build_detection_test_loader

def register_dataset(cfg):
    base_dir = "./turtles-data/data"

    datasets = split_n_prepare_turtle_coco(base_dir, dev_mode=False)

    for _name, _data in datasets.items():
        register_turtle_coco(_data, _name, base_dir)
    
    cfg.DATASETS.TRAIN = ("turtle_parts_train",)
    cfg.DATASETS.TEST = ("turtle_parts_valid",)

def prepare_model(cfg):
    cfg.merge_from_file("./detectron2/configs/COCO-Mask2former/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml")
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl"
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = 3
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3
    cfg.MODEL.RETINANET.NUM_CLASSES = 3
    cfg.OUTPUT_DIR = "./output_mask2former"
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10000
    cfg.TEST.EVAL_PERIOD = 1000

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    prepare_model(cfg)
    register_dataset(cfg)
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    
    return cfg


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    cfg = setup(args)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    evaluator = TurtleCOCOEvaluator("turtle_parts_test", output_dir=cfg.OUTPUT_DIR)
    tst_loader = build_detection_test_loader(cfg, "turtle_parts_test")
    print(inference_on_dataset(predictor.model, tst_loader, evaluator))