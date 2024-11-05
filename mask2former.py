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

def register_dataset(cfg, dev_mode, base_dir):
    datasets = split_n_prepare_turtle_coco(base_dir, dev_mode=dev_mode)

    for _name, _data in datasets.items():
        register_turtle_coco(_data, _name, base_dir)
    
    cfg.DATASETS.TRAIN = ("turtle_parts_train",)
    cfg.DATASETS.TEST = ("turtle_parts_valid",)

def prepare_model(cfg, dev_mode, output_dir):
    cfg.merge_from_file("./detectron2/configs/COCO-Mask2former/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml")
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl"
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = 3
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3
    cfg.MODEL.RETINANET.NUM_CLASSES = 3
    cfg.OUTPUT_DIR = output_dir
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    if dev_mode:
        cfg.SOLVER.MAX_ITER = 40
        cfg.TEST.EVAL_PERIOD = 20
    else:
        cfg.SOLVER.MAX_ITER = 20000
        cfg.TEST.EVAL_PERIOD = 1000

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    prepare_model(cfg, args.dev, args.output_dir)
    register_dataset(cfg, args.dev, args.data_dir)
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    
    return cfg

if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--dev', action='store_true', help='Enable development mode')
    parser.add_argument('--output_dir', type=str, default='./output_mask2former',
                       help='Directory for output files')
    parser.add_argument('--data_dir', type=str, default='./turtles-data/data',
                       help='Directory containing the dataset')
    args = parser.parse_args()
    
    assert not os.path.exists(args.output_dir), f"Output directory {args.output_dir} already exists"
    assert os.path.exists(args.data_dir), f"Data directory {args.data_dir} does not exist"

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