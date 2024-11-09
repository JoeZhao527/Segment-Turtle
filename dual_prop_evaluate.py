# Some basic setup:
import sys, os, distutils.core
sys.path.insert(0, os.path.abspath('./detectron2'))

import torch

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine.trainer import CustomTrainer
from detectron2.config import get_cfg

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.evaluation.turtle_coco_evaluation import TurtleCOCOEvaluator
from detectron2.data import build_detection_test_loader

from detectron2.data.datasets.turtle_dual_proposal_rcnn import split_n_prepare_turtle_coco, register_turtle_coco

import argparse

def register_dataset(cfg, dev_mode):
    base_dir = "./turtles-data/data"

    datasets = split_n_prepare_turtle_coco(base_dir, dev_mode=dev_mode)

    for _name, _data in datasets.items():
        register_turtle_coco(_data, _name, base_dir)
    
    cfg.DATASETS.TRAIN = ("turtle_parts_train",)
    cfg.DATASETS.TEST = ("turtle_parts_valid",)


def prepare_model(cfg, dev_mode, output_dir):
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/dual_proposal_rcnn.yaml"))

    # Switch the valid and test split, as the detectron2 uses cfg.DATASETS.TEST for validation
    cfg.DATASETS.TRAIN = ("turtle_parts_train",)
    cfg.DATASETS.TEST = ("turtle_parts_valid",)

    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml")

    cfg.SOLVER.IMS_PER_BATCH = 4  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    if dev_mode:
        cfg.SOLVER.MAX_ITER = 40
        cfg.TEST.EVAL_PERIOD = 20
    else:
        cfg.SOLVER.MAX_ITER = 20000
        cfg.TEST.EVAL_PERIOD = 1000
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.MODEL.SUB_INSTANCE_INTERSECTION_THRESHOLD = 0.7
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.OUTPUT_DIR = output_dir
    
def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true', help='Enable development mode')
    parser.add_argument('--output_dir', type=str, default='./evaluate_dual_prop_rcnn',
                       help='Directory for output files')
    parser.add_argument('--model_path', type=str, default='./output_dual_prop_rcnn/model_best.pth',
                       help='Path to the model weights')
    parser.add_argument('--score_thresh', type=float, default=0.7,
                       help='Score threshold for evaluation')
    args = parser.parse_args()
    
    output_dir = os.path.join(args.output_dir, f"thr_{args.score_thresh.replace('.', '_')}")
    assert not os.path.exists(output_dir), f"Output directory {output_dir} already exists"

    cfg = get_cfg()
    register_dataset(cfg, args.dev)
    prepare_model(cfg, args.dev, output_dir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=False)

    cfg.MODEL.WEIGHTS = args.model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    return cfg

if __name__ == '__main__':
    cfg = setup()

    predictor = DefaultPredictor(cfg)

    evaluator = TurtleCOCOEvaluator("turtle_parts_test", output_dir=cfg.OUTPUT_DIR)
    tst_loader = build_detection_test_loader(cfg, "turtle_parts_test")
    print(inference_on_dataset(predictor.model, tst_loader, evaluator))