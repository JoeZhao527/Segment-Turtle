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
from detectron2.engine.trainer import CustomTrainer, TurtleSemanticTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.dataset_mapper import TurtleSemanticDatasetMapper

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.evaluation.turtle_coco_evaluation import TurtleCOCOEvaluator
from detectron2.evaluation.turtle_coco_semantic_evaluation import TurtleSemSegEvaluator
from detectron2.data import build_detection_test_loader

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.turtle_coco_semantic import register_turtle_coco, split_n_prepare_turtle_semantic_coco

import argparse

def register_dataset(cfg, dev_mode):
    base_dir = "./turtles-data/data"

    datasets = split_n_prepare_turtle_semantic_coco(base_dir, dev_mode=dev_mode, test_only=True)

    for _name, _data in datasets.items():
        register_turtle_coco(_data, _name, base_dir)
    
    cfg.DATASETS.TRAIN = ("turtle_parts_train",)
    cfg.DATASETS.TEST = ("turtle_parts_valid",)


def prepare_model(cfg, dev_mode, output_dir):
    cfg.merge_from_file(model_zoo.get_config_file("Turtle-Semantic/unet-semantic.yaml"))

    cfg.MODEL.BACKBONE.ENCODER_NAME = "resnet34"
    cfg.MODEL.BACKBONE.WEIGHTS = "imagenet"
    cfg.MODEL.BACKBONE.IN_CHANNELS = 3
    cfg.MODEL.BACKBONE.NUM_CLASSES = 4
    cfg.MODEL.BACKBONE.SIZE_DIVISIBILITY = 32

    # Switch the valid and test split, as the detectron2 uses cfg.DATASETS.TEST for validation
    cfg.DATASETS.TRAIN = ("turtle_parts_train",)
    cfg.DATASETS.TEST = ("turtle_parts_valid",)

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = 4  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    if dev_mode:
        cfg.SOLVER.MAX_ITER = 40
        cfg.TEST.EVAL_PERIOD = 20
    else:
        cfg.SOLVER.MAX_ITER = 20000
        cfg.TEST.EVAL_PERIOD = 1000
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.OUTPUT_DIR = output_dir
    
def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true', help='Enable development mode')
    parser.add_argument('--output_dir', type=str, default='./evaluate_unet',
                       help='Directory for output files')
    parser.add_argument('--model_path', type=str, default='./output_unet/model_best.pth',
                       help='Path to the model weights')
    args = parser.parse_args()

    assert not os.path.exists(args.output_dir), f"Output directory {args.output_dir} already exists"
    
    cfg = get_cfg()
    register_dataset(cfg, args.dev)
    prepare_model(cfg, args.dev, args.output_dir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=False)

    cfg.MODEL.WEIGHTS = args.model_path
    return cfg

if __name__ == '__main__':
    cfg = setup()

    predictor = DefaultPredictor(cfg)

    evaluator = TurtleSemSegEvaluator("turtle_parts_test", output_dir=cfg.OUTPUT_DIR)
    mapper = TurtleSemanticDatasetMapper(cfg, is_train=False)
    tst_loader = build_detection_test_loader(cfg, "turtle_parts_test", mapper=mapper)
    print(inference_on_dataset(predictor.model, tst_loader, evaluator))