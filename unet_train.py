"""
Train and evaluation entry point for U-Net.

Data Preprocessing:
    - Prepocess the original instances (whole turtle, flippers, head) into one mask contains 4 categories
        0) background
        1) carapace
        2) flippers
        3) head
    - entry function: split_n_prepare_turtle_semantic_coco
    - file location: ./detectron2/detectron2/data/datasets/turtle_coco_semantic.py

Model Architecture:
    - configuration file: ./detectron2/configs/Turtle-Semantic/unet-semantic.yaml
    - model class: SemanticSegmentor
    - file location: ./detectron2/detectron2/modeling/meta_arch/semantic_seg.py

    The SemanticSegmentor uses the U-Net at:
    - ./detectron2/detectron2/modeling/backbone/smp.py

Training:
    - Save the best model based on the mIoU metric, and use a customize dataset mapper for semantic segmentation
    - trainer class: TurtleSemanticTrainer
    - file location: ./detectron2/detectron2/engine/trainer.py

    - dataset mapper: TurtleSemanticDatasetMapper
    - file location: ./detectron2/detectron2/data/dataset_mapper.py

Evaluation:
    - Compute the mIoU metric for each body parts and their average
    - Evaluator class: TurtleSemSegEvaluator
    - file location: ./detectron2/detectron2/evaluation/turtle_coco_semantic_evaluation.py
"""

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
import os, json, random

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

def register_dataset(cfg, dev_mode, data_dir):
    base_dir = data_dir

    datasets = split_n_prepare_turtle_semantic_coco(base_dir, dev_mode=dev_mode)

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

    cfg.SOLVER.IMS_PER_BATCH = 16  # This is the real "batch size" commonly known to deep learning people
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
    parser.add_argument('--output_dir', type=str, default='./output_unet',
                       help='Directory for output files')
    parser.add_argument('--data_dir', type=str, default='./turtles-data/data',
                       help='Directory containing the dataset')
    args = parser.parse_args()

    assert not os.path.exists(args.output_dir), f"Output directory {args.output_dir} already exists"
    
    cfg = get_cfg()
    register_dataset(cfg, args.dev, args.data_dir)
    prepare_model(cfg, args.dev, args.output_dir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=False)

    return cfg

if __name__ == '__main__':
    cfg = setup()

    trainer = TurtleSemanticTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")  # path to the model we just trained
    predictor = DefaultPredictor(cfg)

    evaluator = TurtleSemSegEvaluator("turtle_parts_test", output_dir=cfg.OUTPUT_DIR)
    mapper = TurtleSemanticDatasetMapper(cfg, is_train=True)
    tst_loader = build_detection_test_loader(cfg, "turtle_parts_test", mapper=mapper)

    # Dumping the result to a json file
    result = inference_on_dataset(predictor.model, tst_loader, evaluator)
    print(result)
    
    with open(os.path.join(cfg.OUTPUT_DIR, "result.json"), "w") as f:
        json.dump(result, f)