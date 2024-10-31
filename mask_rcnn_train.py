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
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.evaluation.turtle_coco_evaluation import TurtleCOCOEvaluator
from detectron2.data import build_detection_test_loader

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.turtle_coco import split_n_prepare_turtle_coco, register_turtle_coco

base_dir = "./turtles-data/data"
# register_coco_instances("turtle_train", {}, os.path.join(base_dir, "annotations_train.json"), "./turtles-data/data")
# register_coco_instances("turtle_val", {}, os.path.join(base_dir, "annotations_valid.json"), "./turtles-data/data")
# register_coco_instances("turtle_test", {}, os.path.join(base_dir, "annotations_test.json"), "./turtles-data/data")

datasets = split_n_prepare_turtle_coco(base_dir, dev_mode=True)

for _name, _data in datasets.items():
    register_turtle_coco(_data, _name, base_dir)

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml"))
# cfg.DATASETS.TRAIN = ("turtle_train",)
# cfg.DATASETS.VALID = ("turtle_val",)
# cfg.DATASETS.TEST = ("turtle_test",)

# Switch the valid and test split, as the detectron2 uses cfg.DATASETS.TEST for validation
cfg.DATASETS.TRAIN = ("turtle_parts_train",)
cfg.DATASETS.TEST = ("turtle_parts_valid",)

cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml")
# cfg.MODEL.WEIGHTS = ""
cfg.SOLVER.IMS_PER_BATCH = 4  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg.TEST.EVAL_PERIOD = 100
cfg.INPUT.MASK_FORMAT = 'bitmask'
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
print(cfg)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg) 
trainer = CustomTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

evaluator = TurtleCOCOEvaluator("turtle_parts_test", output_dir="./output")
tst_loader = build_detection_test_loader(cfg, "turtle_parts_test")
print(inference_on_dataset(predictor.model, tst_loader, evaluator))