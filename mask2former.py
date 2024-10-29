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

import torch.version

import sys, os, distutils.core
sys.path.insert(0, os.path.abspath('./detectron2'))
sys.path.insert(0, os.path.abspath('./Mask2Former'))

import torch, detectron2
torch_version = ".".join(torch.__version__.split(".")[:2])

if torch.version.cuda:
    cuda_version = ''.join(torch.version.cuda.split("."))
else:
    cuda_version = 'none'

print("torch: ", torch_version, "; cuda: ", cuda_version)
print("detectron2:", detectron2.__version__)

from detectron2.utils.logger import setup_logger
setup_logger()
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="mask2former")

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config

# import Mask2Former project
from mask2former import add_maskformer2_config
from detectron2.data.datasets import register_coco_instances

base_dir = "./turtles-data/data"
register_coco_instances("sea_turtle_train", {}, os.path.join(base_dir, "annotations_train.json"), "./turtles-data/data")
register_coco_instances("sea_turtle_valid", {}, os.path.join(base_dir, "annotations_valid.json"), "./turtles-data/data")
register_coco_instances("sea_turtle_test", {}, os.path.join(base_dir, "annotations_test.json"), "./turtles-data/data")

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)

# Ensure the config file path corresponds to your model's requirements
cfg.merge_from_file("./Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_base_384_bs16_50ep.yaml")
# cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'
cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_base_384_bs16_50ep/model_final_f6e0f6.pkl'
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode, BitMasks, Instances
import numpy as np


# %%
from detectron2 import model_zoo
from detectron2.config import get_cfg
from mask2former.config import add_maskformer2_config
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
import torch
from detectron2.utils.events import EventStorage
from torch.utils.tensorboard import SummaryWriter

class Trainer(DefaultTrainer):
    """Custom trainer with evaluation"""
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

def setup_cfg():
    #print(cfg.dump())
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    add_deeplab_config(cfg)      # Add ResNet/DeepLab configs

    # Load base Mask2Former configuration
    cfg.merge_from_file("./Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_base_384_bs16_50ep.yaml")
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_base_384_bs16_50ep/model_final_f6e0f6.pkl'
    # Dataset settings
    cfg.DATASETS.TRAIN = ("sea_turtle_train",)
    cfg.DATASETS.TEST = ("sea_turtle_valid",)
    
    # Model settings
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3  # turtle, flipper, head
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
    
    # Training parameters - conservative settings for fine-tuning
    cfg.SOLVER.IMS_PER_BATCH = 2  # Adjust based on your GPU memory
    cfg.SOLVER.BASE_LR = 0.00005  # Small learning rate for fine-tuning
    cfg.SOLVER.MAX_ITER = 5000    # Adjust based on your dataset size
    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    
    # Learning rate scheduler
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    cfg.SOLVER.STEPS = (3000, 4500)  # Steps to decrease learning rate
    cfg.SOLVER.GAMMA = 0.1  # Learning rate decay factor
    
    # Gradient clipping settings
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"  # Use "norm" for clipping
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0  # Clipping value

    
    # Validation period
    cfg.TEST.EVAL_PERIOD = 500  # Validate every 500 iterations
    
    # Data augmentation (mild for fine-tuning)
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.RANDOM_FLIP = "horizontal"
    
    # Set training device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Output directory
    cfg.OUTPUT_DIR = "./output_maskformer_pretrained"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg

def train_model():
    cfg = setup_cfg()

    # Print training configuration
    print("\nTraining Configuration:")
    print(f"Device: {cfg.MODEL.DEVICE}")
    print(f"Learning Rate: {cfg.SOLVER.BASE_LR}")
    print(f"Max Iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"Batch Size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"Number of Classes: {cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES}")
    print(f"Pre-trained Weights: {cfg.MODEL.WEIGHTS}")
    print(f"Output Directory: {cfg.OUTPUT_DIR}")
    
    # Setup tensorboard
    writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'tensorboard'))
    
    # Initialize trainer
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    
    print("\nStarting training...")
    
    # Training loop with monitoring
    try:
        with EventStorage(start_iter=0):
            trainer.train()
    except Exception as e:
        print(f"Training interrupted: {str(e)}")
    finally:
        writer.close()

# Function to test the model
def test_model(cfg, dataset):
    from detectron2.engine import DefaultPredictor
    
    # Load the trained model
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)
    
    # Run evaluation
    evaluator = COCOEvaluator(
        cfg.DATASETS.TEST[0],
        output_dir=os.path.join(cfg.OUTPUT_DIR, "final_evaluation")
    )
    
    print("\nRunning final evaluation...")
    evaluator.reset()
    
    for i in range(min(5, len(dataset))):  # Test on first 5 images
        image, target = dataset[i]
        
        # Prepare image for inference
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)
        
        # Run inference
        outputs = predictor(image)
        
        # Visualize predictions
        #visualize_predictions(image, outputs)
        
        # Print prediction info
        instances = outputs["instances"].to("cpu")
        print(f"\nPredictions for image {i}:")
        print(f"Number of instances: {len(instances)}")
        print(f"Predicted classes: {instances.pred_classes}")
        print(f"Scores: {instances.scores}")

if __name__ == "__main__":
    # Start training
    train_model()
    
    # After training, test the model
    cfg = setup_cfg()
    test_model(cfg, valid_dataset)
