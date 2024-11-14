# Some basic setup:
import sys, os, distutils.core
sys.path.insert(0, os.path.abspath('./detectron2'))

import torch

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine.trainer import CustomTrainer
from detectron2.config import get_cfg

from detectron2.evaluation import inference_on_dataset
from detectron2.evaluation.turtle_coco_evaluation import TurtleCOCOEvaluator
from detectron2.data import build_detection_test_loader

import cv2
import json
import argparse  # Add this import at the top

def prepare_model(cfg, dev_mode, output_dir):
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"))

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
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.OUTPUT_DIR = output_dir
    
def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true', help='Enable development mode')
    parser.add_argument('--output_dir', type=str, default='./predict_mask_rcnn',
                       help='Directory for output files')
    parser.add_argument('--data_path', type=str, default='./turtles-data/data/images/t001/anuJvqUqBB.JPG',
                       help='Path to the image to be predicted')
    args = parser.parse_args()
    
    cfg = get_cfg()
    prepare_model(cfg, args.dev, args.output_dir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=False)

    return cfg, args

if __name__ == '__main__':
    cfg, args = setup()
    
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")  # path to the model we just trained
    predictor = DefaultPredictor(cfg)

    # Load image and run prediction
    image = cv2.imread(args.data_path)
    image_name = args.data_path.split('/')[-1].split('.')[0]
    outputs = predictor(image)

    # Save the results
    torch.save(outputs, os.path.join(cfg.OUTPUT_DIR, f"{image_name}.pth"))
    