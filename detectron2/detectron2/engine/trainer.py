from detectron2.engine import DefaultTrainer, hooks
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.evaluation.turtle_coco_evaluation import TurtleCOCOEvaluator
from detectron2.evaluation.turtle_coco_semantic_evaluation import TurtleSemSegEvaluator
from detectron2.evaluation.sem_seg_evaluation import SemSegEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data.dataset_mapper import TurtleSemanticDatasetMapper
import detectron2.data.transforms as T
import os

class BestCheckpointer(hooks.BestCheckpointer):
    def __init__(self, eval_period, checkpointer, val_metric, mode):
        super().__init__(eval_period, checkpointer, val_metric, mode)

class CustomTrainer(DefaultTrainer):
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, BestCheckpointer(
            self.cfg.TEST.EVAL_PERIOD,
            self.checkpointer,
            "average_miou",
            mode="max",
        ))
        return hooks
    
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        return super().test(cfg, model, evaluators=TurtleCOCOEvaluator(
            cfg.DATASETS.TEST[0], output_dir=cfg.OUTPUT_DIR
        ))


def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN,
            cfg.INPUT.MAX_SIZE_TRAIN,
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    augs.append(T.RandomFlip())
    return augs


class TurtleSemanticTrainer(DefaultTrainer):
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, BestCheckpointer(
            self.cfg.TEST.EVAL_PERIOD,
            self.checkpointer,
            "average_miou",
            mode="max",
        ))
        return hooks
    
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = TurtleSemanticDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def test(cls, cfg, model, evaluator=None):
        return super().test(cfg, model, evaluators=TurtleSemSegEvaluator(
            cfg.DATASETS.TEST[0], output_dir=cfg.OUTPUT_DIR
        ))