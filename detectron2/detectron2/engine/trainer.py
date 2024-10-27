from detectron2.engine import DefaultTrainer, hooks
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
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
            "bbox/AP50",
            mode="max",
        ))
        return hooks

