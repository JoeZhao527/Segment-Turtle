from detectron2.engine import DefaultTrainer, hooks
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import os

class BestCheckpointer(hooks.BestCheckpointer):
    def __init__(self, eval_period, checkpointer, val_metric, mode):
        super().__init__(eval_period, checkpointer, val_metric, mode)

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, BestCheckpointer(
            self.cfg.TEST.EVAL_PERIOD,
            self.checkpointer,
            "bbox/AP50",
            mode="max",
        ))
        return hooks

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        if isinstance(evaluators, str):
            evaluators = [evaluators]
        if evaluators is None:
            evaluators = [cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference", name)
            ) for name in cfg.DATASETS.TEST]
        res = {}
        for evaluator in evaluators:
            results = inference_on_dataset(
                model, 
                build_detection_test_loader(cfg, evaluator.dataset_name),
                evaluator
            )
            res.update(results)
        return res

