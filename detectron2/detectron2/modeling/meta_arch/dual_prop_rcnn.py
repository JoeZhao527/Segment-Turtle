import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances
from detectron2.structures.boxes import Boxes, pairwise_intersection
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
from .rcnn import ProposalNetwork, GeneralizedRCNN


__all__ = ["DualProposalRCNNSingleHead"]


@META_ARCH_REGISTRY.register()
class DualProposalRCNNSingleHead(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        sub_instance_intersection_threshold: float = 0.7  # New threshold parameter
    ):
        """
        Args:
            backbone, proposal_generator, roi_heads, pixel_mean, pixel_std,
            input_format, vis_period: (Same as GeneralizedRCNN)

            sub_instance_intersection_threshold: float, threshold for the 
                intersection ratio of sub-instances with super-instances.
                Retain sub-instances only if intersection ratio exceeds this.
        """
        super().__init__(
            backbone=backbone,
            proposal_generator=proposal_generator,
            roi_heads=roi_heads,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            input_format=input_format,
            vis_period=vis_period,
        )
        self.sub_instance_intersection_threshold = sub_instance_intersection_threshold  # Set the threshold

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "sub_instance_intersection_threshold": cfg.MODEL.SUB_INSTANCE_INTERSECTION_THRESHOLD,  # Read threshold from config
        }

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes, threshold: float):
        """
        Rescale the output instances to the target size, keeping sub-instances
        only if the intersection with a super-instance is above the threshold
        of the sub-instance's bounding box area.
        """
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)

            # Separate instances by category: super-category (3) and sub-categories (0, 1, 2)
            super_category_mask = (r.pred_classes == 3)
            sub_category_mask = ~super_category_mask
            
            super_instances = r[super_category_mask]
            sub_instances = r[sub_category_mask]
            
            # If there are no super-category instances, skip filtering
            if len(super_instances) > 0 and len(sub_instances) > 0:
                # Ensure pred_boxes are in the Boxes format
                super_boxes = Boxes(super_instances.pred_boxes.tensor)
                sub_boxes = Boxes(sub_instances.pred_boxes.tensor)

                # Calculate pairwise intersections between all sub and super boxes
                intersection_matrix = pairwise_intersection(sub_boxes, super_boxes)
                sub_areas = sub_boxes.area()  # [N] tensor of sub-instance areas
                
                # Calculate intersection ratios for each sub-instance with each super-instance
                intersection_ratios = intersection_matrix / sub_areas[:, None]  # [N, M] matrix
                
                # Determine which sub-instances have a sufficient intersection with any super-instance
                valid_sub_instance_mask = (intersection_ratios > threshold).any(dim=1)

                # Filter the sub-instances that meet the intersection ratio criterion
                filtered_sub_instances = sub_instances[valid_sub_instance_mask]

                # Combine the super-instances with the filtered sub-instances
                r = Instances.cat([super_instances, filtered_sub_instances])

            # Add to processed results
            processed_results.append({"instances": r})

        return processed_results

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return DualProposalRCNNSingleHead._postprocess(results, batched_inputs, images.image_sizes, self.sub_instance_intersection_threshold)
        return results
