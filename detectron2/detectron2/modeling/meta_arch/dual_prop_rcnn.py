import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
from .rcnn import ProposalNetwork, GeneralizedRCNN


__all__ = ["DualProposalRCNN"]


@META_ARCH_REGISTRY.register()
class DualProposalRCNN(GeneralizedRCNN):
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

            # Separate instances by category: super-category (4) and sub-categories (1, 2, 3)
            super_category_mask = (r.pred_classes == 4)
            sub_category_mask = ~super_category_mask

            super_instances = r[super_category_mask]
            sub_instances = r[sub_category_mask]

            # If there are no super-category instances, skip filtering
            if len(super_instances) > 0:
                # Create a mask to filter out sub-instances based on intersection ratio
                valid_sub_instance_mask = torch.zeros(len(sub_instances), dtype=torch.bool, device=r.pred_boxes.device)

                # Check each sub-instance bounding box for intersection with any super-instance bounding box
                for i, sub_box in enumerate(sub_instances.pred_boxes):
                    sub_box_area = sub_box.area().item()
                    keep_sub_instance = False

                    for super_box in super_instances.pred_boxes:
                        intersection_area = sub_box.intersection(super_box).area().item()

                        # Calculate intersection ratio
                        intersection_ratio = intersection_area / sub_box_area if sub_box_area > 0 else 1.0
                        
                        # Keep the sub-instance if intersection ratio exceeds the threshold
                        if intersection_ratio > threshold:
                            keep_sub_instance = True
                            break  # No need to check other super boxes once condition is met

                    # Update the mask based on whether the sub-instance meets the criterion
                    valid_sub_instance_mask[i] = keep_sub_instance

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
            return DualProposalRCNN._postprocess(results, batched_inputs, images.image_sizes, self.sub_instance_intersection_threshold)
        return results
