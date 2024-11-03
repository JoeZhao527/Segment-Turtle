# Segment-Turtle

## Installation

TODO

## Output

For each training model, there are following files in the output directory:

`metrics.json`: Contains the losses and intermediate validation mIoU during training process

`model_best.pth`: The best checkpoint selected based on the validation performance. We used that to perform prediction and testing on test set

`coco_instances_results.json` (mask2former, mask-rcnn): contains the prediction results for the test set, based on the best model checkpoint during training. The content is structured as following:

```python
{
  # Image id
  "666": {
    # Predicted mask for carapace
    "1": {
      "size": [1333, 2000],
      "counts": "encoded prediction mask string"
    },
    # Predicted mask for flippers
    "2": {
      "size": [1333, 2000],
      "counts": "encoded prediction mask string"
    },
    # Predicted mask for head
    "3": {
      "size": [1333, 2000],
      "counts": "encoded prediction mask string"
    }
  },
  ...
}
```

`coco_instances_results.json` (unet, deeplab, fcn): contains the prediction results for the test set, based on the best model checkpoint during training. The content is structured as following:

```python
{
    # Image id
    "122":{
        # Preprocessed ground truth mask. each pixel is labeled as one of (0: background, 1: carapace, 2: flippers, 3: head)
        "gt": {"size": [1333, 2000], "counts": "encoded prediction mask string"},
        # Predicted mask
        "pred": {"size": [1333, 2000], "counts": "encoded prediction mask string"},
        "turtle_iou": 0.9662467918411455,
        "flippers_iou": 0.8998383719899341,
        "head_iou": 0.7218838319851866
    },
    ...
}
```
