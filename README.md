# Segment-Turtle

## Installation

### Setup the codebase
Change your current working directory to the same directory with this README. Run the following command to find the entry points of all our 4 methods. If you got no error, then you are in the correct directory!
```
find mask_rcnn_train.py dual_prop_rcnn.py unet_train.py focused_unet_evaluate.py
```

### Install Dataset
To setup dataset, download the dataset from kaggle: https://www.kaggle.com/datasets/wildlifedatasets/seaturtleid2022. After unzip the downloaded data, you should see a directory called `turtles-data`. Move that to the root directory of our project.

All of our methods accept an input of `data_dir`. The input `data_dir` is expected to look like the following:
```
# Run this command to check if the data directory is correct. Assume the downloaded data is under ./turtles-data
ls ./turtles-data/data

# If you see the following output, then the dataset setup is complete
ls ./turtles-data/data
annotations.json  images  metadata.csv  metadata_splits.csv
```

## Output

For each training model, there are following files in the output directory:

`metrics.json`: Contains the losses and intermediate validation mIoU during training process

`model_best.pth`: The best checkpoint selected based on the validation performance. We used that to perform prediction and testing on test set

`coco_instances_results.json` (mask2former, mask-rcnn): contains the prediction results for the test set, based on the best model checkpoint during training. See `instance_analysis.ipynb` for an demo of processing the file, evaluation and analysis. The json is structured as following:

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
