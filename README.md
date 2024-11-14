# Segment-Turtle

## Installation

### Install Dataset
To setup dataset, download the dataset from kaggle: https://www.kaggle.com/datasets/wildlifedatasets/seaturtleid2022. After unzip the downloaded data, you should see a directory called `turtles-data`. Move that to the root directory of our project.

### Install Dependencies

#### Setup segmentation-models-pytorch (for native U-Net implementation) and other dependencies
```
pip install -r requirements.txt
```

#### Setup detectron2 (for training / evaluation loop and native Mask R-CNN implementation):
```
python ./scripts/detectron2_setup.py
```

## Quick Testing
Run the following command for a quick test of all methods. This should takes no more than 10 minutes:
```
./scripts/deployment_test.sh
```

## Training and Evaluation

1. Train and evaluate native Mask R-CNN:
```
python mask_rcnn_train.py \
  --data_dir=./turtles-data/data \
  --output_dir=./output_mask_rcnn
```

2. Train and evaluate Daul Proposal Mask R-CNN (DPMR)
```
python dual_prop_rcnn.py \
  --data_dir=./turtles-data/data \
  --output_dir=./output_dual_prop_rcnn
```

3. Train and evaluate native U-Net
```
python unet_train.py \
  --data_dir=./turtles-data/data \
  --output_dir=./output_unet
```

4. Evaluate U-Net on Focus (UFO) with the trained U-Net (Notice: this have to be ran after running 3. to get the trained weights and first stage prediction of the U-Net)
```
python focused_unet_evaluate.py \
  --data_dir=./turtles-data/data \
  --output_dir=./focused_unet_results \
  --model_path=./output_unet/model_best.pth \
  --background_path=./output_unet/sem_seg_predictions.json \
  --focus_crop_ratio=0.1
```

## Output

For each training model, there are following files in the output directory:

`metrics.json`: Contains the losses and intermediate validation mIoU during training process

`model_best.pth`: The best checkpoint selected based on the validation performance. We used that to perform prediction and testing on test set

`coco_instances_results.json` (mask-rcnn): contains the prediction results for the test set, based on the best model checkpoint during training. See `instance_analysis.ipynb` for an demo of processing the file, evaluation and analysis. The json is structured as following:

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

`coco_instances_results.json` (unet): contains the prediction results for the test set, based on the best model checkpoint during training. The content is structured as following:

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
