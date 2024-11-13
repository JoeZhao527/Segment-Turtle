# Segment-Turtle

## Installation

### Software Enviroment
We developed and tested our methods with pytroch 2.0.0 official docker image (https://hub.docker.com/layers/pytorch/pytorch/2.0.0-cuda11.7-cudnn8-devel/images/sha256-96ccb2997a131f2455d70fb78dbb284bafe4529aaf265e344bae932c8b32b2a4?context=explore).

We strongly recommend you to setup the docker container for a stable enviroment. You will need to mount the paths and gpus approriately on your own devices if you use docker container.

### Setup the codebase
Change your current working directory to the same directory with this README. Run the following command to find the entry points of all our 4 methods. If you got no error, then you are in the correct directory!
```
find mask_rcnn_train.py dual_prop_rcnn.py unet_train.py focused_unet_evaluate.py
```

### Install Dataset
To setup dataset, download the dataset from kaggle: https://www.kaggle.com/datasets/wildlifedatasets/seaturtleid2022. After unzip the downloaded data, you should see a directory called `turtles-data`. Move that to the root directory of our project.

All of our methods accept an input of `data_dir`. The input `data_dir` can be verified as the following:

Run this command to check if the data directory is correct. Assume the downloaded data is under `./turtles-data`
```
ls ./turtles-data/data
```

If you see the following output, then the dataset setup is complete
```
ls ./turtles-data/data
annotations.json  images  metadata.csv  metadata_splits.csv
```

### Install Dependencies

#### Setup segmentation-models-pytorch (for native U-Net implementation) and other dependencies
```
pip install -r requirements.txt
```

#### Setup detectron2 (for training / evaluation loop and native Mask R-CNN implementation):
```
python ./scripts/detectron2_setup.py
```

#### Setup opencv-python

If you are using the docker container:
```
pip install opencv-python-headless
```

If you are using standard desktop environments
```
pip install opencv-python
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
