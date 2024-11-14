# Segment-Turtle

## Install Dataset
To setup dataset, download the dataset from kaggle: https://www.kaggle.com/datasets/wildlifedatasets/seaturtleid2022. After unzip the downloaded data, you should see a directory called `turtles-data`. Move that to the root directory of our project.

## Prepare Working Directory
After the data and code is set up, the working directory should have four files:
`Segment-Turtle-main.zip`: our submitted code
`seaturtleid2022.zip`: dataset zip file
`turtles-data`: unzipped dataset
`license.txt`: dataset license

## Enviroment Setup
We used docker to ensure a stable reproduction of our methods. First pull the docker image from pytorch official release:
```
docker pull pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
```

Then start a docker container with the pytorch image. You will need to replace the mounted path with the working directory on your own device:
```
docker run -d --name turtle --gpus all -v /home/haokaizhao/scratch/9517/submit:/workspace pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel tail -f /dev/null
```

Next, we get in to the container to run our models. By correctly mounting the working directory, you should see the same files as in the step for **Prepare Working Directory**
```
docker exec -it turtle /bin/bash
```

## Install Dependencies
We used detectron2 for deep learning pipeline framework and Mask R-CNN implementation. We used pytorch-segmentation-model for U-Net implementation. Our DPMR and UFO are developed based on these softwares.

First setup the detectron2 and install its dependencies:
```
python ./scripts/detectron2_setup.py
```

Then install other requirements:
```
pip install -r requirements.txt
```

## Quick Testing
Now we are ready to train and evaluate the models. Run `./scripts/deployment_test.sh` for a quick test on the whole pipeline for all 4 methods with only 140 samples:
```
./scripts/deployment_test.sh
```

## Training and Evaluation

We have one entry point for each of Mask R-CNN, DPMR, U-Net and UFO. Notice that UFO relies on U-Net results, so it has to be ran after U-Net training and evaluation.

1. Train and evaluate standard Mask R-CNN:
```
python mask_rcnn_train.py \
  --data_dir=./turtles-data/data \
  --output_dir=./output_mask_rcnn
```

2. Train and evaluate Daul Proposal Mask R-CNN (DPMR)
```
python dual_prop_rcnn.py \
  --data_dir=./turtles-data/data \
  --output_dir=./output_dual_prop_rcnn \
  --score_thresh=0.6
```

3. Train and evaluate standard U-Net
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
