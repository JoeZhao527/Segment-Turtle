# Segment-Turtle

## Authors
Haokai Zhao, Jonas Macken, Liangji Kong, Ruiyuan Yang, Yaqing He

## Project Report
Please find out our project report at [Report.pdf](https://github.com/JoeZhao527/Segment-Turtle/blob/main/Report.pdf).

## Project Structure

### Directories
- `scripts/`: some integrated shell scripts for developing purpose
- `detectron2/`: data processing, model, training and evaluation implementation
- `analysis/`: some notebooks for analysis the model results

### Main Entry (for training and evaluation)
- Mask R-CNN: `mask_rcnn_train.py`
- DPMR: `dual_prop_rcnn.py`
- U-Net: `unet_train.py`
- UFO: `focused_unet_evaluate.py`

## Install Dataset
To setup dataset, download the dataset from kaggle: https://www.kaggle.com/datasets/wildlifedatasets/seaturtleid2022. After unzip the downloaded data, you should see a directory called `turtles-data`. Move that to the root directory of our project.

## Prepare Working Directory
After the dataset and our code is downloaded, we will have a working directory with the dataset directory at `turtles-data`, and our submitted codebase at `Segment-Turtle-main`.

## Enviroment Setup
We used docker to ensure a stable reproduction of our methods. First pull the docker image from pytorch official release:
```
docker pull pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
```

Then start a docker container with the pytorch image. You will need to replace the mounted path with the working directory on your own device:
```
docker run -d --name turtle --gpus all -v /home/haokaizhao/scratch/9517/submit:/workspace -v /dev/shm:/dev/shm pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel tail -f /dev/null
```

Next, we get in to the container to run our models. By correctly mounting the working directory, you should see the same files as in the step for **Prepare Working Directory**
```
docker exec -it turtle /bin/bash
```

Then change the working directory to the project codebase
```
cd Segment-Turtle-main
```

And copy the dataset to the codebase
```
cp -r ../turtles-data ./
```

## Install Dependencies
We used detectron2 for deep learning pipeline framework and Mask R-CNN implementation. We used pytorch-segmentation-model for U-Net implementation. Our DPMR and UFO are developed based on these softwares.

First setup the detectron2 and install its requirements:
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

We have one entry point for each of the Mask R-CNN, DPMR, U-Net and UFO. Notice that UFO relies on the U-Net results, so it has to be ran after U-Net training and evaluation.

Train and evaluate standard Mask R-CNN by running `mask_rcnn_train.py`:
```
python mask_rcnn_train.py \
  --data_dir=./turtles-data/data \
  --output_dir=./output_mask_rcnn
```

Train and evaluate DPMR by running `dual_prop_rcnn.py`
```
python dual_prop_rcnn.py \
  --data_dir=./turtles-data/data \
  --output_dir=./output_dual_prop_rcnn \
  --score_thresh=0.6
```

Train and evaluate standard U-Net by running `unet_train.py`
```
python unet_train.py \
  --data_dir=./turtles-data/data \
  --output_dir=./output_unet
```

Evaluate UFO based on the trained U-Net by running `focused_unet_evaluate.py` (Notice UFO has to be ran after U-Net training to get the trained weights and first stage prediction of the U-Net)
```
python focused_unet_evaluate.py \
  --data_dir=./turtles-data/data \
  --output_dir=./focused_unet_results \
  --model_path=./output_unet/model_best.pth \
  --background_path=./output_unet/sem_seg_predictions.json \
  --focus_crop_ratio=0.1
```

## Output

Each training and evaluation script will output following files in the output directory:

`result.json`: which contains the carapace, flippers and head mIoU, and the average mIoU

`model_best.pth`: which is the best checkpoint selected based on the validation performance. We used that to perform prediction and testing on test set

For Mask R-CNN and DPMR, the prediction results are output in `coco_instances_results.json`:

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

For U-Net and UFO, the prediction results are output in `sem_seg_predictions.json`:
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

## Acknowledgement
This repo is built based on [detectron2](https://github.com/facebookresearch/detectron2).
