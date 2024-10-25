# Segment-Turtle

## Installation

Install Requirements
```
pip install -r requriements
```

Clone dependent repo
```
git clone https://github.com/facebookresearch/detectron2
git clone https://github.com/facebookresearch/Mask2Former.git
```

Set up detectron2
```
python scripts/detectron2_setup.py
```

Set up mask2former
```
./scripts/mask2former_setup.sh
```

Train detectron2
```
python scripts/data_split.py
python mask_rcnn_train.py
```

Train Mask2Former
```
python mask2former.py
```

