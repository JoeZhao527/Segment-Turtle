import os
import json
import pandas as pd
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from tqdm import tqdm  # Import tqdm for progress bar
import numpy as np  # Ensure numpy is imported for mask operations
import cv2

# Paths
base_dir = "./turtles-data/data"
annotations_path = os.path.join(base_dir, 'annotations.json')
split_path = os.path.join(base_dir, 'metadata_splits.csv')

# Load the split dataframe
split = pd.read_csv(split_path)

# Load the COCO annotations
coco = COCO(annotations_path)

# Separate the COCO ids into train/valid/test based on the split DataFrame
train_ids = split[split['split_closed'] == 'train']['id'].tolist()
valid_ids = split[split['split_closed'] == 'valid']['id'].tolist()
test_ids = split[split['split_closed'] == 'test']['id'].tolist()

def rle_to_polygon_annToMask(coco, ann):
    """
    Convert the RLE or polygon annotation into a binary mask and then extract polygon contours using OpenCV.
    :param coco: COCO object.
    :param ann: Single annotation from coco.anns.
    :return: Polygon segmentation.
    """
    # Convert annotation to binary mask
    binary_mask = coco.annToMask(ann)
    
    # Find contours using OpenCV
    binary_mask = np.asfortranarray(binary_mask).astype(np.uint8)  # Ensure uint8 format for OpenCV
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert contours into polygon format
    polygons = []
    for contour in contours:
        contour = contour.flatten().tolist()  # Convert (x, y) to list of points
        if len(contour) >= 6:  # At least 3 points needed for a valid polygon
            polygons.append(contour)

    return polygons

# Function to filter COCO annotations by image ids and convert iscrowd=1 annotations using annToMask
def filter_coco_by_ids(coco, ids):
    new_coco = {
        'images': [],
        'annotations': [],
        'categories': coco.dataset['categories']
    }

    # Filter images
    id_set = set(ids)
    new_coco['images'] = [img for img in coco.dataset['images'] if img['id'] in id_set]

    # Filter annotations by image ids and handle iscrowd
    image_id_set = {img['id'] for img in new_coco['images']}
    annotations = coco.dataset['annotations']

    # Use tqdm to show progress for annotation processing
    for ann in tqdm(annotations, desc="Processing annotations", total=len(annotations)):
        if ann['image_id'] in image_id_set:
            ann_copy = ann.copy()
            
            # If iscrowd is 1, convert the annotation mask to polygons and set iscrowd to 0
            if ann['iscrowd'] == 1:
                # Convert annotation to mask and extract polygons
                ann_copy['segmentation'] = rle_to_polygon_annToMask(coco, ann)
                ann_copy['iscrowd'] = 0  # Set iscrowd to 0 after conversion
            
            # Append to new annotations
            new_coco['annotations'].append(ann_copy)

    return new_coco

# Filter and create new COCO annotations for each split
train_coco = filter_coco_by_ids(coco, train_ids)
valid_coco = filter_coco_by_ids(coco, valid_ids)
test_coco = filter_coco_by_ids(coco, test_ids)

# Save the new COCO datasets as JSON
def save_coco_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f)

# Paths to save new annotations
save_coco_json(train_coco, os.path.join(base_dir, 'annotations_train.json'))
save_coco_json(valid_coco, os.path.join(base_dir, 'annotations_valid.json'))
save_coco_json(test_coco, os.path.join(base_dir, 'annotations_test.json'))

print("COCO dataset split and saved as JSON for train, valid, and test sets with iscrowd converted to 0.")
