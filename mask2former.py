# %% [markdown]
# ## Install Library


# %% [markdown]
# ## Install Common Libraries

# %%
import os
import glob
import json
import pandas as pd 
from pycocotools.coco import COCO
from PIL import Image
import numpy as np 
import skimage.io as io
from matplotlib import pyplot as plt
import random
from pprint import pprint

import torch.version

# %% [markdown]
# ## Load Dataset

# %%
prefix = './turtles-data/data'
annotation_path = f'{prefix}/annotations.json'
metadata_path = f'{prefix}/metadata_splits.csv'
try:
    # Check if the file exists
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"The file {annotation_path} does not exist.")

    # Attempt to open and load the file
    with open(annotation_path, 'r', encoding='utf8') as file:
        annotations = json.load(file)

    # Check if data is not empty
    if not annotations:
        print("Warning: The loaded JSON data is empty.")
    else:
        print("JSON file loaded successfully.")
        
        # Print the type of the loaded data
        print(f"Type of loaded data: {type(annotations)}")
        
        # Print the keys if it's a dictionary
        if isinstance(annotations, dict):
            print(f"Keys in the JSON data: {', '.join(annotations.keys())}")
        
        # Print the length if it's a list
        elif isinstance(annotations, list):
            print(f"Number of items in the JSON data: {len(annotations)}")
        

except FileNotFoundError as e:
    print(f"Error: {e}")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Let's print out the first few image filenames/paths
for img in annotations['images'][:10]:
    print(img['file_name'])

coco = COCO(annotation_path)

# %%
print(annotations.keys())

# %% [markdown]
# ## Install Detectron2
# Some basic setup:
import sys, os, distutils.core
sys.path.insert(0, os.path.abspath('./detectron2'))
sys.path.insert(0, os.path.abspath('./Mask2Former'))

# %% [markdown]
# ### Check Installation

# %%
import torch, detectron2
torch_version = ".".join(torch.__version__.split(".")[:2])

if torch.version.cuda:
    cuda_version = ''.join(torch.version.cuda.split("."))
else:
    cuda_version = 'none'

print("torch: ", torch_version, "; cuda: ", cuda_version)
print("detectron2:", detectron2.__version__)

# %%
# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# %%
# %% [markdown]
# ## Install Mask2Former

# %%
# Install Mask2Former
# !git clone https://github.com/facebookresearch/Mask2Former.git

# # %%
# # Set up Mask2Former
# %cd Mask2Former
# !pip install -U opencv-python
# !pip install git+https://github.com/cocodataset/panopticapi.git
# %cd mask2former/modeling/pixel_decoder/ops
# !sh make.sh
# !pip install .
# %cd ../../../../../
# %pwd

# %%
# You may need to restart your runtime prior to this, to let your installation take effect
# %cd /kaggle/working/Mask2Former
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="mask2former")

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config

# import Mask2Former project
from mask2former import add_maskformer2_config

# %% [markdown]
# ## Prepare Dataset
# 
# Split the dataset into training, validation, and test sets basedon the metadata_splits.csv which indicate this kind of open-set splitting methodology. 

# %%
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class SeaTurtleDataset(Dataset):
    def __init__(self, root_dir, split_type, transform=None):
        """
        Dataset class for Sea Turtle segmentation
        
        Args:
            root_dir (str): Root directory containing data folder
            train (bool): If True, use training data, else use test data
            transform: Optional transforms to be applied to images
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'data')
        self.transform = transform
        
        # Load split information
        splits_path = os.path.join(root_dir, 'data', 'metadata_splits.csv')
        self.metadata_df = pd.read_csv(splits_path)
        
        # Filter based on split_open column
        self.data = self.metadata_df[self.metadata_df['split_open'] == split_type]
        print(f"Number of {split_type} images: {len(self.data)}")
        
        # Load COCO annotations
        annot_path = os.path.join(root_dir, 'data', 'annotations.json')
        self.coco = COCO(annot_path)
        
        # Get category mapping
        self.cat_ids = sorted(self.coco.getCatIds())
        print("Gets category ids: ", self.cat_ids)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        print("Gets category labels: ", self.cat2label)
    
    def __getitem__(self, index):
        """Get dataset sample"""
        # Get image info from filtered dataframe
        row = self.data.iloc[index]
        img_id = row['id']
        image_path = os.path.join(self.image_dir, row['file_name'])
        
        # Ensure image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Prepare masks and labels
        masks = []
        labels = []
        
        for ann in anns:
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            labels.append(self.cat2label[ann['category_id']])
            
        # Convert to tensor format
        if len(masks) > 0:
            masks = np.stack(masks)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            masks = torch.zeros((0, row['height'], row['width']), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
            
        # Prepare target dictionary
        target = {
            'masks': masks,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.tensor([row['height'], row['width']]),
            'size': torch.tensor([row['height'], row['width']]),
            'identity': row['identity'],
            'timestamp': row['timestamp'],
            'date': row['date'],
            'year': row['year']
        }
        
        # Convert image to tensor
        image = torch.as_tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)
        
        # Apply transforms
        if self.transform is not None:
            image, target = self.transform(image, target)
            
        return image, target
    
    def __len__(self):
        return len(self.data)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class Normalize:
    def __call__(self, image, target):
        image = F.normalize(image, 
                          mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
        return image, target

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = torch.flip(image, [2])
            if "masks" in target:
                target["masks"] = torch.flip(target["masks"], [2])
        return image, target

def create_transforms(is_train=True):
    """Create transform pipeline"""
    transforms = []
    
    if is_train:
        transforms.extend([
            RandomHorizontalFlip(prob=0.5),
        ])
    
    # Always apply normalization
    transforms.append(Normalize())
    
    return Compose(transforms)

def prepare_dataset(root_dir, split_type, train=True):
    """
    Prepare dataset
    
    Args:
        root_dir (str): Root directory containing the data folder
        train (bool): If True, prepare training dataset, else test dataset
    Returns:
        Dataset: Prepared dataset
    """
    transforms = create_transforms(is_train=train)
    dataset = SeaTurtleDataset(
        root_dir=root_dir,
        split_type=split_type,
        transform=transforms
    )
    return dataset


if __name__ == "__main__":
    ROOT_DIR = "./turtles-data"

    # Create training dataset
    train_dataset = prepare_dataset(ROOT_DIR, 'train')
    print(f"Training set size: {len(train_dataset)}\n")
    
    # Create validation dataset
    valid_dataset = prepare_dataset(ROOT_DIR, 'valid')
    print(f"Validation set size: {len(valid_dataset)}\n")
    
    # Create test dataset
    test_dataset = prepare_dataset(ROOT_DIR, 'test')
    print(f"Test set size: {len(test_dataset)}\n")
    
    total = len(train_dataset) + len(valid_dataset) + len(test_dataset)
    print("Total number of splits dataset: ", total)
    
    # Print sample information
    print("\nSample data information:")
    image, target = train_dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Number of instances: {len(target['masks'])}")
    print(f"Labels: {target['labels']}")
    print(f"Identity: {target['identity']}")
    print(f"Date: {target['date']}")
    print(f"Year: {target['year']}")

# %% [markdown]
# ## Access Mask of the certain image

# %%
# Method 1: Direct access from a dataset sample
image, target = train_dataset[0]  # Get first image and its annotations
masks = target['masks']  # This is a tensor of shape [N, H, W] where N is number of instances

# Visualize image and its masks
def visualize_image_and_masks(image, target):
    """
    Visualize image and its segmentation masks
    """
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    image_denorm = image * std + mean
    image_np = image_denorm.permute(1, 2, 0).numpy()
    
    # Get masks and labels
    masks = target['masks'].numpy()
    labels = target['labels'].numpy()
    
    # Create subplot grid
    n_instances = len(masks)
    fig, axes = plt.subplots(2, n_instances + 1, figsize=(5*(n_instances + 1), 8))
    
    # Show original image
    axes[0,0].imshow(image_np)
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')
    
    # Show combined masks on image
    axes[1,0].imshow(image_np)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_instances))
    for mask, color in zip(masks, colors):
        mask_overlay = np.zeros_like(image_np)
        for c in range(3):
            mask_overlay[:, :, c] = color[c]
        mask_overlay = mask_overlay * mask[:, :, np.newaxis]
        axes[1,0].imshow(mask_overlay, alpha=0.4)
    axes[1,0].set_title('All Masks')
    axes[1,0].axis('off')
    
    # Show individual masks
    for i, (mask, label) in enumerate(zip(masks, labels)):
        # Original image with single mask
        axes[0,i+1].imshow(image_np)
        mask_overlay = np.zeros_like(image_np)
        color = colors[i][:3]
        for c in range(3):
            mask_overlay[:, :, c] = color[c]
        mask_overlay = mask_overlay * mask[:, :, np.newaxis]
        axes[0,i+1].imshow(mask_overlay, alpha=0.4)
        axes[0,i+1].set_title(f'Instance {i+1}\nLabel: {label}')
        axes[0,i+1].axis('off')
        
        # Mask only
        axes[1,i+1].imshow(mask, cmap='gray')
        axes[1,i+1].set_title(f'Mask {i+1}')
        axes[1,i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Method 2: Access through COCO API directly
def get_image_annotations(dataset, index):
    """
    Get detailed annotations for an image using COCO API
    """
    row = dataset.data.iloc[index]
    img_id = row['id']
    
    # Get annotations
    ann_ids = dataset.coco.getAnnIds(imgIds=img_id)
    anns = dataset.coco.loadAnns(ann_ids)
    
    # Get masks for each annotation
    masks = []
    labels = []
    cat_names = []
    
    for ann in anns:
        mask = dataset.coco.annToMask(ann)
        category_id = ann['category_id']
        label = dataset.cat2label[category_id]
        category = dataset.coco.loadCats([category_id])[0]
        
        masks.append(mask)
        labels.append(label)
        cat_names.append(category['name'])
    
    return masks, labels, cat_names

# Example usage:
# 1. Using the dataset directly
image, target = train_dataset[0]
# visualize_image_and_masks(image, target)

# 2. Using COCO API
masks, labels, categories = get_image_annotations(train_dataset, 0)
print("\nDetailed annotation information:")
for i, (label, category) in enumerate(zip(labels, categories)):
    print(f"Instance {i+1}:")
    print(f"  - Label ID: {label}")
    print(f"  - Category: {category}")
    print(f"  - Mask shape: {masks[i].shape}")

# To save masks for further use:
def save_masks(masks, save_dir="masks"):
    """
    Save masks as numpy arrays
    """
    os.makedirs(save_dir, exist_ok=True)
    for i, mask in enumerate(masks):
        np.save(os.path.join(save_dir, f"mask_{i}.npy"), mask)
    print(f"Saved {len(masks)} masks to {save_dir}")

# Example of saving masks
# save_masks(masks, "image_masks")

# To load a specific mask later:
# mask = np.load("image_masks/mask_0.npy")

# %% [markdown]
# ## Test Image with pre-trained model 

# %%
import cv2 as cv
# Test with the image data
image_id = 7
img = coco.imgs[image_id]
image_path = os.path.join(prefix, img['file_name'])
image_data = cv.imread(image_path)
image_rgb = cv.cvtColor(image_data, cv.COLOR_BGR2RGB)

plt.imshow(image_rgb)

# %%
# Step 2: Set up the configuration

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)

# Ensure the config file path corresponds to your model's requirements
cfg.merge_from_file("./Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
# cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'
cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_R50_bs16_50ep/model_final_94dc52.pkl'
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Step 3: Create the predictor
predictor = DefaultPredictor(cfg)
outputs = predictor(image_data)

# Print detailed statistics
instances = outputs["instances"].to('cpu')
scores = instances.scores.numpy()
classes = instances.pred_classes.numpy()

print("\nScore distribution:")
print(f"Min score: {scores.min():.3f}")
print(f"Max score: {scores.max():.3f}")
print(f"Mean score: {scores.mean():.3f}")

print("\nClass distribution:")
unique_classes, counts = np.unique(classes, return_counts=True)
for cls, count in zip(unique_classes, counts):
    print(f"Class {cls}: {count} instances")

# Show score histogram
# plt.figure(figsize=(10, 5))
# plt.hist(scores, bins=20)
# plt.title('Distribution of Confidence Scores')
# plt.xlabel('Score')
# plt.ylabel('Count')
# plt.show()

# %% [markdown]
# ### Register dataset in detectron2 format

# %%
# First, let's see what categories are in your dataset
print("Category IDs:", train_dataset.cat_ids)
print("Category mapping:", train_dataset.cat2label)

# Get category names from COCO annotations
categories = train_dataset.coco.loadCats(train_dataset.cat_ids)
category_names = [cat['name'] for cat in categories]
print("Category names:", category_names)

# %% [markdown]
# ### Remove Previous Wrong Registration

# %%
# Check initial state
print("Initially registered datasets:", DatasetCatalog.list())

# Try to unregister
for d in ["train", "valid", "test"]:
    dataset_name = f"sea_turtle_{d}"
    try:
        if dataset_name in DatasetCatalog:
            print(f"Found {dataset_name}, attempting to remove...")
            DatasetCatalog.remove(dataset_name)
            print(f"Successfully removed {dataset_name}")
        else:
            print(f"{dataset_name} not found in catalog")
    except Exception as e:
        print(f"Error removing {dataset_name}: {str(e)}")

# Check what's still registered
print("\nCurrently registered datasets:", DatasetCatalog.list())

# Also check MetadataCatalog
print("\nMetadataCatalog entries:")
for name in MetadataCatalog:
    if 'sea_turtle' in name:
        print(f"Found metadata for: {name}")
        try:
            # Try to remove from MetadataCatalog as well
            MetadataCatalog.remove(name)
            print(f"Removed metadata for {name}")
        except Exception as e:
            print(f"Error removing metadata for {name}: {str(e)}")

# %%
## Register New metadata
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import numpy as np

def get_sea_turtle_dicts(dataset):
    """Convert your dataset to detectron2 format"""
    dataset_dicts = []
    
    for idx in range(len(dataset)):
        try:
            # Get image and target
            image, target = dataset[idx]
            
            # Get image info
            height, width = target['size'].tolist()
            
            # Get file path
            file_name = os.path.join(dataset.image_dir, dataset.data.iloc[idx]['file_name'])
            
            # Create record
            record = {}
            record["file_name"] = file_name
            record["image_id"] = target['image_id'].item()
            record["height"] = height
            record["width"] = width
            
            # Convert annotations
            objs = []
            for mask, label in zip(target['masks'], target['labels']):
                # Convert mask to numpy
                binary_mask = mask.numpy()
                
                # Get bounding box
                bbox = mask_to_bbox(mask)
                
                # Create annotation
                obj = {
                    "category_id": int(label.item()),
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": binary_mask,
                    "iscrowd": 0
                }
                objs.append(obj)
            
            record["annotations"] = objs
            dataset_dicts.append(record)
            
        except Exception as e:
            print(f"Error processing image {idx}: {str(e)}")
            continue
    
    return dataset_dicts

def mask_to_bbox(mask):
    """Convert binary mask to bounding box [x1, y1, x2, y2]"""
    # Convert to numpy for processing
    binary_mask = mask.numpy()
    
    # Find non-zero points
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # Return zeros if mask is empty
        return [0, 0, 0, 0]
    
    # Find boundaries
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Return as [x1, y1, x2, y2]
    return [float(cmin), float(rmin), float(cmax), float(rmax)]


# Now register datasets with correct names
temp_dataset = prepare_dataset(ROOT_DIR, 'train')
cat_ids = temp_dataset.cat_ids
categories = temp_dataset.coco.loadCats(cat_ids)
category_names = [cat['name'] for cat in categories]

print("Category IDs:", cat_ids)
print("Category names:", category_names)

# Register with correct names
for d in ["train", "valid", "test"]:
    dataset_name = f"sea_turtle_{d}"
    DatasetCatalog.register(
        dataset_name,
        lambda d=d: get_sea_turtle_dicts(prepare_dataset(ROOT_DIR, d))
    )
    MetadataCatalog.get(dataset_name).set(
        thing_classes=category_names,  # ['turtle', 'flipper', 'head']
        thing_dataset_id_to_contiguous_id={
            cat_id: idx for idx, cat_id in enumerate(cat_ids)
        }
    )
    print(f"Registered {dataset_name} with categories: {category_names}")

# Verify registration
print("\nVerifying registration:")
train_metadata = MetadataCatalog.get("sea_turtle_train")
print(f"Thing classes: {train_metadata.thing_classes}")
print(f"Thing dataset id mapping: {train_metadata.thing_dataset_id_to_contiguous_id}")

#print(MetadataCatalog.list())
#print(DatasetCatalog.list())

# Test loading data
#train_dicts = DatasetCatalog.get("sea_turtle_train")
#print(f"\nNumber of training samples: {len(train_dicts)}")
#if len(train_dicts) > 0:
#     print("\nFirst sample annotations:")
#     first_sample = train_dicts[0]
#     print(f"Image: {first_sample['file_name']}")
#     print(f"Number of annotations: {len(first_sample['annotations'])}")
#     if len(first_sample['annotations']) > 0:
#         print("First annotation:", first_sample['annotations'][0])

# %% [markdown]
# ## Set Up the Mask2Former Model

# %%
from detectron2 import model_zoo
from detectron2.config import get_cfg
from mask2former.config import add_maskformer2_config
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
import torch
from detectron2.utils.events import EventStorage
from torch.utils.tensorboard import SummaryWriter

class Trainer(DefaultTrainer):
    """Custom trainer with evaluation"""
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

def setup_cfg():
    #print(cfg.dump())
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    add_deeplab_config(cfg)      # Add ResNet/DeepLab configs

    # Load base Mask2Former configuration
    cfg.merge_from_file("./Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml")
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl'
    # Dataset settings
    cfg.DATASETS.TRAIN = ("sea_turtle_train",)
    cfg.DATASETS.TEST = ("sea_turtle_valid",)
    
    # Model settings
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3  # turtle, flipper, head
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
    
    # Training parameters - conservative settings for fine-tuning
    cfg.SOLVER.IMS_PER_BATCH = 2  # Adjust based on your GPU memory
    cfg.SOLVER.BASE_LR = 0.00005  # Small learning rate for fine-tuning
    cfg.SOLVER.MAX_ITER = 5000    # Adjust based on your dataset size
    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    
    # Learning rate scheduler
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    cfg.SOLVER.STEPS = (3000, 4500)  # Steps to decrease learning rate
    cfg.SOLVER.GAMMA = 0.1  # Learning rate decay factor
    
    # Gradient clipping settings
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"  # Use "norm" for clipping
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0  # Clipping value

    
    # Validation period
    cfg.TEST.EVAL_PERIOD = 500  # Validate every 500 iterations
    
    # Data augmentation (mild for fine-tuning)
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.RANDOM_FLIP = "horizontal"
    
    # Set training device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Output directory
    cfg.OUTPUT_DIR = "./output_maskformer_pretrained"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg

def train_model():
    cfg = setup_cfg()

    # Print training configuration
    print("\nTraining Configuration:")
    print(f"Device: {cfg.MODEL.DEVICE}")
    print(f"Learning Rate: {cfg.SOLVER.BASE_LR}")
    print(f"Max Iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"Batch Size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"Number of Classes: {cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES}")
    print(f"Pre-trained Weights: {cfg.MODEL.WEIGHTS}")
    print(f"Output Directory: {cfg.OUTPUT_DIR}")
    
    # Setup tensorboard
    writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'tensorboard'))
    
    # Initialize trainer
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    
    print("\nStarting training...")
    
    # Training loop with monitoring
    try:
        with EventStorage(start_iter=0):
            trainer.train()
    except Exception as e:
        print(f"Training interrupted: {str(e)}")
    finally:
        writer.close()

# Function to test the model
def test_model(cfg, dataset):
    from detectron2.engine import DefaultPredictor
    
    # Load the trained model
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)
    
    # Run evaluation
    evaluator = COCOEvaluator(
        cfg.DATASETS.TEST[0],
        output_dir=os.path.join(cfg.OUTPUT_DIR, "final_evaluation")
    )
    
    print("\nRunning final evaluation...")
    evaluator.reset()
    
    for i in range(min(5, len(dataset))):  # Test on first 5 images
        image, target = dataset[i]
        
        # Prepare image for inference
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)
        
        # Run inference
        outputs = predictor(image)
        
        # Visualize predictions
        #visualize_predictions(image, outputs)
        
        # Print prediction info
        instances = outputs["instances"].to("cpu")
        print(f"\nPredictions for image {i}:")
        print(f"Number of instances: {len(instances)}")
        print(f"Predicted classes: {instances.pred_classes}")
        print(f"Scores: {instances.scores}")

if __name__ == "__main__":
    # Start training
    train_model()
    
    # After training, test the model
    cfg = setup_cfg()
    test_model(cfg, valid_dataset)

# %% [markdown]
# ## Visualise Output


