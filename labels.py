# This file contains the labels for the different models. 
# The labels are used to map the class indices to the class names.

# Import the required libraries
from torchvision.models.segmentation import FCN_ResNet50_Weights
import numpy as np

# COCO-Labels (for models: fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenetv3_large)
COCO_LABELS = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.meta["categories"]

# Ensure background ('__background__') is the first entry if not already
# This matches the typical output where index 0 is background for these models.
if COCO_LABELS[0] != '__background__':
    COCO_LABELS.insert(0, '__background__')

NUM_COCO_CLASSES = len(COCO_LABELS)
print(f"Number of COCO classes (including background): {NUM_COCO_CLASSES}")

# Define fixed colors for COCO classes
np.random.seed(42) # For reproducible "random" colors (better to define manually if needed)
COCO_COLORS = np.random.randint(0, 255, size=(NUM_COCO_CLASSES, 3), dtype=np.uint8)
COCO_COLORS[0] = [0, 0, 0] # Ensure background (index 0) is black

# Cityscapes-Labels (for model: mask2former)

# Labels corresponding to trainIds 0-18
CITYSCAPES_LABELS = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle'
]
NUM_CITYSCAPES_CLASSES = len(CITYSCAPES_LABELS) # Should be 19
print(f"Number of Cityscapes classes: {NUM_CITYSCAPES_CLASSES}")

# Define STANDARD fixed colors for Cityscapes classes (trainId 0-18)
# Using standard colors makes visualizations comparable and understandable.
CITYSCAPES_COLORS_STANDARD = [
    (128, 64, 128),    # 0: road
    (244, 35, 232),    # 1: sidewalk
    (70, 70, 70),      # 2: building
    (102, 102, 156),   # 3: wall
    (190, 153, 153),   # 4: fence
    (153, 153, 153),   # 5: pole
    (250, 170, 30),    # 6: traffic light
    (220, 220, 0),     # 7: traffic sign
    (107, 142, 35),    # 8: vegetation
    (152, 251, 152),   # 9: terrain
    (70, 130, 180),    # 10: sky
    (220, 20, 60),     # 11: person
    (255, 0, 0),       # 12: rider
    (0, 0, 142),       # 13: car
    (0, 0, 70),        # 14: truck
    (0, 60, 100),      # 15: bus
    (0, 80, 100),      # 16: train
    (0, 0, 230),       # 17: motorcycle
    (119, 11, 32)      # 18: bicycle
]
# Convert to NumPy array
CITYSCAPES_COLORS = np.array(CITYSCAPES_COLORS_STANDARD, dtype=np.uint8)

# Verify consistency
assert len(CITYSCAPES_COLORS) == NUM_CITYSCAPES_CLASSES, \
    "Mismatch between number of Cityscapes labels and defined standard colors."

# --- Color Dictionaries for Easy Lookup in app.py ---

# COCO: Maps index (0..N-1) to RGB tuple
COCO_COLOR_DICT = {i: tuple(color) for i, color in enumerate(COCO_COLORS)}

# Cityscapes: Maps trainId (0..18) to RGB tuple
CITYSCAPES_COLOR_DICT = {i: tuple(color) for i, color in enumerate(CITYSCAPES_COLORS)}

# Default color for any unexpected label IDs (e.g., 255 for 'unlabeled' if it appears)
DEFAULT_COLOR = (128, 128, 128) # Grey

print("Defined COCO and Cityscapes labels and color mappings (using standard Cityscapes colors).")