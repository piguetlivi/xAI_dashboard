# This file contains the labels and color maps for different semantic segmentation datasets.

# Import the required libraries
from torchvision.models.segmentation import FCN_ResNet50_Weights
import numpy as np
import sys

# --- COCO-Labels ---
# (for models: fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenetv3_large)
try:
    # Attempt to get labels directly from a standard PyTorch model weight meta data
    # This is typical for models pre-trained on COCO Segmentation, often including VOC classes.
    COCO_LABELS = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.meta["categories"]

    # Ensure background ('__background__') is the first entry if not already.
    # PyTorch models often output index 0 for background.
    if COCO_LABELS[0] != '__background__':
        COCO_LABELS.insert(0, '__background__')

    NUM_COCO_CLASSES = len(COCO_LABELS)
    print(f"Defined {NUM_COCO_CLASSES} COCO classes (including background).")

    # Define colors for COCO classes. Using random colors is less standard,
    # but common when no specific palette is mandated. We ensure background is black.
    np.random.seed(42) # For reproducible "random" colors
    COCO_COLORS = np.random.randint(0, 255, size=(NUM_COCO_CLASSES, 3), dtype=np.uint8)
    COCO_COLORS[0] = [0, 0, 0] # Ensure background (index 0) is black

    # COCO: Maps index (0..N-1) to RGB tuple
    COCO_COLOR_DICT = {i: tuple(color) for i, color in enumerate(COCO_COLORS)}
    # Also create a mapping from label string to index
    COCO_LABEL_TO_INDEX = {label: i for i, label in enumerate(COCO_LABELS)}

except Exception as e:
    print(f"Warning: Could not load COCO labels from torchvision weights. Error: {e}", file=sys.stderr)
    print("COCO labels and colors will not be available.", file=sys.stderr)
    COCO_LABELS = []
    NUM_COCO_CLASSES = 0
    COCO_COLORS = np.array([], dtype=np.uint8).reshape(0, 3)
    COCO_COLOR_DICT = {}
    COCO_LABEL_TO_INDEX = {}


# --- Cityscapes-Labels ---
# (for model: mask2former_model_small and other Cityscapes models)

# Labels corresponding to trainIds 0-18. The dataset also has other IDs (like 19-33 and 255)
# but models are typically trained to predict the 19 'trainId' classes.
CITYSCAPES_LABELS = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle'
]
NUM_CITYSCAPES_CLASSES = len(CITYSCAPES_LABELS) # Should be 19
print(f"Defined {NUM_CITYSCAPES_CLASSES} Cityscapes classes.")

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

# Cityscapes: Maps trainId (0..18) to RGB tuple
CITYSCAPES_COLOR_DICT = {i: tuple(color) for i, color in enumerate(CITYSCAPES_COLORS)}
# Maps label string to trainId (0..18)
CITYSCAPES_LABEL_TO_INDEX = {label: i for i, label in enumerate(CITYSCAPES_LABELS)}


# --- ADE20K-Labels ---
# for models mask2former_model_large (ADE20K) and other ADE20K models.
# Standard ADE20K has 150 classes, often indexed 1-150, with index 0 for 'void'/background.
# We will define the 150 labels and create a palette including background at index 0.

# Helper function to generate a deterministic color based on index (commonly used for ADE20K)
def bitget(byteval, idx):
    """Checks the idx'th bit of a byte."""
    return (byteval & (1 << idx)) != 0

def index_to_color(idx):
    """
    Generates a deterministic color tuple for a given index.
    Used for ADE20K standard palette (for indices 0-150).
    """
    r = 0
    g = 0
    b = 0
    i = 0
    # Iterate through bits of the index
    while idx > 0:
        # Use bits of index to set color components (reordered for better distinction)
        r |= (bitget(idx, 0) << (7 - i))
        g |= (bitget(idx, 1) << (7 - i))
        b |= (bitget(idx, 2) << (7 - i))
        # Shift index and bit position
        idx >>= 3 # Process 3 bits at a time
        i += 1
    # Ensure color is within [0, 255] range (though bitget up to 7 won't exceed this)
    return (r % 256, g % 256, b % 256)


# The 150 foreground classes for ADE20K
ADE20K_FOREGROUND_LABELS = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain",
    "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion",
    "base", "box", "column", "signboard", "chest of drawers", "counter", "sand", "sink", "skyscraper",
    "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool table",
    "pillow", "screen door", "stairway", "river", "bridge", "bookcase", "blind", "coffee table",
    "toilet", "flower", "book", "hill", "bench", "countertop", "stove", "palm", "kitchen island",
    "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel", "light",
    "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television receiver",
    "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman",
    "bottle", "buffet", "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy",
    "washer", "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag",
    "minibike", "cradle", "oven", "ball", "food", "step", "tank", "trade name", "microwave", "pot",
    "animal", "bicycle", "lake", "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce",
    "vase", "traffic light", "tray", "ashcan", "fan", "pier", "crt screen", "plate", "monitor",
    "bulletin board", "shower", "radiator", "glass", "clock", "flag"
]

# Add background/void label at the beginning (index 0)
ADE20K_LABELS = ["__background__"] + ADE20K_FOREGROUND_LABELS

NUM_ADE20K_CLASSES = len(ADE20K_LABELS) # Should be 151
print(f"Defined {NUM_ADE20K_CLASSES} ADE20K classes (including background).")

# Define colors for ADE20K classes (using the deterministic generation)
# Index 0 will be black (index_to_color(0) == (0,0,0)), which is suitable for background/void.
# Indices 1-150 will get deterministic colors for the foreground classes.
ADE20K_COLORS_LIST = [index_to_color(i) for i in range(NUM_ADE20K_CLASSES)]
ADE20K_COLORS = np.array(ADE20K_COLORS_LIST, dtype=np.uint8)

# Verify consistency
assert len(ADE20K_COLORS) == NUM_ADE20K_CLASSES, \
    "Mismatch between number of ADE20K labels and defined colors."

# ADE20K: Maps index (0..150) to RGB tuple
ADE20K_COLOR_DICT = {i: tuple(color) for i, color in enumerate(ADE20K_COLORS)}
# Maps label string to index (0..150)
ADE20K_LABEL_TO_INDEX = {label: i for i, label in enumerate(ADE20K_LABELS)}

print("Defined ADE20K labels and color mappings (using standard deterministic palette).")

# --- Default Handling ---
# Default color for any unexpected label IDs (e.g., 255 for 'unlabeled' in some datasets)
DEFAULT_COLOR = (128, 128, 128) # Grey

# Example usage (optional, for testing)
if __name__ == "__main__":
    print("\n--- Examples ---")
    print(f"COCO Labels (first 5): {COCO_LABELS[:5]}")
    if 1 in COCO_COLOR_DICT:
         print(f"Color for '{COCO_LABELS[1]}': {COCO_COLOR_DICT[1]}")
    if '__background__' in COCO_LABEL_TO_INDEX:
         print(f"Index for '__background__': {COCO_LABEL_TO_INDEX['__background__']}")

    print(f"\nCityscapes Labels (first 5): {CITYSCAPES_LABELS[:5]}")
    print(f"Color for '{CITYSCAPES_LABELS[0]}': {CITYSCAPES_COLOR_DICT[0]}") # road
    if 'car' in CITYSCAPES_LABEL_TO_INDEX:
        car_index = CITYSCAPES_LABEL_TO_INDEX['car']
        print(f"Index for 'car': {car_index}")
        print(f"Color for 'car': {CITYSCAPES_COLOR_DICT[car_index]}") # car

    print(f"\nADE20K Labels (first 5): {ADE20K_LABELS[:5]}")
    print(f"ADE20K Labels (last 5): {ADE20K_LABELS[-5:]}")
    print(f"Color for '{ADE20K_LABELS[0]}': {ADE20K_COLOR_DICT[0]}") # __background__ (should be black)
    print(f"Color for '{ADE20K_LABELS[1]}': {ADE20K_COLOR_DICT[1]}") # wall
    if 'building' in ADE20K_LABEL_TO_INDEX:
        building_index = ADE20K_LABEL_TO_INDEX['building']
        print(f"Index for 'building': {building_index}")
        print(f"Color for 'building': {ADE20K_COLOR_DICT[building_index]}") # building
    if 'flag' in ADE20K_LABEL_TO_INDEX:
        flag_index = ADE20K_LABEL_TO_INDEX['flag']
        print(f"Index for 'flag': {flag_index}")
        print(f"Color for 'flag': {ADE20K_COLOR_DICT[flag_index]}") # flag (should be index 150)