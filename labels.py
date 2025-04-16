# This file contains the labels for the different models. 
# The labels are used to map the class indices to the class names.

# Import the required libraries
from torchvision.models.segmentation import FCN_ResNet50_Weights

# COCO-Labels (for models: fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenetv3_large)
COCO_LABELS = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.meta["categories"]

# Cityscapes-Labels (for model: mask2former)
CITYSCAPES_LABELS = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle'
]