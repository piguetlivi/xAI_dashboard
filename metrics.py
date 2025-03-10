#This script contains the metrics to evaluate XAI results
import quantus
import torch
import torchvision
from torchvision import transforms
from predict_models import predict_fcn_resnet101, predict_fcn_resnet50, predict_deeplabv3_resnet50, predict_deeplabv3_resnet101, predict_deeplabv3_mobilenetv3_large, predict_oneformer
import methods
import models
  
# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

