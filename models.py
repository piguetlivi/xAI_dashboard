##Description: This script contains functions to load the pre-trained models from the PyTorch model zoo and Hugging Face.
# import necessary libraries
import torch
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor


# check if GPU is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def mask2former_model():
    ''' 
    Function to load the Mask2Former model from Hugging Face
    '''
    processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes")
    model.eval()
    return model, processor

def oneformer_model():
    '''
    Function to load the OneFormer model from Hugging Face
    '''
    processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_swin_large", safe_kwargs=True)
    model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
    
    return model, processor



def fcn_resnet50():
    '''
    Function to load the FCN ResNet50 model from the PyTorch model zoo
    '''
    
    fcn_resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True).to(device).eval()
    
    return fcn_resnet50

def fcn_resnet101():
    '''
    Function to load the FCN ResNet101 model from the PyTorch model zoo
    '''
    
    fcn_resnet101 = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet101', pretrained=True).to(device).eval()
    
    return fcn_resnet101

def deeplabv3_resnet50():
    '''
    Function to load the DeepLabV3 ResNet50 model from the PyTorch model zoo
    '''
    
    deeplabv3_resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True).to(device).eval()
    
    return deeplabv3_resnet50

def deeplabv3_resnet101():
    '''
    Function to load the DeepLabV3 ResNet101 model from the PyTorch model zoo
    '''
    
    deeplabv3_resnet101 = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True).to(device).eval()
   
    return deeplabv3_resnet101

def deeplabv3_mobilenetv3_large():
    '''
    Function to load the DeepLabV3 MobileNetV3 Large model from the PyTorch model zoo
    '''
    
    deeplabv3_mobilenetv3_large = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True).to(device).eval()

    return deeplabv3_mobilenetv3_large
