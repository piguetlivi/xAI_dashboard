# This file contains the code to predict the output of the image using the pretrained models
# From torchvision library. The models used are FCN Resnet101, FCN Resnet50, DeepLabV3 Resnet50, DeepLabV3 Resnet101 and DeepLabV3 MobilenetV3 Large.
# The code also contains the code to predict the output of the image using the Mask2Former model from Hugging Face.

# Import required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from models import mask2former_model_large, mask2former_model_small

device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to predict the output of the image using the Mask2Former model.
def predict_mask2former_small(image_path, task="semantic"):
    
    """
    Function to predict the output of the image using the Mask2Former model (small).
    """

    input_image = Image.open(image_path).convert("RGB")

    # Load model and processor
    model, processor = mask2former_model_small()

    inputs = processor(images=input_image, task_inputs=[task], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Use Hugging Face processor to get final semantic segmentation map
    target_size = input_image.size[::-1]  # (height, width)
    processed = processor.post_process_semantic_segmentation(outputs, target_sizes=[target_size])[0]  # single image

    return input_image, processed

def predict_mask2former_large(image_path, task="semantic"):
        
        """
        Function to predict the output of the image using the Mask2Former model (large).
        """
    
        input_image = Image.open(image_path).convert("RGB")
    
        # Load model and processor
        model, processor = mask2former_model_large()
    
        inputs = processor(images=input_image, task_inputs=[task], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
        with torch.no_grad():
            outputs = model(**inputs)
    
        # Use Hugging Face processor to get final semantic segmentation map
        target_size = input_image.size[::-1]  # (height, width)
        processed = processor.post_process_semantic_segmentation(outputs, target_sizes=[target_size])[0]  # single image
    
        return input_image, processed


# Function to predict the output of the image using the FCN Resnet101 model.
def predict_fcn_resnet101(input_image_pil: Image.Image) -> tuple[Image.Image, Image.Image]:
    """
    Predicts segmentation using the FCN Resnet101 model from PyTorch Hub.
    Args:
        input_image_pil (PIL.Image.Image): The input image as a PIL Image object (RGB).
    Returns:
        tuple: A tuple containing:
            - input_image_pil (PIL.Image.Image): The original input image.
            - output_predictions_pil (PIL.Image.Image): The predicted segmentation map as a PIL Image.
    """
    print("Running FCN ResNet101 prediction...")
    # Load the model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet101', pretrained=True)
    model.to(device).eval() # Set model to evaluation mode and move to device

    # Define preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Preprocess the input PIL image
    input_tensor = preprocess(input_image_pil)
    input_batch = input_tensor.unsqueeze(0).to(device) # Add batch dimension and move to device

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)['out'][0] # Get output, remove batch dim

    # Get class predictions
    output_predictions = output.argmax(0) # Get the class index for each pixel

    # Convert predictions tensor to PIL Image
    output_predictions_np = output_predictions.byte().cpu().numpy() # Ensure uint8 for PIL
    output_predictions_pil = Image.fromarray(output_predictions_np)

    print("FCN ResNet101 prediction finished.")
    # Return the original input PIL and the output PIL prediction
    return input_image_pil, output_predictions_pil

def predict_fcn_resnet50(input_image_pil: Image.Image) -> tuple[Image.Image, Image.Image]:
    """
    Predicts segmentation using the FCN Resnet50 model from PyTorch Hub.
    Args:
        input_image_pil (PIL.Image.Image): The input image as a PIL Image object (RGB).
    Returns:
        tuple: A tuple containing:
            - input_image_pil (PIL.Image.Image): The original input image.
            - output_predictions_pil (PIL.Image.Image): The predicted segmentation map as a PIL Image.
    """
    print("Running FCN ResNet50 prediction...")
    # Load the model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
    model.to(device).eval() # Set model to evaluation mode and move to device

    # Define preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Preprocess the input PIL image
    input_tensor = preprocess(input_image_pil)
    input_batch = input_tensor.unsqueeze(0).to(device) # Add batch dimension and move to device

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)['out'][0] # Get output, remove batch dim

    # Get class predictions
    output_predictions = output.argmax(0) # Get the class index for each pixel

    # Convert predictions tensor to PIL Image
    output_predictions_np = output_predictions.byte().cpu().numpy() # Ensure uint8 for PIL
    output_predictions_pil = Image.fromarray(output_predictions_np)

    print("FCN ResNet50 prediction finished.")
    # Return the original input PIL and the output PIL prediction
    return input_image_pil, output_predictions_pil

def predict_deeplabv3_resnet50(input_image_pil: Image.Image) -> tuple[Image.Image, Image.Image]:
    """
    Predicts segmentation using the DeepLabV3 Resnet50 model from PyTorch Hub.
    Args:
        input_image_pil (PIL.Image.Image): The input image as a PIL Image object (RGB).
    Returns:
        tuple: A tuple containing:
            - input_image_pil (PIL.Image.Image): The original input image.
            - output_predictions_pil (PIL.Image.Image): The predicted segmentation map as a PIL Image.
    """
    print("Running DeepLabV3 ResNet50 prediction...")
    # Load the model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.to(device).eval() # Set model to evaluation mode and move to device

    # Define preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Preprocess the input PIL image
    input_tensor = preprocess(input_image_pil)
    input_batch = input_tensor.unsqueeze(0).to(device) # Add batch dimension and move to device

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)['out'][0] # Get output, remove batch dim

    # Get class predictions
    output_predictions = output.argmax(0) # Get the class index for each pixel

    # Convert predictions tensor to PIL Image
    output_predictions_np = output_predictions.byte().cpu().numpy() # Ensure uint8 for PIL
    output_predictions_pil = Image.fromarray(output_predictions_np)

    print("DeepLabV3 ResNet50 prediction finished.")
    # Return the original input PIL and the output PIL prediction
    return input_image_pil, output_predictions_pil

def predict_deeplabv3_resnet101(input_image_pil: Image.Image) -> tuple[Image.Image, Image.Image]:
    """
    Predicts segmentation using the DeepLabV3 Resnet101 model from PyTorch Hub.
    Args:
        input_image_pil (PIL.Image.Image): The input image as a PIL Image object (RGB).
    Returns:
        tuple: A tuple containing:
            - input_image_pil (PIL.Image.Image): The original input image.
            - output_predictions_pil (PIL.Image.Image): The predicted segmentation map as a PIL Image.
    """
    print("Running DeepLabV3 ResNet101 prediction...")
    # Load the model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    model.to(device).eval() # Set model to evaluation mode and move to device

    # Define preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Preprocess the input PIL image
    input_tensor = preprocess(input_image_pil)
    input_batch = input_tensor.unsqueeze(0).to(device) # Add batch dimension and move to device

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)['out'][0] # Get output, remove batch dim

    # Get class predictions
    output_predictions = output.argmax(0) # Get the class index for each pixel

    # Convert predictions tensor to PIL Image
    output_predictions_np = output_predictions.byte().cpu().numpy() # Ensure uint8 for PIL
    output_predictions_pil = Image.fromarray(output_predictions_np)

    print("DeepLabV3 ResNet101 prediction finished.")
    # Return the original input PIL and the output PIL prediction
    return input_image_pil, output_predictions_pil

def predict_deeplabv3_mobilenetv3_large(input_image_pil: Image.Image) -> tuple[Image.Image, Image.Image]:
    """
    Predicts segmentation using the DeepLabV3 MobileNetV3-Large model from PyTorch Hub.
    Args:
        input_image_pil (PIL.Image.Image): The input image as a PIL Image object (RGB).
    Returns:
        tuple: A tuple containing:
            - input_image_pil (PIL.Image.Image): The original input image.
            - output_predictions_pil (PIL.Image.Image): The predicted segmentation map as a PIL Image.
    """
    print("Running DeepLabV3 MobileNetV3-Large prediction...")
    # Load the model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
    model.to(device).eval() # Set model to evaluation mode and move to device

    # Define preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Preprocess the input PIL image
    input_tensor = preprocess(input_image_pil)
    input_batch = input_tensor.unsqueeze(0).to(device) # Add batch dimension and move to device

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)['out'][0] # Get output, remove batch dim

    # Get class predictions
    output_predictions = output.argmax(0) # Get the class index for each pixel

    # Convert predictions tensor to PIL Image
    output_predictions_np = output_predictions.byte().cpu().numpy() # Ensure uint8 for PIL
    output_predictions_pil = Image.fromarray(output_predictions_np)

    print("DeepLabV3 MobileNetV3-Large prediction finished.")
    # Return the original input PIL and the output PIL prediction
    return input_image_pil, output_predictions_pil