import torch
import numpy as np
import cv2
from quantus import IROF, max_sensitivity, focus, effective_complexity, explain

# --- Helper Functions ---

def normalize_heatmap(heatmap):
    """
    Normalizes the heatmap to the range [0, 1].
    """
    
    heatmap -= heatmap.min()
    heatmap /= heatmap.max() + 1e-8
    return heatmap


# --- IROF Implementation using Quantus ---

def calculate_irof_quantus(input_image, explanation, model, device, segmentation_method="slic", perturb_baseline="mean", disable_warnings=True):
    """
    Calculates the Intersection over Region of Fluctuation (IROF) using quantus.
    """

    # 1. Define the IROF metric with desired parameters
    irof_metric = IROF(
        segmentation_method=segmentation_method,  # Use "slic" for now
        perturb_baseline=perturb_baseline,  # Use "mean" for now
        perturb_func=lambda x, indices, baseline: np.where(indices, baseline, x),
        return_aggregate=True, # We want one score over the image
        disable_warnings=disable_warnings
    )

    # 2. The IROF metric expects: image, model, explanation function, and explanation kwargs

    def model_prediction_func(images):
        """Wrapper to comply with quantus model prediction function requirement."""
        model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(images).permute(0, 3, 1, 2).float().to(device)  # B, C, H, W
            output = model(input_tensor)
            output_key = 'out' if hasattr(output, 'out') else 'sem_seg'
            probs = torch.softmax(output[output_key], dim=1)  # Get probabilities
            return probs.cpu().numpy()  # Return probabilities as numpy array

    #Prepare Image and explanation:
    input_image = input_image.astype(np.float32) #Image requires float32
    explanation = normalize_heatmap(explanation)  #Quantus works better with normalize explanations
    explanation = np.expand_dims(explanation, axis=0) #needs a batch dimension

    # 3. Calculate IROF score
    irof_score = irof_metric(
        model=model,
        x_batch=np.expand_dims(input_image, axis=0),  # Needs a batch dimension
        y_batch=None,  # Not needed for IROF
        a_batch=explanation,  # Pass the pre-computed explanation
        device=device,
        explain_func=None,  # We're not using quantus to explain, we have the XAI method
        explain_func_kwargs=None  # Because we are passing in the explanation
    )[0] #Takes first element of IROF

    return irof_score

def calculate_max_sensitivity_quantus(heatmap, input_image, model, device, label_id, perturbation_size=0.1):
    """
    Calculates the Max-Sensitivity using quantus.
    """
    
    # Normalizes the heatmap to the range [0, 1].
    heatmap = normalize_heatmap(heatmap)

    def perturbation_func(image, perturbation_region): #Wrapper to comply with quantus
        image_perturbed = image.copy()
        y_start, y_end, x_start, x_end = perturbation_region #Region must be defined this way
        image_perturbed[y_start:y_end, x_start:x_end] = cv2.GaussianBlur(image_perturbed[y_start:y_end, x_start:x_end], (5, 5), 0)
        return image_perturbed
    
    def model_prediction_func(images):  #Wrapper to comply with quantus
      model.eval()
      with torch.no_grad():
        input_tensor = torch.tensor(images).permute(0,3,1,2).float().to(device) #Image expected as numpy HWC
        output = model(input_tensor)

        output_key = 'out' if hasattr(output, 'out') else 'sem_seg' #Check for proper key for the model output
        probs = torch.softmax(output[output_key], dim=1)[:,label_id].cpu().numpy() #label_id indicates the class
        return probs #Returns probabilites
  
    return max_sensitivity(
        explanation=heatmap,
        img=input_image,
        model_prediction_func=model_prediction_func,
        perturbation_func=perturbation_func,
        perturbation_size=perturbation_size #The rest of the parameters use default parameters

    )

def calculate_focus_quantus(heatmap, segmentation_mask):
    """
    Calculates the Focus using quantus.
    """
    
    # Quantus expects binary explanation and segmentation mask
    heatmap = normalize_heatmap(heatmap)

    # Binarize explanation
    explanation = (heatmap > 0.5).astype(int)
    
    # Binarize segmentation mask
    segmentation_mask = (segmentation_mask > 0).astype(int)  

    return focus(explanation=explanation, segmentation_mask=segmentation_mask)

def calculate_effective_complexity_quantus(heatmap):
    """
    Calculates the Effective Complexity using quantus.
    """
    
    # Quantus expects a numpy array as input
    heatmap = normalize_heatmap(heatmap)
    
    return effective_complexity(explanation=heatmap)