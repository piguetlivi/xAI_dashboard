import torch
import numpy as np
import cv2
from quantus import IROF, MaxSensitivity, Focus, EffectiveComplexity

# --- Helper Functions ---

def normalize_heatmap(heatmap):
    """
    Normalizes the heatmap to the range [0, 1] using min-max scaling.
    Converts input to float32 if necessary and handles potential division by zero.
    """
    if not isinstance(heatmap, np.ndarray):
        raise TypeError(f"normalize_heatmap expects a numpy array, got {type(heatmap)}")
    if heatmap.size == 0:
        return np.array([], dtype=np.float32) # Handle empty array

    # Convert to float32 if necessary
    if heatmap.dtype != np.float32:
        heatmap_float = heatmap.astype(np.float32)
    else:
        heatmap_float = heatmap

    min_val = np.min(heatmap_float)
    max_val = np.max(heatmap_float)
    range_val = max_val - min_val

    if range_val < 1e-8:
        normalized_heatmap = np.zeros_like(heatmap_float)
    else:
        normalized_heatmap = (heatmap_float - min_val) / range_val

    return normalized_heatmap


# --- IROF Implementation using Quantus ---

def calculate_irof_quantus(input_image, explanation, model, device, segmentation_method="slic", perturb_baseline="mean", disable_warnings=True):
    """
    Calculates the Intersection over Region of Fluctuation (IROF) using quantus.
    Expects input_image as HWC uint8/float, explanation as HW float.
    Expects device as torch.device object.
    """
    if not isinstance(input_image, np.ndarray) or input_image.ndim != 3:
         raise ValueError("IROF expects input_image as HWC numpy array.")
    if not isinstance(explanation, np.ndarray) or explanation.ndim != 2:
         raise ValueError("IROF expects explanation as HW numpy array.")
    if not isinstance(device, torch.device):
         # If device string is passed, convert it. Prefer passing the object directly.
         print("Warning (IROF): Received device string, converting to torch.device. Pass the object directly.")
         device = torch.device(device)


    # 1. Define the IROF metric
    irof_metric = IROF(
        segmentation_method=segmentation_method,
        perturb_baseline=perturb_baseline,
        # perturb_func=lambda x, indices, baseline: np.where(indices, baseline, x), # Default perturb_func is usually fine
        return_aggregate=True,
        disable_warnings=disable_warnings
    )

    # 2. Prepare inputs for Quantus
    # Ensure image is float32 for model processing inside wrapper
    input_image_float = input_image.astype(np.float32)
    x_batch = np.expand_dims(input_image_float, axis=0) # Add batch dim (B=1, H, W, C)

    # Normalize explanation and add batch dim
    explanation_norm = normalize_heatmap(explanation)
    a_batch = np.expand_dims(explanation_norm, axis=0) # Add batch dim (B=1, H, W)

    # 3. Calculate IROF score
    # Note: IROF in Quantus doesn't typically need y_batch or a model. It compares segmentation
    # of the image with regions derived from the explanation.
    # Check Quantus docs if your version requires model/y_batch here. Assuming it doesn't.
    try:
        # Simpler call signature based on common IROF usage:
        irof_score = irof_metric(
            x_batch=x_batch,
            a_batch=a_batch,
            # If model and device are strictly required by your Quantus version/IROF impl:
            # model=model,
            # device=device,
        )
        # If the metric returns a list/dict, extract the score
        if isinstance(irof_score, list):
            return irof_score[0]
        elif isinstance(irof_score, dict):
             # Find the appropriate key, e.g., 'irof_score' or the metric name
             # This depends on the specific Quantus version and metric settings
             return list(irof_score.values())[0] # Example: just return the first value
        else:
            return irof_score # Assume it's the score directly

    except Exception as e:
        print(f"Error during Quantus IROF calculation: {e}")
        raise # Re-raise the exception for debugging in the main app

def calculate_max_sensitivity_quantus(heatmap, input_image, model, device, label_id, perturbation_size=0.1, nr_samples=10, disable_warnings=True):
    """
    Calculates the Max-Sensitivity using quantus.
    Expects heatmap HW float, input_image HWC uint8/float.
    Expects device as torch.device object, label_id as int.
    """
    if not isinstance(heatmap, np.ndarray) or heatmap.ndim != 2:
        raise ValueError("MaxSensitivity expects heatmap as HW numpy array.")
    if not isinstance(input_image, np.ndarray) or input_image.ndim != 3:
        raise ValueError("MaxSensitivity expects input_image as HWC numpy array.")
    if not isinstance(device, torch.device):
         print("Warning (MaxSens): Received device string, converting to torch.device. Pass the object directly.")
         device = torch.device(device)
    if not isinstance(label_id, int):
         raise ValueError("MaxSensitivity expects label_id as an integer.")


    # Normalize the heatmap
    heatmap_norm = normalize_heatmap(heatmap)
    a_batch = np.expand_dims(heatmap_norm, axis=0) # (1, H, W)

    # Prepare image batch
    input_image_float = input_image.astype(np.float32)
    x_batch = np.expand_dims(input_image_float, axis=0) # (1, H, W, C)

    # Prepare label batch
    y_batch = np.array([label_id]) # (1,)

    # 1. Instantiate the metric
    # Note: Quantus MaxSensitivity often requires nr_samples, perturb_func etc. during init
    max_sensitivity_metric = MaxSensitivity(
        nr_samples=nr_samples, # Number of perturbation samples
        # lower_bound=0.2, # Example: lower bound for perturbation region size
        # norm_numerator=quantus.fro_norm, # How to measure explanation difference
        # norm_denominator=quantus.fro_norm, # How to measure input difference
        # perturb_func=quantus.uniform_noise, # Example perturbation
        # similarity_func=quantus.difference, # How to compare explanations
        abs=True,
        normalise=True,
        disable_warnings=disable_warnings
    )

    # 2. Call the metric instance
    try:
        sensitivity_score = max_sensitivity_metric(
            model=model,
            x_batch=x_batch, # Pass the image batch (expects N, H, W, C or N, C, H, W based on Quantus version/backend)
            y_batch=y_batch, # Pass the label batch
            a_batch=a_batch, # Pass the reference explanation batch
            device=device,   # Pass the torch device object
            # MaxSensitivity needs an explain_func to generate perturbed explanations
            # You need to provide a function that takes (model, inputs, targets, **kwargs)
            # and returns explanations (numpy N, H, W)
            explain_func=None, # *** Placeholder: You MUST provide a valid explain_func or precompute perturbed explanations ***
            explain_func_kwargs={} # Arguments for explain_func if needed
        )
        # Extract score
        if isinstance(sensitivity_score, list): return sensitivity_score[0]
        elif isinstance(sensitivity_score, dict): return list(sensitivity_score.values())[0]
        else: return sensitivity_score

    except Exception as e:
        print(f"Error during Quantus MaxSensitivity calculation: {e}")
        # ** Common Error: explain_func is required by MaxSensitivity but not provided **
        if "explain_func" in str(e):
             print("ERROR HINT: MaxSensitivity requires a valid 'explain_func' argument to recompute explanations on perturbed inputs.")
        raise


def calculate_focus_quantus(heatmap, segmentation_mask):
    """
    Calculates the Focus using quantus.
    Expects heatmap HW float, segmentation_mask HW integer.
    """
    if not isinstance(heatmap, np.ndarray) or heatmap.ndim != 2:
        raise ValueError("Focus expects heatmap as HW numpy array.")
    if not isinstance(segmentation_mask, np.ndarray) or segmentation_mask.ndim != 2:
        raise ValueError("Focus expects segmentation_mask as HW numpy array.")

    # Normalize and Binarize explanation
    heatmap_norm = normalize_heatmap(heatmap)
    explanation_bin = (heatmap_norm > 0.5).astype(int) # Use a threshold (0.5 is common)
    a_batch = np.expand_dims(explanation_bin, axis=0) # (1, H, W)

    # Binarize segmentation mask (assuming positive values indicate the class region)
    # Important: Ensure the mask corresponds to the *target class* for which the heatmap was generated
    segmentation_mask_bin = (segmentation_mask > 0).astype(int)
    y_batch = np.expand_dims(segmentation_mask_bin, axis=0) # Quantus Focus might expect the mask here as 'y_batch' (N, H, W)

    # 1. Instantiate the metric
    focus_metric = Focus(
        return_aggregate=True,
        disable_warnings=True
        )

    # 2. Call the metric instance <<< CORRECTION >>>
    try:
        # Focus compares a_batch (binary explanation) with y_batch (binary target mask)
        focus_score = focus_metric(
            a_batch=a_batch,
            y_batch=y_batch,
            # model, x_batch, device are typically NOT needed for Focus
            )
        # Extract score
        if isinstance(focus_score, list): return focus_score[0]
        elif isinstance(focus_score, dict): return list(focus_score.values())[0]
        else: return focus_score

    except Exception as e:
        print(f"Error during Quantus Focus calculation: {e}")
        raise


def calculate_effective_complexity_quantus(heatmap, model, input_image, label_id, device):
    """
    Calculates the Effective Complexity (Box-Counting Dimension) using quantus.
    Expects heatmap HW float, input_image HWC uint8/float.
    Expects device as torch.device object, label_id as int.
    """
    if not isinstance(heatmap, np.ndarray) or heatmap.ndim != 2:
        raise ValueError("EffectiveComplexity expects heatmap as HW numpy array.")
    if not isinstance(input_image, np.ndarray) or input_image.ndim != 3:
        raise ValueError("EffectiveComplexity expects input_image as HWC numpy array.")
    if not isinstance(device, torch.device):
         print("Warning (EffComp): Received device string, converting to torch.device. Pass the object directly.")
         device = torch.device(device) # <<< CORRECTION: Use torch.device object internally
    if not isinstance(label_id, int):
         # Allow None if the underlying metric doesn't strictly need it, but Quantus API expects it
         if label_id is not None:
             raise ValueError("EffectiveComplexity expects label_id as an integer or None.")


    # Normalize heatmap
    heatmap_normalized = normalize_heatmap(heatmap)
    a_batch = np.expand_dims(heatmap_normalized, axis=0) # (1, H, W)

    # Prepare image batch
    input_image_float = input_image.astype(np.float32)
    x_batch = np.expand_dims(input_image_float, axis=0) # (1, H, W, C)

    # Prepare label batch (handle None)
    if label_id is None:
        # Quantus API requires y_batch. Use a dummy value if None is passed.
        # Check if the specific metric *really* needs it internally. EffectiveComplexity might not.
        print("Warning (EffComp): label_id is None, using dummy label 0 for Quantus API.")
        y_batch = np.array([0])
    else:
        y_batch = np.array([label_id]) # (1,)

    # 1. Instantiate the metric
    eff_complexity_metric = EffectiveComplexity(
        return_aggregate=True,
        disable_warnings=True
    )

    # 2. Call the metric instance
    try:
        complexity_score = eff_complexity_metric(
            model=model,          # Pass the model object
            x_batch=x_batch,      # Pass the input image batch
            y_batch=y_batch,      # Pass the label batch
            a_batch=a_batch,      # Pass the explanation batch
            device=device         # Pass the torch.device object <<< CORRECTION
        )
        # Extract score
        if isinstance(complexity_score, list): return complexity_score[0]
        elif isinstance(complexity_score, dict): return list(complexity_score.values())[0]
        else: return complexity_score

    except Exception as e:
        print(f"Error during Quantus EffectiveComplexity calculation: {e}")
        raise