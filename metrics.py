import torch
import numpy as np
import cv2
import torch.nn as nn
from quantus import IROF, MaxSensitivity, PointingGame, EffectiveComplexity

# --- Wrapper for Mask2Former to use with Quantus ---
class Mask2FormerQuantusWrapper(torch.nn.Module):
    def __init__(self, model, processor, label_id):
        super().__init__()
        self.model = model
        self.processor = processor
        self.label_id = label_id

    def forward(self, x):
        # x shape: (B, C, H, W) – channel-first
        # Convert to PIL for processor
        batch_images = []
        for img in x:
            img_np = img.detach().cpu().numpy().transpose(1, 2, 0)  # CHW → HWC
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            batch_images.append(img_pil)

        inputs = self.processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(x.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            seg = self.processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[img.size[::-1] for img in batch_images]
            )[0]

        # Return dummy tensor with shape (1, H, W) for Quantus
        seg = torch.tensor(seg).unsqueeze(0).float().to(x.device)
        return seg

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

def calculate_irof_quantus(input_image, explanation, model, device,
                           segmentation_method="slic", perturb_baseline="mean", disable_warnings=True):

    # Validate inputs
    if not isinstance(input_image, np.ndarray) or input_image.ndim != 3:
        raise ValueError("IROF expects input_image as HWC numpy array.")
    if not isinstance(explanation, np.ndarray) or explanation.ndim != 2:
        raise ValueError("IROF expects explanation as HW numpy array.")

    input_image_float = input_image.astype(np.float32)
    x_chw = np.transpose(input_image_float, (2, 0, 1))  # Convert HWC → CHW
    x_batch = np.expand_dims(x_chw, axis=0)  # (1, C, H, W)
    
    a_batch = np.expand_dims(normalize_heatmap(explanation), axis=0)
    y_batch = np.array([0])  # ← dummy label for compatibility

    model = Mask2FormerQuantusWrapper(original_model, processor, label_id)

    irof_metric = IROF(
        segmentation_method=segmentation_method,
        perturb_baseline=perturb_baseline,
        return_aggregate=True,
        disable_warnings=disable_warnings
    )

    try:
        irof_score = irof_metric(
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            model=model,
            device=device,
        )
        if isinstance(irof_score, list):
            return irof_score[0]
        elif isinstance(irof_score, dict):
            return list(irof_score.values())[0]
        else:
            return irof_score
    except Exception as e:
        print(f"Error during Quantus IROF calculation: {e}")
        raise


# --- Max Sensitivity Implementation using Quantus ---

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


# --- Pointing Game Implementation using Quantus ---
def calculate_pointing_game_quantus(heatmap, segmentation_mask, input_image, model, device, label_id, disable_warnings=True):
    """
    Calculates the Pointing Game metric using Quantus.

    Checks if the pixel with the highest attribution value in the heatmap
    falls within the provided segmentation mask.

    Args:
        heatmap (np.ndarray): The explanation heatmap (HW, float).
        segmentation_mask (np.ndarray): The binary ground truth segmentation mask (HW, int/bool).
        input_image (np.ndarray): The input image (HWC, uint8 or float).
                                   Needed for the Quantus API.
        model (torch.nn.Module): The model used (or a wrapper). Needed for the Quantus API.
        device (torch.device or str): The device to run calculations on.
        label_id (int or None): The target label ID. Needed for Quantus API (y_batch),
                                even if not directly used by Pointing Game logic.
        disable_warnings (bool): Whether to disable Quantus warnings.

    Returns:
        float: The Pointing Game score (1.0 if the max attribution point is
               within the mask, 0.0 otherwise). Returns aggregated score if
               return_aggregate=True during init.

    Raises:
        ValueError: If input arrays have incorrect dimensions or types.
        TypeError: If device is not a torch.device object (after potential conversion).
        Exception: Propagates exceptions from the Quantus calculation.
    """
    # --- Input Validation ---
    if not isinstance(heatmap, np.ndarray) or heatmap.ndim != 2:
        raise ValueError("PointingGame expects heatmap as HW numpy array.")
    if not isinstance(segmentation_mask, np.ndarray) or segmentation_mask.ndim != 2:
        raise ValueError("PointingGame expects segmentation_mask as HW numpy array.")
    if not isinstance(input_image, np.ndarray) or input_image.ndim != 3:
        raise ValueError("PointingGame expects input_image as HWC numpy array.")
    if not isinstance(device, torch.device):
        try:
            print("Warning (PointingGame): Received device string, converting to torch.device. Pass the object directly for robustness.")
            device = torch.device(device)
        except Exception as e:
             raise TypeError(f"Failed to convert device string to torch.device: {e}")
    if not isinstance(label_id, int) and label_id is not None:
        raise ValueError("PointingGame expects label_id as an integer or None.")

    # --- Data Preparation ---
    # Normalize heatmap
    heatmap_normalized = normalize_heatmap(heatmap)
    # Ensure segmentation mask is binary (0 or 1) and integer type
    segmentation_mask_bin = (segmentation_mask > 0).astype(int)

    # Prepare batches (N=1)
    a_batch = np.expand_dims(heatmap_normalized, axis=0)  # (1, H, W)
    s_batch = np.expand_dims(segmentation_mask_bin, axis=0) # (1, H, W) - This is the segmentation mask!

    # Prepare image batch (Quantus API often expects float32)
    input_image_float = input_image.astype(np.float32)
    # Check channel dimension order if necessary, assuming HWC for consistency with other funcs
    # Quantus might internally expect CHW, but often handles conversion.
    # If errors occur related to shape, you might need:
    # x_chw = np.transpose(input_image_float, (2, 0, 1)) # Convert HWC -> CHW
    # x_batch = np.expand_dims(x_chw, axis=0) # (1, C, H, W)
    x_batch = np.expand_dims(input_image_float, axis=0) # (1, H, W, C) - Assuming Quantus handles this

    # Prepare label batch (use dummy if None)
    if label_id is None:
        print("Warning (PointingGame): label_id is None, using dummy label 0 for Quantus API.")
        y_batch = np.array([0]) # Quantus API requires y_batch
    else:
        y_batch = np.array([label_id]) # (1,)

    # --- Instantiate Metric ---
    # abs=True: Often useful for saliency maps where sign doesn't matter for location.
    # normalise=True: Standard practice for many Quantus metrics.
    # return_aggregate=True: Get a single score for the batch (which has size 1 here).
    pg_metric = PointingGame(
        abs=True,
        normalise=True,  # Normalization is done before call usually, but doesn't hurt? Check docs.
        return_aggregate=True,
        disable_warnings=disable_warnings
        # weighted=False # Default, set to True if you want score weighted by mask size
    )

    # --- Call Metric ---
    try:
        pointing_game_score = pg_metric(
            model=model,        # Your model or wrapper
            x_batch=x_batch,    # Input image batch
            y_batch=y_batch,    # Dummy label batch (required by API)
            a_batch=a_batch,    # Your heatmap batch
            s_batch=s_batch,    # Your segmentation mask batch <--- KEY DIFFERENCE
            device=device       # The torch device object
        )

        # Extract score (Quantus might return list or dict)
        if isinstance(pointing_game_score, list):
            # If return_aggregate=True, it should be a list with one element
            return pointing_game_score[0]
        elif isinstance(pointing_game_score, dict):
             # If it returns a dict (less common for aggregate=True), take the first value
            return list(pointing_game_score.values())[0]
        else:
            # Should directly be the score if aggregate=True
            return pointing_game_score

    except Exception as e:
        print(f"Error during Quantus Pointing Game calculation: {e}")
        # Add specific error checks if needed, e.g., for shape mismatches
        raise

# --- Effective Complexity Implementation using Quantus ---

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
         device = torch.device(device) # Use torch.device object internally
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
            device=device         # Pass the torch.device object 
        )
        # Extract score
        if isinstance(complexity_score, list): return complexity_score[0]
        elif isinstance(complexity_score, dict): return list(complexity_score.values())[0]
        else: return complexity_score

    except Exception as e:
        print(f"Error during Quantus EffectiveComplexity calculation: {e}")
        raise

