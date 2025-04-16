# This file contains the implementation of various metrics for evaluating model explanations.
# It includes functions for calculating Intersection over Union (IoU), Pointing Game, and Effective Complexity.
# The metrics are designed to work with heatmaps and segmentation masks, and they utilize the Quantus library for some calculations.

import torch
import numpy as np
import cv2
import torch.nn as nn
from quantus import PointingGame, EffectiveComplexity

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

# --- Intersection over Union Implementation ---
def calculate_iou(
    explanation_hw: np.ndarray,
    gt_mask_hw: np.ndarray,
    explanation_threshold: float = 0.5,
    normalize_explanation: bool = True,
    epsilon: float = 1e-7 # To avoid division by zero
    ) -> float:
    """
    Calculates the Intersection over Union (IoU) between an explanation heatmap
    and a ground truth binary mask.

    The explanation heatmap is first normalized (optional) and then binarized
    using the specified threshold.

    Args:
        explanation_hw: The explanation heatmap (H, W) as float numpy array.
        gt_mask_hw: The binary ground truth mask (H, W) as uint8/int/bool numpy array.
        explanation_threshold: Threshold to binarize the explanation map (after
                                 optional normalization). Defaults to 0.5.
        normalize_explanation: Whether to normalize the explanation heatmap to [0, 1]
                               before thresholding. Defaults to True.
        epsilon: Small value to add to the denominator to avoid division by zero.

    Returns:
        The IoU score (float) between 0.0 and 1.0.

    Raises:
        ValueError: If input arrays have incorrect dimensions or shapes mismatch.
        TypeError: If inputs are not numpy arrays.
    """
    # --- Input Validation ---
    if not isinstance(explanation_hw, np.ndarray) or explanation_hw.ndim != 2:
        raise ValueError("IoU expects explanation_hw as HW numpy array.")
    if not isinstance(gt_mask_hw, np.ndarray) or gt_mask_hw.ndim != 2:
        raise ValueError("IoU expects gt_mask_hw as HW numpy array.")
    if explanation_hw.shape != gt_mask_hw.shape:
        raise ValueError(f"Shape mismatch: explanation {explanation_hw.shape} vs GT mask {gt_mask_hw.shape}")

    # --- Prepare Explanation Mask ---
    expl_proc = explanation_hw.astype(np.float32) # Work with float copy

    # 1. Normalize
    if normalize_explanation:
        expl_proc = normalize_heatmap(expl_proc) # Use existing helper

    # 2. Binarize using threshold
    explanation_mask_bin = (expl_proc >= explanation_threshold).astype(np.uint8)

    # --- Prepare GT Mask (ensure binary 0/1) ---
    gt_mask_bin = (gt_mask_hw > 0).astype(np.uint8)

    # --- Calculate Intersection and Union ---
    # Intersection: Pixels where BOTH masks are 1
    intersection = np.sum(explanation_mask_bin * gt_mask_bin)

    # Union: Pixels where AT LEAST ONE mask is 1
    union = np.sum((explanation_mask_bin + gt_mask_bin) > 0)

    # Calculate IoU
    iou = intersection / (union + epsilon)

    # Clamp the value just in case (shouldn't be needed with epsilon > 0 if union >= 0)
    iou = max(0.0, min(iou, 1.0))

    print(f"Calculated IoU: Intersection={intersection}, Union={union}, IoU={iou:.4f}")
    return float(iou)

# --- Pointing Game Implementation using Quantus ---
def calculate_pointing_game_quantus(heatmap, segmentation_mask, input_image, model, device, label_id, disable_warnings=True):
    """
    Calculates the Pointing Game metric using Quantus.
    Expects segmentation mask s_batch to have shape (N, 1, H, W).
    """
    # --- Input Validation ---
    if not isinstance(heatmap, np.ndarray) or heatmap.ndim != 2:
        raise ValueError("PointingGame expects heatmap as HW numpy array.")
    if not isinstance(segmentation_mask, np.ndarray) or segmentation_mask.ndim != 2:
        raise ValueError("PointingGame expects segmentation_mask as HW numpy array.")
    # (Device and label_id validation...)
    if not isinstance(device, torch.device):
        try: device = torch.device(str(device))
        except Exception as e: raise TypeError(f"Failed to convert device: {e}")

    # --- Data Preparation ---
    heatmap_normalized = normalize_heatmap(heatmap)
    segmentation_mask_bin = (segmentation_mask > 0).astype(np.uint8) # HW

    # Prepare batches (N=1)
    # a_batch: Quantus often expects (N, H, W) for attributions
    a_batch = np.expand_dims(heatmap_normalized, axis=0)    # (1, H, W)

    # --- MODIFICATION START: Add Channel Dimension to s_batch ---
    # s_batch: Reshape HW mask to (N, C, H, W) where C=1
    s_batch_hw = np.expand_dims(segmentation_mask_bin, axis=0) # (1, H, W)
    s_batch = np.expand_dims(s_batch_hw, axis=1)              # (1, 1, H, W) <--- ADD CHANNEL DIM
    print(f"Calculate Pointing Game: Prepared s_batch shape {s_batch.shape}") # Debugging output

    # --- Create DUMMY x_batch with matching (N, C, H, W) shape ---
    # Use float32 type as often expected
    dummy_x_batch = np.zeros_like(s_batch, dtype=np.float32) # Shape (1, 1, H, W)
    print(f"Calculate Pointing Game: Using dummy x_batch shape {dummy_x_batch.shape}") # Debugging output

    # Prepare label batch (y_batch)
    if label_id is None:
        y_batch = np.array([0]) # Dummy label
    else:
        try: y_batch = np.array([int(label_id)])
        except ValueError: raise ValueError(f"Invalid label_id: {label_id}")


    # Instantiate Metric
    pg_metric = PointingGame(
        abs=True,
        normalise=False, # 'a_batch' already normalized
        return_aggregate=False, # Get score for the single instance
        disable_warnings=disable_warnings
    )

    # Call Metric
    try:
        pointing_game_scores = pg_metric(
            model=model,
            x_batch=dummy_x_batch,  # Shape (1, 1, H, W)
            y_batch=y_batch,
            a_batch=a_batch,        # Shape (1, H, W)
            s_batch=s_batch,        # Shape (1, 1, H, W)
            device=device
        )

        # Extract score
        if isinstance(pointing_game_scores, list) and len(pointing_game_scores) > 0:
            return pointing_game_scores[0]
        else:
            print(f"Warning: Unexpected return from PointingGame: {pointing_game_scores}")
            return None

    except Exception as e:
        print(f"Error during Quantus Pointing Game calculation: {e}")
        import traceback
        traceback.print_exc()
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
        # Quantus API requires y_batch. Dummy value is needed if None is passed.
        print("Warning (EffComp): label_id is None, using dummy label 0 for Quantus API.") # Debugging output
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

