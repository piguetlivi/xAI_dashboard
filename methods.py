# This file contains the code to implement various explainable AI (XAI) methods for semantic segmentation models.
# It includes methods like Seg-Grad-CAM, Grad-CAM, Feature Ablation, Saliency Maps, LIME, and Guided Grad-CAM.
# The code is designed to work with models from Hugging Face and torchvision, specifically for semantic segmentation tasks.

# Import necessary libraries
from torchvision import transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for dashboard
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image
from captum.attr import LayerGradCam, FeatureAblation, Saliency, Lime, GuidedGradCam
from transformers import Mask2FormerImageProcessor
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2
import warnings
from collections import OrderedDict
from io import BytesIO

# Check if GPU is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# -------------------------------------------
# Common helpers
# -------------------------------------------

def prepare_input(image_path):
    """
    Loads and prepares an image for model input.
    Returns both the raw input tensor and the normalized input for attribution methods.
    """
    input_image = Image.open(image_path).convert("RGB")
    input_image = input_image.resize((input_image.width // 2, input_image.height // 2), resample=Image.LANCZOS)

    preprocessing = transforms.Compose([transforms.ToTensor()])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    input_tensor = preprocessing(input_image)
    normalized_inp = normalize(input_tensor).unsqueeze(0).to(device)
    normalized_inp.requires_grad = True

    return input_tensor, normalized_inp

def get_segmentation_output(model_output):
    """
    Extracts the semantic segmentation output from a model.
    Handles both:
    - torchvision models (output as dict with 'out')
    - Hugging Face models (e.g., OneFormer, output as .sem_seg)
    """
    if hasattr(model_output, "sem_seg"):  # Hugging Face models
        return model_output.sem_seg
    elif isinstance(model_output, dict) and 'out' in model_output:  # torchvision models
        return model_output['out']
    else:
        print("DEBUG: model_output type:", type(model_output))
        print("DEBUG: model_output keys/attributes:", dir(model_output))

        raise ValueError("Unknown segmentation output format from model.")
    
# Visualization Function
# It takes a numpy image (H, W, C) [0, 255] uint8 and a numpy heatmap (H, W) [0, 1] float
# and returns an overlayed image as a numpy array (H, W, C) uint8.
def show_cam_on_image(img: np.ndarray, mask: np.ndarray, use_rgb: bool = False, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    """ Overlays the heatmap onto the image. """
    # Ensure mask is float [0, 1]
    mask = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    
    # Ensure image is float [0, 1] if it's not already
    if img.dtype == np.uint8:
        img_float = np.float32(img) / 255
    else:
        img_float = img

    cam = heatmap + img_float
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class SegmentationModelWrapper(nn.Module):
    """
    Wraps a segmentation model to ensure its forward method returns a single tensor,
    handling potential dictionary outputs (like OrderedDict{'out': tensor}).
    This is necessary for compatibility with Captum's gradient-based attribution methods.
    """
    def __init__(self, model):
        """
        Args:
            model (nn.Module): The original segmentation model.
        """
        super().__init__()
        self.model = model

    def forward(self, x):
        """
        Performs a forward pass using the original model and extracts the primary output tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The main output tensor (usually segmentation logits or probabilities).
        """
        output = self.model(x)

        # Handle Hugging Face model outputs (e.g., Mask2Former)
        if hasattr(output, "sem_seg"): # Specific attribute for semantic segmentation logits
             # Check if sem_seg is already a tensor, otherwise try to access logits if nested
            if isinstance(output.sem_seg, torch.Tensor):
                 return output.sem_seg
            # Add checks here if sem_seg itself is an object with logits, e.g. output.sem_seg.logits

        # Handle common dictionary outputs (e.g., torchvision)
        elif isinstance(output, (OrderedDict, dict)):
            if 'out' in output:
                return output['out'] # Common key for torchvision segmentation models
            elif 'logits' in output: # Another possible key
                 return output['logits']
            else:
                # Attempt to find a likely tensor output if common keys fail
                for key, value in output.items():
                    if isinstance(value, torch.Tensor) and value.ndim >= 3: # Check if it looks like segmentation output
                        print(f"Warning: Using fallback key '{key}' from model output dictionary.")
                        return value
                raise KeyError("Could not automatically find the segmentation output tensor in the model's output dictionary. Common keys 'out' or 'logits' not found.")

        # Handle cases where the model might already return a tensor directly
        elif isinstance(output, torch.Tensor):
            return output

        # If output format is unknown
        else:
            raise TypeError(f"Unsupported model output type: {type(output)}. Expected Tensor, dict, OrderedDict, or object with .sem_seg attribute.")


# -------------------------------------------
# XAI Methods
# -------------------------------------------

def seg_grad_cam(model, label, input_tensor, normalized_inp):
    """
    Seg-Grad-CAM for Hugging Face Mask2FormerForUniversalSegmentation.

    Args:
        model: Mask2Former model (from Hugging Face)
        label: int – target class ID to explain
        input_tensor: unnormalized image [3, H, W]
        normalized_inp: normalized image tensor [1, 3, H, W]
    
    Returns:
        PIL.Image with CAM overlay
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    normalized_inp = normalized_inp.to(device)
    normalized_inp.requires_grad = True  # allow backward pass through input

    # Load the official processor to do semantic segmentation post-processing
    processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")

    # Buffers to store hooked outputs
    target_activations = []
    target_gradients = []

    # Hook: capture intermediate features and register gradient callback
    def forward_hook(module, input, output):
        output = output.requires_grad_()  # ensure gradients can flow
        target_activations.append(output)

        def backward_hook(grad):
            target_gradients.append(grad)

        output.register_hook(backward_hook)

    # Register the hook to a deep Swin encoder block (we debugged that this exists)
    for name, module in model.named_modules():
        if name == "model.pixel_level_module.encoder.embeddings.patch_embeddings.projection":
            print(f"✔ Hooked into: {name}")
            handle = module.register_forward_hook(forward_hook)
            break

    # First forward pass: get semantic segmentation output (no gradients)
    with torch.no_grad():
        outputs = model(normalized_inp)
        target_size = (input_tensor.shape[1], input_tensor.shape[2])  # (H, W)
        semseg = processor.post_process_semantic_segmentation(outputs, target_sizes=[target_size])[0]

    # Generate binary mask for selected label
    mask = (semseg == label).float()

    # Resize the binary mask to match the spatial resolution of predicted masks
    _, num_queries, Hm, Wm = outputs.masks_queries_logits.shape
    mask_resized = torch.nn.functional.interpolate(
        mask.unsqueeze(0).unsqueeze(0), size=(Hm, Wm), mode="nearest"
    ).to(device)

    # After mask resizing
    print("Shape of mask_resized:", mask_resized.shape) # Debugging output
    debug_mask = mask_resized.squeeze().detach().cpu().numpy()
    plt.imshow(debug_mask, cmap="gray")
    plt.title("Resized Class Mask")
    plt.savefig("debug_mask.png")

    # Second forward pass: this time with gradients
    outputs = model(normalized_inp)
    masks = outputs.masks_queries_logits  # shape: [1, num_queries, Hm, Wm]
    print("Shape of masks:", masks.shape) # Debugging output

    # Get a scalar "score" by dotting class mask with predicted masks
    score = (masks * mask_resized).sum()
    model.zero_grad()
    score.backward()  # this will trigger our hooks
    handle.remove()

    # Gradient-based weight calculation (SegGradCAM core)
    gradients = target_gradients[0]  # could be 3D or 4D
    activations = target_activations[0]

    print("Shape of gradients:", gradients.shape) # Debugging output
    print("Shape of activations:", activations.shape) # Debugging output

    # Support different gradient shapes (your gradients were [1, C, L])
    if gradients.ndim == 4:
        weights = gradients.mean(dim=(2, 3), keepdim=True)
    elif gradients.ndim == 3:
        weights = gradients.mean(dim=2, keepdim=True).unsqueeze(-1)
    elif gradients.ndim == 2:
        weights = gradients.unsqueeze(-1).unsqueeze(-1)
    else:
        raise ValueError(f"Unexpected gradient shape: {gradients.shape}")

    print("Shape of weights:", weights.shape) # Debugging output

    # Compute class activation map
    cam = torch.relu((weights * activations).sum(dim=1)).squeeze()
    print("Shape of cam before normalization:", cam.shape) # Debugging output

    cam -= cam.min()
    cam /= cam.max() + 1e-8  # Normalize to [0, 1]

    print("Shape of cam after normalization:", cam.shape) # Debugging output

    # Convert original image to numpy (HWC) and scale to [0,1]
    input_np = input_tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
    input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())  # just in case

    # Resize CAM to match original image size
    cam_np = cam.detach().cpu().numpy()
    cam_resized = cv2.resize(cam_np, (input_np.shape[1], input_np.shape[0]))  # [W, H]

    # Convert CAM to RGB heatmap overlay
    overlay = show_cam_on_image(input_np, cam_resized, use_rgb=True)

    return Image.fromarray(overlay)


def grad_cam(model, label, input_tensor, normalized_inp):
    """
    Grad-CAM explanation for semantic segmentation models.
    """

    def outputs(normalized_inp, model):
        out = get_segmentation_output(model(normalized_inp))
        return torch.argmax(out, dim=1, keepdim=True)

    out_max = outputs(normalized_inp, model)

    def agg_wrapper(inp):
        out = get_segmentation_output(model(inp))
        selected_inds = torch.zeros_like(out[0:1]).scatter_(1, out_max, 1)
        return (out * selected_inds).sum(dim=(2, 3))

    # Apply Grad-CAM
    layer_gc = LayerGradCam(agg_wrapper, model.classifier)
    gc_attr = layer_gc.attribute(normalized_inp, target=label)

    # Normalize and resize heatmap
    heatmap = (gc_attr - gc_attr.min()) / (gc_attr.max() - gc_attr.min())
    heatmap = heatmap.detach().cpu().numpy()[0, 0]
    heatmap = cv2.resize(heatmap, (input_tensor.shape[2], input_tensor.shape[1]))

    # Overlay on input image
    result = show_cam_on_image(input_tensor.permute(1, 2, 0).cpu().numpy(), heatmap, use_rgb=True)
    return Image.fromarray(result)


def feature_ablation(model, label, input_tensor, normalized_inp):
    """
    Feature Ablation explanation using Captum.
    MODIFIED: Uses a specific SCALAR AGGREGATION wrapper.
    """
    print(f"Calculating Feature Ablation for class index: {label}...")
    model.eval()

    # Convert label to int if it's not already
    try:
        target_label_int = int(label)
    except ValueError:
        print(f"Error: Invalid target label '{label}' for Feature Ablation.")
        h, w = input_tensor.shape[-2:]
        return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))


    # --- SCALAR AGGREGATION Wrapper ---
    # This wrapper calculates a scalar score for the *target* class.
    # We sum the output logits/probabilities for the target class across all pixels.
    def aggregate_output_for_target_class(inp):
        model_output = model(inp)
        output_tensor = get_segmentation_output(model_output) # Shape (N, C, H, W)
        if target_label_int >= output_tensor.shape[1]:
             print(f"Error: Target label index {target_label_int} is out of bounds for model output with {output_tensor.shape[1]} classes.")
             # Return a tensor of zeros or raise error, prevents index error below
             return torch.zeros(output_tensor.shape[0], device=output_tensor.device) # Return zero score per batch item

        # Sum the output for the target class across spatial dimensions (H, W)
        # Keep the batch dimension (N) -> Shape (N,)
        score_per_batch_item = output_tensor[:, target_label_int, :, :].sum(dim=(1, 2))
        #print(f"DEBUG FA Wrapper Score: {score_per_batch_item.item()}") # Debug print
        return score_per_batch_item
    # --- End Wrapper ---

    # Initialize FeatureAblation with the SCALAR aggregation wrapper
    fa = FeatureAblation(aggregate_output_for_target_class)

    # Calculate attribution
    try:
        baselines = torch.zeros_like(normalized_inp)
        # When the wrapper returns a scalar (per batch item), target is usually None or 0.
        # Let's use target=None
        fa_attr = fa.attribute(normalized_inp, baselines=baselines, target=None, perturbations_per_eval=4)
        print(f"Raw FA Attr Stats: Min={fa_attr.min().item():.4f}, Max={fa_attr.max().item():.4f}, Mean={fa_attr.mean().item():.4f}")

        # Check if attribution is effectively zero
        if torch.abs(fa_attr).max() < 1e-6:
             warnings.warn("Feature Ablation attribution map is nearly zero.")
             h, w = input_tensor.shape[-2:]
             return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))

    except Exception as e:
        print(f"Error during Captum Feature Ablation calculation: {e}")
        import traceback
        traceback.print_exc()
        h, w = input_tensor.shape[-2:]
        return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))

    # --- Post-processing (Same as before, sum abs across channels) ---
    fa_attr_processed = fa_attr.abs().sum(dim=1).squeeze(0)
    heatmap_np = fa_attr_processed.cpu().detach().numpy()

    min_val, max_val = np.min(heatmap_np), np.max(heatmap_np)
    if max_val - min_val > 1e-6:
        heatmap_normalized = (heatmap_np - min_val) / (max_val - min_val)
    else:
        print("Warning: Feature Ablation heatmap range is too small after processing.")
        heatmap_normalized = np.zeros_like(heatmap_np)


    heatmap_resized = cv2.resize(heatmap_normalized, (input_tensor.shape[2], input_tensor.shape[1]))
    input_np = input_tensor.permute(1, 2, 0).cpu().numpy()
    input_np = (input_np - np.min(input_np)) / (np.max(input_np) - np.min(input_np) + 1e-6)

    result = show_cam_on_image(input_np, heatmap_resized, use_rgb=True)
    print("Feature Ablation calculation finished.")
    return Image.fromarray(result)


def saliency_maps(model, label, input_tensor, normalized_inp):
    """
    Saliency map visualization using raw gradients.
    """

    saliency = Saliency(model)
    saliency_attr = saliency.attribute(normalized_inp, target=label)

    saliency_attr = saliency_attr.abs().detach().cpu().numpy()
    saliency_attr = (saliency_attr - saliency_attr.min()) / (saliency_attr.max() - saliency_attr.min())
    heatmap = saliency_attr[0, 0]
    heatmap = cv2.resize(heatmap, (input_tensor.shape[2], input_tensor.shape[1]))

    result = show_cam_on_image(input_tensor.permute(1, 2, 0).cpu().numpy(), heatmap, use_rgb=True, image_weight=0.4)
    return Image.fromarray(result)


def lime(model, label, input_tensor, normalized_inp):
    """
    Local Interpretable Model-Agnostic Explanation (LIME) for segmentation.
    """

    out = get_segmentation_output(model(normalized_inp))
    out_max = torch.argmax(out, dim=1, keepdim=True)

    def agg_wrapper(inp):
        out = get_segmentation_output(model(inp))
        selected_inds = torch.zeros_like(out[0:1]).scatter_(1, out_max, 1)
        return (out * selected_inds).sum(dim=(2, 3))

    lime_explainer = Lime(agg_wrapper)
    baselines = torch.zeros_like(normalized_inp)
    lime_attr = lime_explainer.attribute(
        normalized_inp,
        target=label,
        feature_mask=out_max,
        baselines=baselines,
        n_samples=20
    )

    lime_attr = (lime_attr - lime_attr.min()) / (lime_attr.max() - lime_attr.min())
    heatmap = lime_attr.detach().cpu().numpy()[0, 0]
    heatmap = cv2.resize(heatmap, (input_tensor.shape[2], input_tensor.shape[1]))

    result = show_cam_on_image(input_tensor.permute(1, 2, 0).cpu().numpy(), heatmap, use_rgb=True, image_weight=0.4)
    return Image.fromarray(result)


def guided_grad_cam(original_model, target_layer, label, input_tensor, normalized_inp):
    """
    Generates Guided Grad-CAM explanation for a segmentation model using Captum's class.

    Args:
        original_model (nn.Module): The original PyTorch segmentation model.
        target_layer (nn.Module): The specific layer within the *original_model* for which
                                   Grad-CAM attributions are computed (e.g., the last conv layer).
                                   YOU MUST IDENTIFY AND PROVIDE THIS LAYER.
        label (int): The target class ID for the explanation.
        input_tensor (torch.Tensor): The original input tensor (e.g., unnormalized, CHW or HWC).
                                     Used for visualization and getting shape info.
                                     Should be on the same device as the model expects input.
        normalized_inp (torch.Tensor): The preprocessed (normalized, resized) input tensor
                                       ready to be fed into the model (N, C, H, W).
                                       Must be on the correct device.

    Returns:
        PIL.Image or None: A PIL Image object containing the explanation overlayed on the
                           original image, or None if attribution fails.
    """
    # Ensure the original model is in evaluation mode
    original_model.eval()

    # 1. Wrap the model to handle dictionary outputs
    wrapped_model = SegmentationModelWrapper(original_model)
    wrapped_model.eval() # Also set the wrapper to eval mode

    # 2. Initialize GuidedGradCam
    # Pass the wrapped_model (which returns a tensor) and the target_layer from the original_model.
    # Captum will find the target_layer within the wrapped_model's structure.
    guided_gc = GuidedGradCam(wrapped_model, target_layer)

    # 3. Ensure the input tensor requires gradients for attribution
    if not normalized_inp.requires_grad:
         normalized_inp.requires_grad_()

    # 4. Compute Guided Grad-CAM attribution
    # target=label: For segmentation, this typically computes the gradient of the sum
    #               of the logits/scores for the target class 'label' across all spatial locations.
    # interpolate_mode='bilinear': Recommended by the Grad-CAM paper for smoother results.
    #                              Captum handles upsampling the Grad-CAM part automatically.
    attribution = guided_gc.attribute(normalized_inp, target=label, interpolate_mode='bilinear')

    # Check if attribution calculation was successful
    if attribution is None:
        print("Warning: GuidedGradCam attribution returned None. "
              "This might happen if interpolation failed (e.g., incompatible dimensions).")
        return None

    # 5. Process attribution for visualization
    # The attribution tensor has the same shape as normalized_inp (N, C, H_in, W_in).
    # We usually want a 2D heatmap for visualization.
    # Squeeze the batch dimension (assuming batch size N=1).
    # Take the absolute value and sum across the color channels (dim=0) to get a single heatmap (H_in, W_in).
    heatmap_tensor = attribution.squeeze(0).abs().sum(dim=0)
    heatmap = heatmap_tensor.cpu().detach().numpy() # Move to CPU and convert to numpy

    # 6. Normalize the heatmap to the [0, 1] range for visualization
    heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
    if heatmap_max - heatmap_min > 1e-8: # Avoid division by zero if heatmap is constant
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap = np.zeros_like(heatmap) # Set to zero if constant

    # 7. Prepare the original image for overlay
    # Remove batch dimension if present, permute channels to (H, W, C) for numpy/cv2, move to CPU.
    img_np = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Convert image to uint8 [0, 255] if it's not already (e.g., if it's float [0, 1])
    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0 and img_np.min() >= 0.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            # If it's in another range (e.g., float [-1, 1]), adjust accordingly
            # This basic conversion assumes [0, 1] float or already uint8
            img_np = img_np.astype(np.uint8) # Fallback, might need adjustment

    # 8. Overlay the heatmap on the image using the helper function
    # Ensure the heatmap has the same H, W dimensions as the img_np expects.
    # show_cam_on_image usually expects (H, W) heatmap and (H, W, C) image.
    # Resize heatmap if necessary (though GuidedGradCam output should match input size)
    if heatmap.shape != img_np.shape[:2]:
         heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

    result_np = show_cam_on_image(img_np, heatmap, use_rgb=True) # Assuming show_cam_on_image handles RGB conversion

    # 9. Convert the final numpy array result to a PIL Image
    return Image.fromarray(result_np)