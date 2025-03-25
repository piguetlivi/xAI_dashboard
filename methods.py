from torchvision import transforms
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for dashboard
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image
from captum.attr import LayerGradCam, FeatureAblation, Saliency, Lime, GuidedBackprop
from transformers import Mask2FormerImageProcessor
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2
from io import BytesIO

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
        if name == "model.pixel_level_module.encoder.encoder.layers.2.blocks.11.output":
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

    # Second forward pass: this time with gradients
    outputs = model(normalized_inp)
    masks = outputs.masks_queries_logits  # shape: [1, num_queries, Hm, Wm]

    # Get a scalar "score" by dotting class mask with predicted masks
    score = (masks * mask_resized).sum()
    model.zero_grad()
    score.backward()  # this will trigger our hooks
    handle.remove()

    # Gradient-based weight calculation (SegGradCAM core)
    gradients = target_gradients[0]  # could be 3D or 4D
    activations = target_activations[0]

    # Support different gradient shapes (your gradients were [1, C, L])
    if gradients.ndim == 4:
        weights = gradients.mean(dim=(2, 3), keepdim=True)
    elif gradients.ndim == 3:
        weights = gradients.mean(dim=2, keepdim=True).unsqueeze(-1)
    elif gradients.ndim == 2:
        weights = gradients.unsqueeze(-1).unsqueeze(-1)
    else:
        raise ValueError(f"Unexpected gradient shape: {gradients.shape}")

    # Compute class activation map
    cam = torch.relu((weights * activations).sum(dim=1)).squeeze()
    cam -= cam.min()
    cam /= cam.max() + 1e-8  # Normalize to [0, 1]

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
    """

    fa = FeatureAblation(model)
    fa_attr = fa.attribute(normalized_inp, target=label, perturbations_per_eval=4)

    heatmap = (fa_attr - fa_attr.min()) / (fa_attr.max() - fa_attr.min())
    heatmap = heatmap.detach().cpu().numpy()[0, 0]
    heatmap = cv2.resize(heatmap, (input_tensor.shape[2], input_tensor.shape[1]))

    result = show_cam_on_image(input_tensor.permute(1, 2, 0).cpu().numpy(), heatmap, use_rgb=True)
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


def guided_grad_cam(model, label, input_tensor, normalized_inp):
    """
    Guided Grad-CAM explanation: combines Guided Backpropagation and Grad-CAM.
    """

    out = get_segmentation_output(model(normalized_inp))
    out_max = torch.argmax(out, dim=1, keepdim=True)

    def wrapper(inp):
        out = get_segmentation_output(model(inp))
        selected_inds = torch.zeros_like(out[0:1]).scatter_(1, out_max, 1)
        return (out * selected_inds).sum(dim=(2, 3))

    # Grad-CAM part
    layer_gc = LayerGradCam(wrapper, model.classifier)
    gc_attr = layer_gc.attribute(normalized_inp, target=label)
    gc_attr = (gc_attr - gc_attr.min()) / (gc_attr.max() - gc_attr.min())
    heatmap = gc_attr.detach().cpu().numpy()[0, 0]
    heatmap = cv2.resize(heatmap, (input_tensor.shape[2], input_tensor.shape[1]))

    # Guided Backpropagation part
    gbp = GuidedBackprop(model)
    guided_attr = gbp.attribute(normalized_inp, target=label)
    guided_attr = guided_attr.detach().cpu().numpy()[0].transpose(1, 2, 0)

    # Combine both
    guided_gradcam = guided_attr * heatmap[..., np.newaxis]
    guided_gradcam = (guided_gradcam - guided_gradcam.min()) / (guided_gradcam.max() - guided_gradcam.min())

    result = show_cam_on_image(input_tensor.permute(1, 2, 0).cpu().numpy(), guided_gradcam, use_rgb=True)
    return Image.fromarray(result)

