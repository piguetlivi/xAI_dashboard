# Refactored methods.py (no globals)

from torchvision import transforms
from PIL import Image
from captum.attr import LayerGradCam, FeatureAblation, Saliency, Lime, GuidedBackprop
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Common preprocessing helpers
def prepare_input(image_path):
    input_image = Image.open(image_path).convert("RGB")
    input_image = input_image.resize((input_image.width // 2, input_image.height // 2), resample=Image.LANCZOS)

    preprocessing = transforms.Compose([transforms.ToTensor()])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    input_tensor = preprocessing(input_image)
    normalized_inp = normalize(input_tensor).unsqueeze(0).to(device)
    normalized_inp.requires_grad = True

    return input_tensor, normalized_inp


# --- XAI Methods ---

def grad_cam(model, label, input_tensor, normalized_inp):
    def outputs(normalized_inp, model):
        out = model(normalized_inp)['out']
        return torch.argmax(out, dim=1, keepdim=True)

    out_max = outputs(normalized_inp, model)

    def agg_wrapper(inp):
        out = model(inp)['out']
        selected_inds = torch.zeros_like(out[0:1]).scatter_(1, out_max, 1)
        return (out * selected_inds).sum(dim=(2, 3))

    targets = [2, 6, 7, 14, 15, 19]
    if label not in targets:
        return None

    layer_gc = LayerGradCam(agg_wrapper, model.classifier)
    gc_attr = layer_gc.attribute(normalized_inp, target=label)

    heatmap = (gc_attr - gc_attr.min()) / (gc_attr.max() - gc_attr.min())
    heatmap = heatmap.detach().cpu().numpy()[0, 0]
    heatmap = cv2.resize(heatmap, (input_tensor.shape[2], input_tensor.shape[1]))

    image_np = input_tensor.permute(1, 2, 0).detach().cpu().numpy()
    result = show_cam_on_image(image_np, heatmap, use_rgb=True)

    return Image.fromarray(result)


def feature_ablation(model, label, input_tensor, normalized_inp):
    fa = FeatureAblation(model)
    fa_attr = fa.attribute(normalized_inp, target=label, perturbations_per_eval=4)

    heatmap = (fa_attr - fa_attr.min()) / (fa_attr.max() - fa_attr.min())
    heatmap = heatmap.detach().cpu().numpy()[0, 0]
    heatmap = cv2.resize(heatmap, (input_tensor.shape[2], input_tensor.shape[1]))

    result = show_cam_on_image(input_tensor.permute(1, 2, 0).cpu().numpy(), heatmap, use_rgb=True)
    return Image.fromarray(result)


def saliency_maps(model, label, input_tensor, normalized_inp):
    saliency = Saliency(model)
    saliency_attr = saliency.attribute(normalized_inp, target=label)

    saliency_attr = saliency_attr.abs().detach().cpu().numpy()
    saliency_attr = (saliency_attr - saliency_attr.min()) / (saliency_attr.max() - saliency_attr.min())
    heatmap = saliency_attr[0, 0]
    heatmap = cv2.resize(heatmap, (input_tensor.shape[2], input_tensor.shape[1]))

    result = show_cam_on_image(input_tensor.permute(1, 2, 0).cpu().numpy(), heatmap, use_rgb=True, image_weight=0.4)
    return Image.fromarray(result)


def lime(model, label, input_tensor, normalized_inp):
    out = model(normalized_inp)['out']
    out_max = torch.argmax(out, dim=1, keepdim=True)

    def agg_wrapper(inp):
        out = model(inp)['out']
        selected_inds = torch.zeros_like(out[0:1]).scatter_(1, out_max, 1)
        return (out * selected_inds).sum(dim=(2, 3))

    targets = [2, 6, 7, 14, 15, 19]
    if label not in targets:
        return None

    lime_explainer = Lime(agg_wrapper)
    baselines = torch.zeros_like(normalized_inp)
    lime_attr = lime_explainer.attribute(normalized_inp, target=label, feature_mask=out_max, baselines=baselines, n_samples=20)

    lime_attr = (lime_attr - lime_attr.min()) / (lime_attr.max() - lime_attr.min())
    heatmap = lime_attr.detach().cpu().numpy()[0, 0]
    heatmap = cv2.resize(heatmap, (input_tensor.shape[2], input_tensor.shape[1]))

    result = show_cam_on_image(input_tensor.permute(1, 2, 0).cpu().numpy(), heatmap, use_rgb=True, image_weight=0.4)
    return Image.fromarray(result)


def guided_grad_cam(model, label, input_tensor, normalized_inp):
    out = model(normalized_inp)['out']
    out_max = torch.argmax(out, dim=1, keepdim=True)

    def wrapper(inp):
        out = model(inp)['out']
        selected_inds = torch.zeros_like(out[0:1]).scatter_(1, out_max, 1)
        return (out * selected_inds).sum(dim=(2, 3))

    # Grad-CAM
    layer_gc = LayerGradCam(wrapper, model.classifier)
    gc_attr = layer_gc.attribute(normalized_inp, target=label)
    gc_attr = (gc_attr - gc_attr.min()) / (gc_attr.max() - gc_attr.min())
    heatmap = gc_attr.detach().cpu().numpy()[0, 0]
    heatmap = cv2.resize(heatmap, (input_tensor.shape[2], input_tensor.shape[1]))

    # Guided Backprop
    gbp = GuidedBackprop(model)
    guided_attr = gbp.attribute(normalized_inp, target=label)
    guided_attr = guided_attr.detach().cpu().numpy()[0].transpose(1, 2, 0)

    # Combine
    guided_gradcam = guided_attr * heatmap[..., np.newaxis]
    guided_gradcam = (guided_gradcam - guided_gradcam.min()) / (guided_gradcam.max() - guided_gradcam.min())

    result = show_cam_on_image(input_tensor.permute(1, 2, 0).cpu().numpy(), guided_gradcam, use_rgb=True)
    return Image.fromarray(result)