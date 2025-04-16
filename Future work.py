
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
