# This script creates a Dash web application for visualizing and explaining image segmentation models
# using various XAI methods and evaluating explanations with metrics, including against Ground Truth.

# Importing necessary libraries
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import base64
import io
import numpy as np
from PIL import Image
import torch
import cv2
from methods import (
    prepare_input,
    grad_cam, saliency_maps, lime, feature_ablation, guided_grad_cam, seg_grad_cam
)
from models import (
    fcn_resnet50, fcn_resnet101,
    deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenetv3_large, mask2former_model
)
from predict_models import (
    predict_fcn_resnet50, predict_fcn_resnet101, predict_deeplabv3_resnet50,
    predict_deeplabv3_resnet101, predict_deeplabv3_mobilenetv3_large, predict_mask2former
)
from labels import COCO_LABELS, CITYSCAPES_LABELS

# Import metric calculation functions from metrics.py
from metrics import (
    calculate_irof_quantus,
    calculate_max_sensitivity_quantus,
    calculate_pointing_game_quantus,
    calculate_effective_complexity_quantus
)

# Initialize Dash application with Bootstrap theme and enable callback suppression
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Define the layout of the dashboard
app.layout = dbc.Container([
    # Title Row
    dbc.Row([
        dbc.Col(html.H1("XAI Dashboard for Image Segmentation"), width=12) # Changed title slightly
    ], className="mb-3"),

    # Main Row for displaying images (Original, Segmentation, Explanation)
    dbc.Row([
        # Original Image (Top Left)
        dbc.Col([
            html.H4("Original Image"),
            # Container for the uploaded image display
            html.Div(id="output-image-upload", style={"border": "1px solid lightgrey", "padding": "10px", "height": "400px", "display": "flex", "justify-content": "center", "align-items": "center"})
        ], width=4),

        # Segmented Image (Top Middle) - Result from the model prediction
        dbc.Col([
            html.H4("Predicted Segmentation"), # Clarified title
            # Container for the predicted segmentation map display
            html.Div(id="output-segmentation", style={"border": "1px solid lightgrey", "padding": "10px", "height": "400px", "display": "flex", "justify-content": "center", "align-items": "center"})
        ], width=4),

        # Explanation Image (Top Right) - Result from the XAI method
        dbc.Col([
            html.H4("Explanation Heatmap"), # Clarified title
            # Container for the explanation heatmap display
            html.Div(id="output-method", style={"border": "1px solid lightgrey", "padding": "10px", "height": "400px", "display": "flex", "justify-content": "center", "align-items": "center"})
        ], width=4),
    ], style={"margin-bottom": "20px"}),  # Added more margin below the image row

    # Row for controls (Uploads, Selections) and results (Metrics, Export)
    dbc.Row([
        # Column for Uploads and Model/Method/Label Selections
        dbc.Col([
            html.H4("Settings"),
            # Upload component for the main image
            dcc.Upload(
                id='upload-image',
                children=html.Button('1. Upload Image'),
                accept='.png, .jpg, .jpeg',
                style={"margin-bottom": "5px", "display": "block"} # Ensure block display
            ),
            # --- NEW: Upload component for the optional Ground Truth (GT) mask ---
            dcc.Upload(
                id='upload-gt-mask',
                children=html.Button('2. Upload Ground Truth Mask (Optional)'),
                accept='.png, .jpg, .jpeg', # Accepts common image formats for masks
                style={"margin-bottom": "10px", "display": "block"}
            ),
            # --- NEW: Div to display the status of the GT mask upload ---
            html.Div(id='gt-mask-status', style={'fontSize': 'small', 'margin-bottom': '15px', 'min-height': '20px'}),

            # Dropdown for selecting the segmentation model
            dcc.Dropdown(
                id='model-dropdown',
                options=[
                    {'label': 'FCN ResNet50', 'value': 'fcn_resnet50'},
                    {'label': 'FCN ResNet101', 'value': 'fcn_resnet101'},
                    {'label': 'DeepLabV3 ResNet50', 'value': 'deeplabv3_resnet50'},
                    {'label': 'DeepLabV3 ResNet101', 'value': 'deeplabv3_resnet101'},
                    {'label': 'DeepLabV3 MobileNetV3-Large', 'value': 'deeplabv3_mobilenetv3_large'},
                    {'label': 'Mask2Former', 'value': 'mask2former'}
                ],
                placeholder="3. Select Model",
                className="mb-2"
            ),
            # Dropdown for selecting the XAI explanation method
            dcc.Dropdown(
                id='method-dropdown',
                options=[
                    {'label': 'Grad-CAM', 'value': 'gradcam'},
                    {'label': 'Saliency Map', 'value': 'saliency'},
                    {'label': 'LIME', 'value': 'lime'},
                    {'label': 'Feature Ablation', 'value': 'ablation'},
                    {'label': 'Guided Grad-CAM', 'value': 'guided_gradcam'},
                    {'label': 'Segmentation Grad-CAM', 'value': 'seg_gradcam'}
                ],
                placeholder="4. Select XAI Method",
                className="mb-2"
            ),
            # Dropdown for selecting the target class label for explanation/metrics
            dcc.Dropdown(
                id='label-dropdown',
                options=[], # Options populated dynamically based on prediction
                placeholder="5. Select Target Class Label",
                className="mb-2"
            ),

            # Hidden storage for predicted segmentation map and original image numpy array
            dcc.Store(id='stored-segmentation-map'),
            # --- NEW: Hidden storage for the processed Ground Truth mask numpy array ---
            dcc.Store(id='stored-gt-mask-data')

        ], width=4),  # End of Settings column

        # Column for Metric Selection and Display
        dbc.Col([
            html.H4("Metrics"),
            # Checklist for selecting which metrics to calculate
            dcc.Checklist(
                id='metrics-checklist',
                options=[
                    {'label': ' IROF', 'value': 'irof'},
                    {'label': ' Max Sensitivity', 'value': 'max_sensitivity'},
                    # Updated label to indicate GT mask requirement
                    {'label': ' Pointing Game (needs GT Mask)', 'value': 'pointing_game'},
                    {'label': ' Effective Complexity', 'value': 'effective_complexity'}
                ],
                value=['effective_complexity'],  # Default selected metric(s)
                inline=False, # Display options vertically
                className="mb-3"
            ),
            # Div where calculated metric results will be displayed
            html.Div(id='output-metrics', style={"border": "1px dashed lightgrey", "padding": "10px", "min-height": "100px"}),
        ], width=4),  # End of Metrics column

        # Column for Export functionality (Placeholder)
        dbc.Col([
            html.H4("Export"),
            # Button to trigger PDF export (functionality not implemented here)
            html.Button("Generate PDF Report", id="export-pdf", className="btn btn-primary", disabled=True), # Initially disabled
             html.P("(Export functionality not yet implemented)", style={'fontSize': 'small', 'color': 'grey'})
        ], width=4), # End of Export column
    ]), # End of Controls/Results row
], fluid=True) # Use fluid container for better responsiveness


# === CALLBACKS ===

# Callback 1: Process uploaded image, run model prediction, update displays and label options.
@app.callback(
    [Output('output-image-upload', 'children'),      # Display original image
     Output('output-segmentation', 'children'),     # Display predicted segmentation
     Output('label-dropdown', 'options'),           # Populate label choices
     Output('stored-segmentation-map', 'data')],    # Store prediction and input array
    [Input('upload-image', 'contents'),             # Triggered by image upload
     Input('model-dropdown', 'value')],             # Triggered by model selection
    [State('upload-image', 'filename')],            # Get filename for context
    prevent_initial_call=True # Don't run on initial load
)
def process_image(contents, model_name, filename):
    """
    Processes the uploaded image, runs the selected segmentation model,
    updates the image displays, populates the label dropdown based on predicted classes,
    and stores the prediction map and input numpy array.
    """
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No trigger'

    # Check if essential inputs are missing
    if not contents or not model_name:
        print("process_image: Missing content or model name.")
        # Return default messages/empty states
        return (html.P("Please upload an image and select a model.", style={'textAlign': 'center'}),
                html.P("Prediction will appear here.", style={'textAlign': 'center'}),
                [], None)

    print(f"process_image triggered by: {trigger_id}")
    print(f"Processing image: {filename}, Model: {model_name}")

    # Decode the base64 image string
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        image = Image.open(io.BytesIO(decoded)).convert("RGB") # Ensure image is RGB
    except Exception as e:
        print(f"Error opening image: {e}")
        return (html.P(f"Error: Could not read image file '{filename}'.", style={'color': 'red'}),
                html.P("Prediction failed.", style={'textAlign': 'center'}),
                [], None)

    input_np = np.array(image) # Convert PIL image to NumPy array (HWC) for storage

    # --- Select and Run Model Predictor ---
    model_predictors = {
        'fcn_resnet50': predict_fcn_resnet50,
        'fcn_resnet101': predict_fcn_resnet101,
        'deeplabv3_resnet50': predict_deeplabv3_resnet50,
        'deeplabv3_resnet101': predict_deeplabv3_resnet101,
        'deeplabv3_mobilenetv3_large': predict_deeplabv3_mobilenetv3_large,
        'mask2former': predict_mask2former
    }
    predictor = model_predictors.get(model_name)
    if predictor is None:
        print(f"Error: Invalid model '{model_name}' selected.")
        # Should not happen if dropdown is correct, but good practice
        return (html.Img(src=contents, style={'max-width': '100%', 'max-height': '380px'}), # Show original
                html.P(f"Error: Invalid model selected.", style={'color': 'red'}),
                [], None)

    try:
        # Run the prediction function (assuming it takes PIL Image or path and returns PIL Images/arrays)
        # Modify predictor call if it expects a path
        # image_path = "temp_upload.png"; image.save(image_path) # If path needed
        input_image_pil, output_predictions_pil = predictor(image) # Assuming predictor takes PIL image
        # Convert prediction output to NumPy array (HxW)
        predicted_segmentation_map = np.array(output_predictions_pil)
        print(f"Prediction successful. Segmentation map shape: {predicted_segmentation_map.shape}")

    except Exception as e:
        print(f"Error during model prediction ({model_name}): {e}")
        return (html.Img(src=contents, style={'max-width': '100%', 'max-height': '380px'}), # Show original
                html.P(f"Error running prediction for {model_name}: {e}", style={'color': 'red'}),
                [], None)

    # --- Prepare Predicted Segmentation for Display ---
    # Create a colored version of the segmentation map
    # Using a fixed seed for reproducibility of colors if needed, otherwise random
    # np.random.seed(42)
    unique_labels = np.unique(predicted_segmentation_map)
    # Create a color map for the unique labels found in the prediction
    color_map = np.random.randint(0, 255, size=(int(unique_labels.max()) + 1, 3), dtype=np.uint8)
    color_map[0] = [0, 0, 0] # Ensure background (label 0) is black if present

    # Handle potential grayscale map from predictor (though unlikely for segmentation)
    if predicted_segmentation_map.ndim == 2: # Standard HxW map
        segmented_image_color = color_map[predicted_segmentation_map]
    elif predicted_segmentation_map.ndim == 3 and predicted_segmentation_map.shape[2] == 1: # HxWx1 map
         segmented_image_color = color_map[predicted_segmentation_map.squeeze()]
    else: # Unexpected shape
         print(f"Warning: Unexpected prediction map shape: {predicted_segmentation_map.shape}. Displaying as is.")
         segmented_image_color = predicted_segmentation_map # Attempt to display directly


    # Convert colored map back to PIL Image for encoding
    segmented_pil = Image.fromarray(segmented_image_color.astype(np.uint8))
    buffer = io.BytesIO()
    segmented_pil.save(buffer, format="PNG")
    encoded_segmented_img = base64.b64encode(buffer.getvalue()).decode()
    segmented_display = html.Img(src=f'data:image/png;base64,{encoded_segmented_img}', style={'max-width': '100%', 'max-height': '380px'})

    # --- Prepare Label Dropdown Options ---
    # Select the correct label set based on the model
    if model_name == 'mask2former':
        LABELS = CITYSCAPES_LABELS
    else:
        LABELS = COCO_LABELS # Default to COCO for others

    # Create dropdown options only for labels present in the prediction
    label_options = []
    for label_id in unique_labels:
        if 0 <= label_id < len(LABELS): # Check if label ID is valid
            label_name = LABELS[label_id]
            label_options.append({'label': f"{label_id}: {label_name}", 'value': int(label_id)})
        else:
            label_options.append({'label': f"{label_id}: Unknown", 'value': int(label_id)}) # Handle unknown labels

    print(f"DEBUG: Unique predicted labels: {unique_labels}")
    print(f"DEBUG: Generated label options: {label_options}")

    # --- Store Data ---
    # Store the predicted map and the original input image numpy array for later use in XAI/metrics
    # Convert numpy arrays to lists for JSON serialization compatibility with dcc.Store
    stored_data = {
        'predicted_segmentation_map': predicted_segmentation_map.tolist(),
        'input_np': input_np.tolist()
    }

    # Return updated components
    return (html.Img(src=contents, style={'max-width': '100%', 'max-height': '380px'}), # Display original
            segmented_display, # Display colored prediction
            label_options, # Update dropdown
            stored_data) # Store data

# --- NEW Callback 2: Process uploaded Ground Truth mask ---
@app.callback(
    Output('stored-gt-mask-data', 'data'),      # Store processed GT mask array
    Output('gt-mask-status', 'children'),       # Provide user feedback on upload
    Input('upload-gt-mask', 'contents'),        # Triggered by GT mask upload
    State('upload-gt-mask', 'filename'),        # Get filename for feedback messages
    prevent_initial_call=True # Don't run on initial load
)
def process_ground_truth_mask(contents, filename):
    """
    Processes the uploaded Ground Truth mask file.
    Validates, binarizes, and stores the mask as a NumPy array (list) in dcc.Store.
    Provides status feedback to the user.
    """
    if contents is None:
        # No file uploaded or upload cleared
        return None, html.P("Upload a Ground Truth mask (optional).", style={'color': 'grey'})

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        # Attempt to open the uploaded file as an image
        mask_image = Image.open(io.BytesIO(decoded))
        # Convert the PIL image to a NumPy array
        mask_np = np.array(mask_image)
        print(f"GT Mask '{filename}' initial shape: {mask_np.shape}, dtype: {mask_np.dtype}")

        # --- Validate and Process the Mask Array ---
        processed_mask_np = None
        if mask_np.ndim == 2:
            # Already 2D (HxW), likely grayscale or indexed, proceed to binarization
            processed_mask_np = mask_np
            print("GT Mask is 2D.")
        elif mask_np.ndim == 3:
            # 3D (HxWxChannels)
            if mask_np.shape[2] == 1:
                # Grayscale image saved with a channel dimension, remove it
                processed_mask_np = mask_np.squeeze(axis=2)
                print("GT Mask: Converted HxWx1 to HxW.")
            elif mask_np.shape[2] in [3, 4]:
                # RGB or RGBA image, convert to grayscale ('L') first
                print("Warning: Uploaded GT mask appears to be RGB/RGBA. Converting to grayscale.")
                processed_mask_np = np.array(mask_image.convert('L'))
                print(f"GT Mask: Converted color image to grayscale, shape: {processed_mask_np.shape}")
            else:
                # Unsupported number of channels
                raise ValueError(f"Mask has unsupported 3D shape: {mask_np.shape}")
        else:
            # Not 2D or 3D
            raise ValueError(f"Mask has unsupported dimension: {mask_np.ndim}")

        # --- Binarize the Processed Mask ---
        # Assume non-zero pixels belong to the mask. Common for binary masks (0/1 or 0/255).
        # Adjust threshold if needed (e.g., if mask uses specific index values).
        threshold = 0 # Treat any non-zero value as part of the mask
        # Use '>' for threshold, then convert boolean to uint8 (0 or 1)
        binary_mask_np = (processed_mask_np > threshold).astype(np.uint8)

        # Check if the resulting binary mask is empty (all zeros)
        if not np.any(binary_mask_np):
            print(f"Warning: Processed binary GT mask for '{filename}' is empty (all zeros).")
            status_message = f"Warning: GT Mask '{filename}' processed, but seems empty (all black)."
            status_color = 'orange'
        else:
            # Success
            print(f"Successfully processed GT mask '{filename}' to binary, shape: {binary_mask_np.shape}")
            status_message = f"Ground Truth mask '{filename}' loaded ({binary_mask_np.shape[0]}x{binary_mask_np.shape[1]})."
            status_color = 'green'

        # Store the binarized NumPy array as a list in dcc.Store
        stored_data = {'gt_mask': binary_mask_np.tolist()}
        # Return the stored data and the status message
        return stored_data, html.P(status_message, style={'color': status_color})

    except Exception as e:
        # Handle errors during file processing
        print(f"Error processing Ground Truth mask '{filename}': {e}")
        # Return no data and an error message
        return None, html.P(f"Error processing mask '{filename}': Check file format.", style={'color': 'red'})


# Callback 3: Run selected XAI method and calculate selected metrics.
@app.callback(
    [Output('output-method', 'children'),           # Display explanation heatmap
     Output('output-metrics', 'children')],         # Display calculated metric results
    [Input('method-dropdown', 'value'),             # Triggered by XAI method selection
     Input('label-dropdown', 'value'),              # Triggered by target label selection
     Input('metrics-checklist', 'value')],          # Triggered by metric selection changes
    [State('upload-image', 'contents'),             # Get original image content
     State('model-dropdown', 'value'),              # Get selected model name
     State('stored-segmentation-map', 'data'),      # Get stored prediction and input array
     State('stored-gt-mask-data', 'data')],         # --- NEW: Get stored GT mask data ---
    prevent_initial_call=True # Don't run on initial load
)
def run_xai_and_metrics(method_name, label_id, selected_metrics, contents, model_name, stored_pred_data, stored_gt_data):
    """
    Runs the selected XAI method for the chosen model and target label.
    Calculates the selected metrics, using the Ground Truth mask if available for Pointing Game.
    Updates the explanation heatmap display and the metrics results area.
    """
    # Log input states for debugging
    print("\n--- run_xai_and_metrics ---")
    print(f"Triggered with: method='{method_name}', label_id='{label_id}', metrics={selected_metrics}, model='{model_name}'")
    print(f"Image content available: {contents is not None}")
    print(f"Stored prediction data available: {stored_pred_data is not None}")
    print(f"Stored GT mask data available: {stored_gt_data is not None}")

    # --- Input Validation ---
    # Check if all necessary inputs are provided
    if not method_name or not contents or not model_name or label_id is None or stored_pred_data is None:
        missing = []
        if not method_name: missing.append("XAI method")
        if not contents: missing.append("uploaded image")
        if not model_name: missing.append("model")
        if label_id is None: missing.append("target label")
        if stored_pred_data is None: missing.append("stored prediction data")
        message = f"Please select/provide: {', '.join(missing)}."
        print(f"run_xai_and_metrics: Aborting - {message}")
        # Return placeholder messages
        return (html.P("Explanation requires image, model, method, and label.", style={'textAlign': 'center'}),
                html.P("Select metrics to calculate.", style={'textAlign': 'center'}))

    # --- Load Data and Model ---
    try:
        # Load input image numpy array from store
        input_np = np.array(stored_pred_data['input_np'], dtype=np.uint8) # Ensure uint8 HWC
        # Load predicted segmentation map from store
        predicted_segmentation_map = np.array(stored_pred_data['predicted_segmentation_map'])
        print(f"Loaded input_np shape: {input_np.shape}, predicted_map shape: {predicted_segmentation_map.shape}")

        # Decode image again for XAI methods that might need PIL Image or path
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        image_pil = Image.open(io.BytesIO(decoded)).convert("RGB")

        # Load the selected model architecture
        model_loader = {
            'fcn_resnet50': fcn_resnet50, 'fcn_resnet101': fcn_resnet101,
            'deeplabv3_resnet50': deeplabv3_resnet50, 'deeplabv3_resnet101': deeplabv3_resnet101,
            'deeplabv3_mobilenetv3_large': deeplabv3_mobilenetv3_large, 'mask2former': mask2former_model
        }
        if model_name not in model_loader:
            raise ValueError(f"Invalid model name '{model_name}' encountered.")

        # Instantiate the model (handle Mask2Former case)
        if model_name == 'mask2former':
            # Mask2Former might return model and processor
            model, processor = model_loader[model_name]()
            # Store processor if needed by specific XAI methods or Quantus wrapper
            # You might need to adapt the Mask2FormerQuantusWrapper or pass processor if required
        else:
            model = model_loader[model_name]()

        # Set device (GPU if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval() # Move model to device and set to evaluation mode
        print(f"Loaded model '{model_name}' on device '{device}'.")

        # Prepare input tensor for XAI methods (often requires specific normalization)
        # Assuming prepare_input returns the tensor and potentially a normalized numpy array
        input_tensor, normalized_input_np = prepare_input(image_pil) # Use PIL image
        input_tensor = input_tensor.to(device) # Move input tensor to the same device as the model
        print(f"Prepared input tensor shape: {input_tensor.shape}")

    except Exception as e:
        print(f"Error during data/model loading or preparation: {e}")
        return html.P(f"Error loading data or model: {e}", style={'color': 'red'}), []

    # --- Run Selected XAI Method ---
    explanation_np = None # Initialize explanation numpy array
    try:
        xai_methods = {
            "gradcam": grad_cam, "saliency": saliency_maps, "lime": lime,
            "ablation": feature_ablation, "guided_gradcam": guided_grad_cam,
            "seg_gradcam": seg_grad_cam
        }
        if method_name not in xai_methods:
            raise ValueError(f"Invalid XAI method '{method_name}' selected.")

        # Execute the selected XAI function
        # Pass necessary arguments: model, label_id, input_tensor, etc.
        # Adapt arguments based on what each specific XAI function in methods.py requires
        explanation_pil = xai_methods[method_name](model, label_id, input_tensor, normalized_input_np) # Assuming this signature

        if explanation_pil is None:
            raise RuntimeError(f"XAI method '{method_name}' returned None.")

        # Convert explanation PIL Image to NumPy array for metric calculation and display encoding
        explanation_np = np.array(explanation_pil)
        print(f"Generated explanation using '{method_name}'. Explanation shape: {explanation_np.shape}")

        # Encode explanation for display
        buffer = io.BytesIO()
        explanation_pil.save(buffer, format="PNG")
        encoded_explanation = base64.b64encode(buffer.getvalue()).decode()
        explanation_display = html.Img(src=f"data:image/png;base64,{encoded_explanation}", style={'max-width': '100%', 'max-height': '380px'})

    except Exception as e:
        print(f"Error running XAI method '{method_name}': {e}")
        # Display error in explanation area, return empty metrics
        return html.P(f"Error generating explanation with {method_name}: {e}", style={'color': 'red'}), []


    # --- Calculate Selected Metrics ---
    metrics_output = [] # List to store metric result components (html.P)
    heatmap_resized = None # Initialize resized heatmap

    # Check if any metrics are selected AND if explanation was generated
    if selected_metrics and explanation_np is not None:
        try:
            # --- Prepare Heatmap for Metrics (Grayscale + Resize) ---
            if explanation_np.ndim == 3 and explanation_np.shape[2] in [3, 4]: # Color HWC or HWCA
                # Convert to grayscale
                if explanation_np.shape[2] == 4: # RGBA
                    heatmap_gray = cv2.cvtColor(explanation_np, cv2.COLOR_RGBA2GRAY)
                else: # RGB
                    heatmap_gray = cv2.cvtColor(explanation_np, cv2.COLOR_RGB2GRAY)
                print("Metrics Prep: Converted explanation to grayscale.")
            elif explanation_np.ndim == 2: # Already grayscale HW
                heatmap_gray = explanation_np
                print("Metrics Prep: Explanation is already grayscale.")
            else:
                raise ValueError(f"Unexpected explanation shape for heatmap conversion: {explanation_np.shape}")

            # Get target dimensions from the original input image
            H, W = input_np.shape[:2]
            # Resize the grayscale heatmap to match input dimensions
            # Ensure heatmap is float32 for interpolation
            heatmap_resized = cv2.resize(heatmap_gray.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            print(f"Metrics Prep: Resized heatmap to ({H}, {W}).")


            # --- Execute Selected Metric Calculations ---

            # IROF Metric
            if 'irof' in selected_metrics:
                try:
                    # Note: calculate_irof_quantus might need the Mask2Former wrapper if model is Mask2Former
                    # Ensure the function handles model wrapping internally or pass the wrapped model
                    # Pass the main model for now, assuming internal handling or compatibility
                    irof_score = calculate_irof_quantus(
                        input_image=input_np,       # HWC uint8 original image
                        explanation=heatmap_resized,# HW float resized heatmap
                        model=model,                # Loaded model (might need wrapper for Mask2Former)
                        device=device,              # Torch device object or string
                        # label_id is often handled internally by wrapper in your IROF func
                        disable_warnings=True
                    )
                    metrics_output.append(html.P(f"IROF: {irof_score:.4f}"))
                    print(f"Calculated IROF: {irof_score:.4f}")
                except Exception as e:
                    error_msg = f"IROF Calculation Error: {e}"
                    metrics_output.append(html.P(error_msg, style={'color': 'red'}))
                    print(error_msg)

            # Max Sensitivity Metric
            if 'max_sensitivity' in selected_metrics:
                try:
                    # Requires label_id
                    if label_id is None: raise ValueError("Max Sensitivity requires a target label.")
                    max_sens_score = calculate_max_sensitivity_quantus(
                        heatmap=heatmap_resized,    # HW float resized heatmap
                        input_image=input_np,       # HWC uint8 original image
                        model=model,
                        device=device,
                        label_id=int(label_id),     # Ensure label_id is int
                        disable_warnings=True
                    )
                    metrics_output.append(html.P(f"Max Sensitivity: {max_sens_score:.4f}"))
                    print(f"Calculated Max Sensitivity: {max_sens_score:.4f}")
                except Exception as e:
                    error_msg = f"Max Sensitivity Calculation Error: {e}"
                    metrics_output.append(html.P(error_msg, style={'color': 'red'}))
                    print(error_msg)

            # --- Pointing Game Metric (using Ground Truth) ---
            if 'pointing_game' in selected_metrics:
                gt_mask_np = None # Initialize GT mask variable
                # Check if GT mask data was loaded and stored successfully
                if stored_gt_data and 'gt_mask' in stored_gt_data:
                    try:
                        # Load the binarized GT mask from store
                        gt_mask_np = np.array(stored_gt_data['gt_mask'], dtype=np.uint8) # Ensure uint8
                        print(f"Loaded GT mask for Pointing Game, shape: {gt_mask_np.shape}")

                        # --- Dimension Check ---
                        if gt_mask_np.shape == heatmap_resized.shape:
                            # Dimensions match, proceed with calculation
                            pg_score = calculate_pointing_game_quantus(
                                heatmap=heatmap_resized,            # HW float resized heatmap
                                segmentation_mask=gt_mask_np,       # HW uint8 binary GT mask
                                input_image=input_np,               # HWC uint8 original image
                                model=model,
                                device=device,
                                label_id=int(label_id) if label_id is not None else None, # Pass label_id (API req)
                                disable_warnings=True
                            )
                            metrics_output.append(html.P(f"Pointing Game (vs GT): {pg_score:.4f}"))
                            print(f"Calculated Pointing Game (vs GT): {pg_score:.4f}")
                        else:
                            # Dimension mismatch error
                            error_msg = (f"Pointing Game Error: GT mask shape {gt_mask_np.shape} "
                                         f"does not match image/heatmap shape {heatmap_resized.shape}.")
                            metrics_output.append(html.P(error_msg, style={'color': 'red'}))
                            print(error_msg)

                    except Exception as e:
                        # Error during GT mask loading or metric calculation
                        error_msg = f"Pointing Game Calculation Error (GT): {e}"
                        metrics_output.append(html.P(error_msg, style={'color': 'red'}))
                        print(error_msg)
                else:
                    # GT mask was not uploaded or failed to process
                    error_msg = "Pointing Game Skipped: Ground Truth Mask not uploaded or invalid."
                    metrics_output.append(html.P(error_msg, style={'color': 'orange'}))
                    print(error_msg)
            # --- End Pointing Game ---

            # Effective Complexity Metric
            if 'effective_complexity' in selected_metrics:
                try:
                    # Ensure device is torch.device object if needed internally
                    eff_comp_device = device if isinstance(device, torch.device) else torch.device(str(device))
                    eff_comp_score = calculate_effective_complexity_quantus(
                        heatmap=heatmap_resized,        # HW float resized heatmap
                        model=model,
                        input_image=input_np,           # HWC uint8 original image
                        label_id=int(label_id) if label_id is not None else None, # Ensure label_id is int or None
                        device=eff_comp_device
                    )
                    metrics_output.append(html.P(f"Effective Complexity: {eff_comp_score:.4f}"))
                    print(f"Calculated Effective Complexity: {eff_comp_score:.4f}")
                except Exception as e:
                    error_msg = f"Effective Complexity Calculation Error: {e}"
                    metrics_output.append(html.P(error_msg, style={'color': 'red'}))
                    print(error_msg)

        except Exception as prep_err:
            # Handle errors during heatmap preparation (e.g., resize)
            error_msg = f"Error during metrics preparation: {prep_err}"
            metrics_output.append(html.P(error_msg, style={'color': 'red'}))
            print(error_msg)

    # --- Final Checks for Metrics Output ---
    if not selected_metrics:
        # If no metrics were selected, add a message (unless an error already occurred)
        if not any('Error:' in str(p.children) for p in metrics_output if isinstance(p, html.P)):
             metrics_output.append(html.P("No metrics selected."))
    elif selected_metrics and not metrics_output:
         # Metrics were selected, but list is empty (likely due to prep error)
         metrics_output.append(html.P("Metrics calculation failed. Check preparation steps.", style={'color': 'red'}))
    elif selected_metrics and not any(': ' in str(p.children) for p in metrics_output if isinstance(p, html.P)):
        # Metrics selected, list not empty, but no successful score strings (all failed)
         if not any('Error:' in str(p.children) or 'Skipped:' in str(p.children) for p in metrics_output if isinstance(p, html.P)):
             metrics_output.append(html.P("Selected metrics failed. Check logs.", style={'color': 'red'}))


    # Ensure metrics_output is always a list for the return value
    if not isinstance(metrics_output, list):
        metrics_output = [metrics_output] if metrics_output else []

    print("--- run_xai_and_metrics finished. ---")
    # Return the explanation image display and the list of metric result components
    return explanation_display, metrics_output


# Entry point for running the Dash server
if __name__ == '__main__':
    # Run the app server in debug mode (auto-reloads on code change)
    app.run_server(debug=True)