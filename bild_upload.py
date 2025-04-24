# This script creates a Dash web application for visualizing and explaining image segmentation models.
# Using various XAI methods and evaluating explanations with metrics, including against ground truth.

# Importing necessary libraries
import dash
from dash import html, dcc, Input, Output, State, page_registry, page_container
from info_page import layout as info_layout
import dash_bootstrap_components as dbc
import base64
import io
import numpy as np
from PIL import Image
import torch
import traceback 
import cv2
from methods import (
    prepare_input,
    grad_cam, saliency_maps, lime, feature_ablation, guided_grad_cam, seg_grad_cam
)
from models import (
    fcn_resnet50, fcn_resnet101,
    deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenetv3_large, mask2former_model_large, mask2former_model_small
)
from predict_models import (
    predict_fcn_resnet50, predict_fcn_resnet101, predict_deeplabv3_resnet50,
    predict_deeplabv3_resnet101, predict_deeplabv3_mobilenetv3_large, predict_mask2former_small, predict_mask2former_large
)
from labels import COCO_LABELS, CITYSCAPES_LABELS, COCO_COLOR_DICT, CITYSCAPES_COLOR_DICT, ADE20K_COLOR_DICT, ADE20K_LABELS, DEFAULT_COLOR

# Import metric calculation functions from metrics.py
from metrics import (
    calculate_iou,
    calculate_pointing_game_quantus,
    calculate_effective_complexity_quantus
)

external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"
]

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True
)

# === PAGE ROUTING ===
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == "/info":
        return info_layout
    else:
        return dashboard_layout()

# === DASHBOARD PAGE ===
def dashboard_layout():
    return dbc.Container([
        html.Div([
            html.H1("xAI dashboard for image segmentation", style={"display": "inline-block", "margin-right": "20px"}),
            html.A(
                html.I(className="bi bi-info-circle", style={"fontSize": "24px"}),
                href="/info",
                style={"float": "right"}
            )
        ], style={"padding": "10px 20px"}),

        # Main Row for displaying images (Original, Segmentation, Explanation)
        dbc.Row([
            # Original Image (Top Left)
            dbc.Col([
                html.H4("Original image"),
                # Container for the uploaded image display
                html.Div(id="output-image-upload", style={"border": "1px solid lightgrey", "padding": "5px", "height": "400px", "display": "flex", "justify-content": "center", "align-items": "center"})
            ], width=4),

            # Segmented Image (Top Middle) - Result from the model prediction
            dbc.Col([
                html.H4("Predicted segmentation"), # Clarified title
                # Container for the predicted segmentation map display
                html.Div(id="output-segmentation", style={"border": "1px solid lightgrey", "padding": "5px", "height": "400px", "display": "flex", "justify-content": "center", "align-items": "center"})
            ], width=4),

            # Explanation Image (Top Right) - Result from the XAI method
            dbc.Col([
                html.H4("Explanation heatmap"), # Clarified title
                # Container for the explanation heatmap display
                html.Div(id="output-method", style={"border": "1px solid lightgrey", "padding": "5px", "height": "400px", "display": "flex", "justify-content": "center", "align-items": "center"})
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
                # Upload component for the optional Ground Truth (GT) mask
                dcc.Upload(
                    id='upload-gt-mask',
                    children=html.Button('2. Upload ground truth (GT) mask (optional)'),
                    accept='.png, .jpg, .jpeg', # Accepts common image formats for masks
                    style={"margin-bottom": "5px", "display": "block"}
                ),
                # Div to display the status of the GT mask upload
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
                        {'label': 'Mask2Former (small)', 'value': 'mask2former_model_small'},
                        {'label': 'Mask2Former (large)', 'value': 'mask2former_model_large'}
                    ],
                    placeholder="3. Select model",
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
                    placeholder="4. Select xAI method",
                    className="mb-2"
                ),
                # Dropdown for selecting the target class label for explanation/metrics
                dcc.Dropdown(
                    id='label-dropdown',
                    options=[], # Options populated dynamically based on prediction
                    placeholder="5. Select target class label",
                    className="mb-2"
                ),

                # Hidden storage for predicted segmentation map and original image numpy array
                dcc.Store(id='stored-segmentation-map'),
                # Hidden storage for the processed Ground Truth mask numpy array
                dcc.Store(id='stored-gt-mask-data')

            ], width=4),  # End of Settings column

            # Column for Metric Selection and Display
            dbc.Col([
                html.H4("Metrics"),
                # Checklist for selecting which metrics to calculate
                dcc.Checklist(
                    id='metrics-checklist',
                    options=[
                        {'label': ' Intersection over Union (explanation vs GT mask)', 'value': 'iou'},
                        {'label': ' Pointing Game (needs GT mask)', 'value': 'pointing_game'},
                        {'label': ' Effective Complexity', 'value': 'effective_complexity'}
                    ],
                    value=['effective_complexity'],  # Default selected metric(s)
                    inline=False, # Display options vertically
                    className="mb-3"
                ),
                # Div where calculated metric results will be displayed
                html.Div(id='output-metrics', style={"border": "1px dashed lightgrey", "padding": "10px", "min-height": "100px"}),
            ], width=4),  # End of Metrics column
        ]), # End of Controls/Results row
    ], fluid=True) # Use fluid container for better responsiveness


# === CALLBACKS ===

# Callback 1: Process uploaded image, run model prediction, update displays and label options.
@app.callback(
    [Output('output-image-upload', 'children'),     # Display original image
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
        'mask2former_model_small': predict_mask2former_small, 
        'mask2former_model_large': predict_mask2former_large
    }
    predictor = model_predictors.get(model_name)
    if predictor is None:
        print(f"Error: Invalid model '{model_name}' selected.")
        # Should not happen if dropdown is correct, but good practice
        return (html.Img(src=contents, style={'max-width': '100%', 'max-height': '380px'}), # Show original
                html.P(f"Error: Invalid model selected.", style={'color': 'red'}),
                [], None)

    try:
        # --- MODIFICATION START ---
        # Check if the predictor is for mask2former, which expects a file path
        if model_name == 'mask2former_model_small' or model_name == 'mask2former_model_large':
            # Save the PIL image to a temporary file
            temp_image_path = "temp_mask2former_input.png" # Use a specific name
            image.save(temp_image_path)
            print(f"Saved temporary image for Mask2Former at {temp_image_path}")
            # Call the predictor with the file path
            input_processed, output_predictions_processed = predictor(temp_image_path)
            # predictor for mask2former might return tensors or other types, ensure conversion to PIL if needed
            # Assuming predictor returns something convertible to PIL for consistency:
            if isinstance(output_predictions_processed, torch.Tensor):
                 # Example conversion if it returns a tensor map (adjust based on actual output)
                 output_predictions_pil = Image.fromarray(output_predictions_processed.byte().cpu().numpy())
            elif isinstance(output_predictions_processed, np.ndarray):
                 output_predictions_pil = Image.fromarray(output_predictions_processed.astype(np.uint8))
            elif isinstance(output_predictions_processed, Image.Image):
                 output_predictions_pil = output_predictions_processed # Already PIL
            else:
                 raise TypeError(f"Unexpected output type from mask2former predictor: {type(output_predictions_processed)}")

            # input_image_pil handling depends on what predictor returns for the first value
            if isinstance(input_processed, Image.Image):
                 input_image_pil = input_processed
            else:
                 input_image_pil = image # Fallback to original PIL if first return isn't PIL

        else:
            # For other models, assume they accept the PIL image directly
            input_image_pil, output_predictions_pil = predictor(image)
        

        # Convert prediction output to NumPy array (HxW)
        # Ensure the output_predictions_pil is indeed a PIL Image before converting
        if not isinstance(output_predictions_pil, Image.Image):
             raise TypeError(f"Predictor output for {model_name} was not a PIL Image (type: {type(output_predictions_pil)})")

        predicted_segmentation_map = np.array(output_predictions_pil)
        print(f"Prediction successful. Segmentation map shape: {predicted_segmentation_map.shape}")

    except Exception as e:
        print(f"Error during model prediction ({model_name}): {e}")
        import traceback
        traceback.print_exc() # Print full traceback for detailed debugging
        return (html.Img(src=contents, style={'max-width': '100%', 'max-height': '380px'}), # Show original
                html.P(f"Error running prediction for {model_name}: Check logs for details.", style={'color': 'red'}),
                [], None)

    # --- Prepare Predicted Segmentation for Display ---
    # (Rest of the function remains the same)
    # Create a colored version of the segmentation map
    unique_labels = np.unique(predicted_segmentation_map)
    color_map = np.random.randint(0, 255, size=(int(unique_labels.max()) + 1, 3), dtype=np.uint8)
    color_map[0] = [0, 0, 0]

    if predicted_segmentation_map.ndim == 2:
        segmented_image_color = color_map[predicted_segmentation_map]
    elif predicted_segmentation_map.ndim == 3 and predicted_segmentation_map.shape[2] == 1:
         segmented_image_color = color_map[predicted_segmentation_map.squeeze()]
    else:
         print(f"Warning: Unexpected prediction map shape: {predicted_segmentation_map.shape}. Displaying as is.")
         # Attempt conversion if possible, otherwise display raw (might fail)
         try:
            segmented_image_color = predicted_segmentation_map.astype(np.uint8)
         except Exception:
             segmented_image_color = np.zeros_like(input_np) # Fallback to black image


    segmented_pil = Image.fromarray(segmented_image_color.astype(np.uint8))
    buffer = io.BytesIO()
    segmented_pil.save(buffer, format="PNG")
    encoded_segmented_img = base64.b64encode(buffer.getvalue()).decode()
    segmented_display = html.Img(src=f'data:image/png;base64,{encoded_segmented_img}', style={'max-width': '100%', 'max-height': '380px'})

    # --- Prepare Label Dropdown Options ---
    # Select the appropriate color dictionary based on the model
    if model_name == 'mask2former_model_small':
        COLOR_DICT = CITYSCAPES_COLOR_DICT
        LABELS = CITYSCAPES_LABELS # For dropdown later

    elif model_name == 'mask2former_model_large':
        COLOR_DICT = ADE20K_COLOR_DICT
        LABELS = ADE20K_LABELS # For dropdown later
    else:
        COLOR_DICT = COCO_COLOR_DICT
        LABELS = COCO_LABELS # For dropdown later

    # Ensure prediction map is 2D (H, W) for coloring
    if predicted_segmentation_map.ndim == 3 and predicted_segmentation_map.shape[-1] == 1:
        pred_map_2d = predicted_segmentation_map.squeeze(-1)
    elif predicted_segmentation_map.ndim == 2:
        pred_map_2d = predicted_segmentation_map
    else:
        print(f"Error: Unexpected prediction map shape: {predicted_segmentation_map.shape}")
        # Fallback: return black image
        segmented_image_color = np.zeros((input_np.shape[0], input_np.shape[1], 3), dtype=np.uint8)
        pred_map_2d = None # Flag to skip coloring loop

    if pred_map_2d is not None:
        # Ensure the map dtype is suitable for indexing/comparison (e.g., integer)
        if not np.issubdtype(pred_map_2d.dtype, np.integer):
             print(f"Warning: Prediction map dtype is {pred_map_2d.dtype}, converting to int.")
             try:
                 pred_map_2d = pred_map_2d.astype(int)
             except ValueError:
                 print("Error: Could not convert prediction map to integer type for coloring.")
                 pred_map_2d = None # Skip coloring
                 segmented_image_color = np.zeros((input_np.shape[0], input_np.shape[1], 3), dtype=np.uint8)


    if pred_map_2d is not None:
        # Create an empty RGB image to store the colored segmentation
        segmented_image_color = np.zeros((pred_map_2d.shape[0], pred_map_2d.shape[1], 3), dtype=np.uint8)

        # Iterate through unique labels found in the map and assign colors
        for label_id in unique_labels:
            # Convert label_id to int just in case it's not already
            label_id_int = int(label_id)

            # Get the color from the dictionary, use default if label_id is not found
            color = COLOR_DICT.get(label_id_int, DEFAULT_COLOR)

            # Find all pixels with this label_id and set their color
            segmented_image_color[pred_map_2d == label_id_int] = color

            # Optional: Print warning for unexpected labels that use default color
            if label_id_int not in COLOR_DICT:
                 print(f"Warning: Predicted label ID {label_id_int} not found in {model_name}'s color dictionary. Using default color {DEFAULT_COLOR}.")
   

    # Convert the colored NumPy array to a PIL image for display
    segmented_pil = Image.fromarray(segmented_image_color) # segmented_image_color is already uint8
    buffer = io.BytesIO()
    segmented_pil.save(buffer, format="PNG")
    encoded_segmented_img = base64.b64encode(buffer.getvalue()).decode()
    segmented_display = html.Img(src=f'data:image/png;base64,{encoded_segmented_img}', style={'max-width': '100%', 'max-height': '380px'})

    # --- Prepare Label Dropdown Options ---
    # LABELS list was already selected above based on model_name
    label_options = []
    for label_id in unique_labels:
        label_id_int = int(label_id) # Ensure integer
        # Check if label_id is a valid index for the selected LABELS list
        if 0 <= label_id_int < len(LABELS):
            label_name = LABELS[label_id_int]
            # Provide a clear name, handling COCO background specifically
            if model_name != 'mask2former_model_small' and label_id_int == 0 and label_name == '__background__':
                 display_text = f"{label_id_int}: Background"
            else:
                 display_text = f"{label_id_int}: {label_name}"
            label_options.append({'label': display_text, 'value': label_id_int})
        else:
            # Handle labels that might be outside the defined list (e.g., 'unlabeled' 255)
            label_options.append({'label': f"{label_id_int}: Unknown/Other", 'value': label_id_int})

    print(f"DEBUG: Generated label options (using fixed labels): {label_options}")

    # --- Store Data ---
    # Store the raw prediction map (integers) and the original input image
    stored_data = {
        # Ensure stored map is integer list
        'predicted_segmentation_map': predicted_segmentation_map.astype(int).tolist(),
        'input_np': input_np.tolist()
    }

    # Return updated components
    return (html.Img(src=contents, style={'max-width': '100%', 'max-height': '380px'}), # Display original
            segmented_display, # Display prediction colored with FIXED map
            label_options,     # Update dropdown with correct names
            stored_data)       # Store raw prediction and input array
    

# --- Callback 2: Process uploaded Ground Truth mask ---
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
     State('stored-gt-mask-data', 'data')],         # Get stored GT mask data
    prevent_initial_call=True # Don't run on initial load
)
def run_xai_and_metrics(method_name, label_id, selected_metrics, contents, model_name, stored_pred_data, stored_gt_data):
    """
    Runs the selected XAI method for the chosen model and target label.
    Calculates the selected metrics, using the Ground Truth mask if available.
    Updates the explanation heatmap display and the metrics results area.
    """
    # Log input states for debugging
    print("\n--- run_xai_and_metrics ---")
    print(f"Triggered with: method='{method_name}', label_id='{label_id}', metrics={selected_metrics}, model='{model_name}'")
    print(f"Image content available: {contents is not None}")
    print(f"Stored prediction data available: {stored_pred_data is not None}")
    print(f"Stored GT mask data available: {stored_gt_data is not None}")

    # --- Input Validation ---
    if not method_name or not contents or not model_name or label_id is None or stored_pred_data is None:
        missing = []
        if not method_name: missing.append("XAI method")
        if not contents: missing.append("uploaded image")
        if not model_name: missing.append("model")
        if label_id is None: missing.append("target label")
        if stored_pred_data is None: missing.append("stored prediction data (run prediction first)")
        message = f"Please select/provide: {', '.join(missing)}."
        print(f"run_xai_and_metrics: Aborting - {message}")
        # Return default messages for both outputs
        return (html.P(message, style={'textAlign': 'center', 'color': 'orange'}),
                html.P("Select metrics to calculate.", style={'textAlign': 'center'}))

    # Initialize outputs
    explanation_display = html.P("Generating explanation...", style={"textAlign": "center"})
    metrics_output = []
    explanation_np = None # Will store the numpy array of the explanation if successful

    try:
        # --- Load Data ---
        input_np = np.array(stored_pred_data['input_np'], dtype=np.uint8)
        # predicted_segmentation_map = np.array(stored_pred_data['predicted_segmentation_map']) # Not directly needed here, but good to know it exists
        print(f"Loaded input_np shape: {input_np.shape}")

        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        image_pil = Image.open(io.BytesIO(decoded)).convert("RGB")
        print(f"Loaded original PIL image size (WxH): {image_pil.size}")

        # --- Load Model ---
        model_loader = {
            'fcn_resnet50': fcn_resnet50, 'fcn_resnet101': fcn_resnet101,
            'deeplabv3_resnet50': deeplabv3_resnet50, 'deeplabv3_resnet101': deeplabv3_resnet101,
            'deeplabv3_mobilenetv3_large': deeplabv3_mobilenetv3_large,
            'mask2former_model_small': mask2former_model_small,
            'mask2former_model_large': mask2former_model_large
        }
        if model_name not in model_loader:
            raise ValueError(f"Invalid model name '{model_name}' encountered.")

        # Handle Mask2Former loading potentially returning model and processor
        loaded_model_or_tuple = model_loader[model_name]()
        if isinstance(loaded_model_or_tuple, tuple):
            model = loaded_model_or_tuple[0]
            # processor = loaded_model_or_tuple[1] # Processor might not be needed here
        else:
            model = loaded_model_or_tuple

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        print(f"Loaded model '{model_name}' on device '{device}'.")

        # --- Prepare Model Input ---
        # Save temporary file needed by prepare_input if it expects a path
        temp_prep_input_path = "temp_prepare_input.png"
        image_pil.save(temp_prep_input_path)
        print(f"Saved temporary image for prepare_input at {temp_prep_input_path}")

        # Assuming prepare_input returns: processed tensor for model, normalized numpy array
        input_tensor, normalized_input_np = prepare_input(temp_prep_input_path)
        input_tensor = input_tensor.to(device)
        print(f"Prepared input tensor shape: {input_tensor.shape}")
        # Optional: Clean up temporary file
        # import os; try: os.remove(temp_prep_input_path) except OSError: pass


        # --- Define XAI Methods Mapping ---
        # Make sure the function names here match your imports and definitions
        xai_methods = {
            "gradcam": grad_cam,
            "saliency": saliency_maps,
            "lime": lime,
            "ablation": feature_ablation,
            "guided_gradcam": guided_grad_cam, # Assumes this is your corrected captum version
            "seg_gradcam": seg_grad_cam
        }
        if method_name not in xai_methods:
             raise ValueError(f"XAI method '{method_name}' is not defined in the methods mapping.")

        # --- Determine Target Layer (Conditionally for Guided Grad-CAM) ---
        target_layer = None
        if method_name == 'guided_gradcam':
            print(f"Finding target layer for Guided Grad-CAM and model '{model_name}'...")
            try:
                # Define target layers based on known model structures
                if model_name in ['fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101']:
                    target_layer = model.backbone.layer4[-1] # Last block of ResNet backbone
                elif model_name == 'deeplabv3_mobilenetv3_large':
                    # Inspect model structure (e.g., print(model)) to find the correct path
                    # Example: Might be the last conv block in the 'features' part
                    target_layer = model.backbone[-1][-1] # Placeholder - ADJUST THIS PATH!
                # Add elif clauses for other models supporting Guided Grad-CAM

                # Check if layer was found for compatible models
                mask2former_models = ['mask2former_model_small', 'mask2former_model_large'] # M2F incompatible check later
                if target_layer is None and model_name not in mask2former_models:
                    raise ValueError(f"Target layer configuration missing for Guided Grad-CAM with model '{model_name}'.")
                elif target_layer is not None:
                    print(f"Identified target_layer for Guided Grad-CAM: {type(target_layer)}")

            except AttributeError as ae:
                print(f"Error accessing expected structure to find target layer for {model_name}: {ae}")
                raise ValueError(f"Could not find target layer for Guided Grad-CAM in model '{model_name}'. Check model structure and layer path definition.") from ae
            except ValueError as ve:
                raise ve # Re-raise config missing error

        # --- Check Method/Model Compatibility (e.g., Mask2Former) ---
        # Define methods known to be incompatible with M2F in this setup
        incompatible_methods_m2f = ['gradcam', 'saliency', 'lime', 'ablation', 'guided_gradcam']
        mask2former_models = ['mask2former_model_small', 'mask2former_model_large']
        if method_name in incompatible_methods_m2f and model_name in mask2former_models:
            raise ValueError(f"XAI method '{method_name}' is currently not compatible with Mask2Former models.")

        # --- Select and Run the XAI Method ---
        print(f"Running XAI method: {method_name}")
        explanation_pil = None # Initialize

        if method_name == 'guided_gradcam':
            if target_layer is None: # Should have been caught above, but double-check
                 raise ValueError("Target layer required for Guided Grad-CAM but was not identified.")

            # Prepare the *original* image tensor (CPU, CHW, float [0,1]) needed by guided_grad_cam
            if input_np is None: raise ValueError("Original image numpy array (input_np) is missing for Guided Grad-CAM.")
            # Convert HWC uint8 [0,255] to NCHW float [0,1]
            original_input_tensor_cpu = torch.from_numpy(input_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            # Call the specific guided_grad_cam function
            explanation_pil = guided_grad_cam( # Ensure this is the correct imported function name
                original_model=model,
                target_layer=target_layer,
                label=label_id,
                input_tensor=original_input_tensor_cpu, # Original image tensor (CPU)
                normalized_inp=input_tensor           # Processed tensor for model (on device)
            )
        else:
            # Call other methods using the dictionary lookup
            # Assuming they expect: model, label, processed_tensor, normalized_numpy_array
            explanation_pil = xai_methods[method_name](
                model=model,
                label=label_id,
                input_tensor=input_tensor,         # Processed tensor (on device)
                normalized_inp=normalized_input_np # Numpy version (verify if needed by methods)
            )

        # --- Check if the method execution failed silently ---
        if explanation_pil is None:
            raise RuntimeError(f"XAI method '{method_name}' did not return a result (returned None).")

        print(f"XAI method '{method_name}' executed successfully.")

        # --- If successful, process explanation for display and metrics ---
        # 1. Convert to NumPy (original size) - needed for metrics
        explanation_np = np.array(explanation_pil)
        print(f"Converted explanation PIL to NumPy array, shape: {explanation_np.shape}")

        # 2. Resize Explanation PIL *only for display* to match original image size
        if explanation_pil.size != image_pil.size:
            print(f"Resizing explanation from {explanation_pil.size} to {image_pil.size} for display.")
            explanation_pil_resized_display = explanation_pil.resize(image_pil.size, Image.Resampling.LANCZOS)
        else:
            explanation_pil_resized_display = explanation_pil # No resize needed

        # 3. Encode the *resized for display* explanation for the html.Img tag
        buffer = io.BytesIO()
        explanation_pil_resized_display.save(buffer, format="PNG")
        encoded_explanation = base64.b64encode(buffer.getvalue()).decode()

        # 4. Update the display component
        explanation_display = html.Img(
            src=f"data:image/png;base64,{encoded_explanation}",
            style={"maxWidth": "100%", "maxHeight": "380px", "display": "block", "margin": "0 auto"}
        )

    # --- Catch errors during XAI execution or preparation ---
    except (ValueError, RuntimeError, AttributeError, KeyError) as err: # Catch config/runtime/lookup errors
        error_msg = f"Error during XAI setup or execution ({method_name}): {err}"
        print(error_msg)
        traceback.print_exc() # Print traceback for these specific errors
        explanation_display = html.P(error_msg, style={"color": "red", "textAlign": "center"})
        explanation_np = None # Ensure numpy explanation is None if XAI failed

    except Exception as e: # Catch any other unexpected errors
        error_msg = f"Unexpected error during XAI execution for '{method_name}': {e}"
        print(error_msg)
        traceback.print_exc()
        explanation_display = html.P(error_msg, style={"color": "red", "textAlign": "center"})
        explanation_np = None # Ensure numpy explanation is None

    # --- Calculate Selected Metrics ---
    print(f"Calculating selected metrics: {selected_metrics}")
    heatmap_resized_metrics = None # Initialize heatmap used for metrics

    # Proceed only if metrics are selected AND the explanation was generated successfully
    if selected_metrics and explanation_np is not None:
        try:
            # --- Prepare Heatmap for METRICS (Grayscale + Resize to match Original INPUT size) ---
            print("Preparing heatmap for metrics calculation...")
            # Use the original explanation_np (before display resizing)
            if explanation_np.ndim == 3 and explanation_np.shape[2] in [3, 4]:
                # Convert color heatmap to grayscale
                if explanation_np.shape[2] == 4: # RGBA
                    heatmap_gray = cv2.cvtColor(explanation_np, cv2.COLOR_RGBA2GRAY)
                else: # RGB
                    heatmap_gray = cv2.cvtColor(explanation_np, cv2.COLOR_RGB2GRAY)
            elif explanation_np.ndim == 2:
                # Already grayscale
                heatmap_gray = explanation_np
            else:
                raise ValueError(f"Unexpected explanation NumPy array shape for heatmap conversion: {explanation_np.shape}")

            # Target shape for metrics (H, W) from the original input image numpy array
            H, W = input_np.shape[:2]
            # Resize the grayscale heatmap specifically for metric calculations to match input H, W
            heatmap_resized_metrics = cv2.resize(heatmap_gray.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            # Normalize metrics heatmap to [0, 1] range - often required by metric functions
            min_val, max_val = heatmap_resized_metrics.min(), heatmap_resized_metrics.max()
            if max_val - min_val > 1e-8:
                 heatmap_resized_metrics = (heatmap_resized_metrics - min_val) / (max_val - min_val)
            else:
                 heatmap_resized_metrics = np.zeros_like(heatmap_resized_metrics) # Avoid division by zero

            print(f"Metrics Prep: Resized grayscale heatmap for metrics to ({H}, {W}) and normalized.")


            # --- Execute Selected Metric Calculations ---

            # Intersection over Union (IoU) Metric
            if 'iou' in selected_metrics:
                iou_score = "N/A" # Default value
                if stored_gt_data and 'gt_mask' in stored_gt_data:
                    try:
                        gt_mask_np = np.array(stored_gt_data['gt_mask'], dtype=np.uint8)
                        if gt_mask_np.shape == heatmap_resized_metrics.shape[:2]: # Compare H,W
                            iou_score = calculate_iou(
                                explanation_hw=heatmap_resized_metrics, # HW float heatmap [0,1]
                                gt_mask_hw=gt_mask_np              # HW uint8 binary GT mask
                            )
                            iou_score = f"{iou_score:.4f}"
                        else:
                            iou_score = f"Error: GT shape {gt_mask_np.shape} != Heatmap shape {heatmap_resized_metrics.shape[:2]}"
                            print(iou_score)
                    except Exception as e:
                        iou_score = f"Error: {e}"
                        print(f"IoU Calculation Error: {e}")
                        # traceback.print_exc() # Optional detailed traceback for IoU error
                else:
                    iou_score = "GT Mask Missing"
                metrics_output.append(html.P(f"IoU (Expl vs GT): {iou_score}"))

            # Pointing Game Metric (using Ground Truth)
            if 'pointing_game' in selected_metrics:
                pg_score = "N/A"
                if stored_gt_data and 'gt_mask' in stored_gt_data:
                    try:
                        gt_mask_np = np.array(stored_gt_data['gt_mask'], dtype=np.uint8)
                        if gt_mask_np.shape == heatmap_resized_metrics.shape[:2]:
                            # Assuming calculate_pointing_game_quantus handles normalization if needed
                            pg_score = calculate_pointing_game_quantus(
                                heatmap=heatmap_resized_metrics, # Pass the prepared metrics heatmap
                                segmentation_mask=gt_mask_np,
                                input_image=input_np, # Original numpy image might be needed by metric
                                model=model,          # Pass the loaded model
                                device=device,        # Pass the device
                                label_id=int(label_id), # Pass the target label
                                disable_warnings=True   # Optional: suppress quantus warnings
                            )
                            pg_score = f"{pg_score:.4f}"
                        else:
                            pg_score = f"Error: GT shape {gt_mask_np.shape} != Heatmap shape {heatmap_resized_metrics.shape[:2]}"
                            print(pg_score)
                    except Exception as e:
                        pg_score = f"Error: {e}"
                        print(f"Pointing Game Calculation Error (GT): {e}")
                        # traceback.print_exc() # Optional detailed traceback for PG error
                else:
                    pg_score = "GT Mask Missing"
                metrics_output.append(html.P(f"Pointing Game (vs GT): {pg_score}"))

            # Effective Complexity Metric
            if 'effective_complexity' in selected_metrics:
                eff_comp_score = "N/A"
                try:
                     # Ensure device is torch.device object
                    eff_comp_device = device if isinstance(device, torch.device) else torch.device(str(device))
                    # Assuming calculate_effective_complexity_quantus handles normalization if needed
                    eff_comp_score = calculate_effective_complexity_quantus(
                        heatmap=heatmap_resized_metrics, # Pass the prepared metrics heatmap
                        model=model,
                        input_image=input_np,
                        label_id=int(label_id),
                        device=eff_comp_device
                    )
                    eff_comp_score = f"{eff_comp_score:.4f}"
                except Exception as e:
                    eff_comp_score = f"Error: {e}"
                    print(f"Effective Complexity Calculation Error: {e}")
                    # traceback.print_exc() # Optional detailed traceback for EC error
                metrics_output.append(html.P(f"Effective Complexity: {eff_comp_score}"))

        # Catch errors during metric *preparation* (resizing, grayscale)
        except Exception as prep_err:
            error_msg = f"Error during metrics preparation: {prep_err}"
            metrics_output.append(html.P(error_msg, style={'color': 'red'}))
            print(error_msg)
            traceback.print_exc() # Show traceback for prep errors

    # Add messages if metrics selected but XAI failed, or no metrics selected
    elif selected_metrics and explanation_np is None:
        metrics_output.append(html.P("Metrics Skipped: Explanation generation failed.", style={'color': 'orange'}))
    elif not selected_metrics:
        metrics_output.append(html.P("No metrics selected."))

    # Final check if metrics list is empty despite being selected (implies calculation errors)
    if selected_metrics and not metrics_output:
         metrics_output.append(html.P("Metrics calculation failed or produced no output. Check logs.", style={'color': 'red'}))

    print("--- run_xai_and_metrics finished. ---")
    # Return the explanation display (image or error message) and the list of metric results/errors
    return explanation_display, metrics_output

# Entry point for running the Dash server
if __name__ == '__main__':
    app.run(debug=True)