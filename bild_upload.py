import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import base64
import io
import numpy as np
from PIL import Image
import torch
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
from torchvision.models.segmentation import FCN_ResNet50_Weights
from labels import COCO_LABELS, CITYSCAPES_LABELS
from metrics import ( #Imported the quantus metric functions
    calculate_irof_quantus,
    calculate_max_sensitivity_quantus,
    calculate_focus_quantus,
    calculate_effective_complexity_quantus
)

# Initialize Dash application with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Define the layout of the dashboard
app.layout = dbc.Container([
    # Title Row
    dbc.Row([
        dbc.Col(html.H1("XAI Dashboard"), width=12)
    ], className="mb-3"),

    # Main Row for displaying images
    dbc.Row([
        # Original Image (Top Left) - Increased size
        dbc.Col([
            html.H4("Original Image"),
            html.Div(id="output-image-upload", style={"border": "2px solid black", "padding": "10px", "height": "400px"})
        ], width=4),

        # Segmented Image (Top Right) - Increased size
        dbc.Col([
            html.H4("Segmented Image"),
            html.Div(id="output-segmentation", style={"border": "2px solid black", "padding": "10px", "height": "400px"})
        ], width=4),

        # Explanation/Metrics (Bottom Right) - Increased size
        dbc.Col([
            html.H4("Explained Image"),
            html.Div(id="output-method", style={"border": "2px solid black", "padding": "10px", "height": "400px"})
        ], width=4),
    ], style={"margin-bottom": "10px"}),  # Added margin to the main row

    # Row for model, method, and metric selection
    dbc.Row([
        # Filter selection (Model, Method, Label)
        dbc.Col([
            html.H4("Settings"),
            dcc.Upload(id='upload-image', children=html.Button('Upload Image'), accept='.png, .jpg, .jpeg', style={"margin-bottom": "10px"}),
            dcc.Dropdown(id='model-dropdown', options=[{'label': 'FCN ResNet50', 'value': 'fcn_resnet50'}, 
                                                       {'label': 'FCN ResNet101', 'value': 'fcn_resnet101'}, 
                                                       {'label': 'DeepLabV3 ResNet50', 'value': 'deeplabv3_resnet50'}, 
                                                       {'label': 'DeepLabV3 ResNet101', 'value': 'deeplabv3_resnet101'}, 
                                                       {'label': 'DeepLabV3 MobileNetV3-Large', 'value': 'deeplabv3_mobilenetv3_large'}, 
                                                       {'label': 'Mask2Former', 'value': 'mask2former'}], 
                                                       placeholder="Select model", className="mb-2"),
            dcc.Dropdown(id='method-dropdown', options=[{'label': 'Grad-CAM', 'value': 'gradcam'}, 
                                                        {'label': 'Saliency Map', 'value': 'saliency'}, 
                                                        {'label': 'LIME', 'value': 'lime'}, 
                                                        {'label': 'Feature Ablation', 'value': 'ablation'},
                                                        {'label': 'Guided Grad-CAM', 'value': 'guided_gradcam'}, 
                                                        {'label': 'Segmentation Grad-CAM', 'value': 'seg_gradcam'}], 
                                                        placeholder="Select method", className="mb-2"),

            # Label selection dropdown
            dcc.Dropdown(id='label-dropdown', options=[], placeholder="Select Segmented Class", className="mb-2"),
            
            # Store segmentation data
            dcc.Store(id='stored-segmentation-map'),
        ], width=4),  # Width of the filter selection

        # Metric Selection and Display
        dbc.Col([
            html.H4("Select Metrics"),
            dcc.Checklist(
                id='metrics-checklist',
                options=[
                    {'label': ' IROF', 'value': 'irof'},
                    {'label': ' Max Sensitivity', 'value': 'max_sensitivity'},
                    {'label': ' Focus', 'value': 'focus'},
                    {'label': ' Effective Complexity', 'value': 'effective_complexity'}
                ],
                value=['irof'],  # IROF selected by default
                inline=False
            ),
            html.Div(id='output-metrics'), # Results of the metrics will be displayed here
        ], width=4),  # Width of the metric selection and display

        dbc.Col([
            html.H4("Results"),
            html.Button("Generate PDF", id="export-pdf", className="btn btn-primary"),
        ]),
    ]),
], fluid=True)


@app.callback(
    [Output('output-image-upload', 'children'),
     Output('output-segmentation', 'children'),
     Output('label-dropdown', 'options'),
     Output('stored-segmentation-map', 'data')],
    [Input('upload-image', 'contents'),
     Input('model-dropdown', 'value')],
    prevent_initial_call=True
)

def process_image(contents, model_name):
    if contents is None or model_name is None:
        return html.P("Kein Bild oder Modell ausgewählt."), None, [], None

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded)).convert("RGB")
    input_np = np.array(image) # Convert to numpy array
    image_path = "temp_image.jpg"
    image.save(image_path)

    # Select model
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
        return html.P("Ungültiges Modell ausgewählt."), None, [], None

    # Run model & get segmentation output
    input_image, output_predictions = predictor(image_path)
    segmentation_map = np.array(output_predictions)

    # Convert segmentation output to an image
    color_map = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)
    segmented_image = color_map[segmentation_map]
    segmented_pil = Image.fromarray(segmented_image.astype(np.uint8))

    buffer = io.BytesIO()
    segmented_pil.save(buffer, format="PNG")
    encoded_segmented_img = base64.b64encode(buffer.getvalue()).decode()

    if model_name == 'mask2former':
        label_options = [{'label': CITYSCAPES_LABELS[l], 'value': l}
                         for l in np.unique(segmentation_map)
                         if l < len(CITYSCAPES_LABELS)]
    else:
        label_options = [{'label': COCO_LABELS[l], 'value': l}
                         for l in np.unique(segmentation_map)
                         if l < len(COCO_LABELS)]
        
    print("DEBUG: Unique labels in segmentation:", np.unique(segmentation_map))
    print("DEBUG: label_options:", label_options)
    
    return (
        html.Img(src=contents, style={'width': '100%', 'height': '100%'}),
        html.Img(src=f'data:image/png;base64,{encoded_segmented_img}', style={'width': '100%', 'height': '100%'}),
        label_options,
        {'segmentation_map': segmentation_map.tolist(), 'input_np': input_np.tolist()} #Saves input as numpy array to dcc.store
    )

@app.callback(
    [Output('output-method', 'children'),
     Output('output-metrics', 'children')],
    [Input('method-dropdown', 'value'),
     Input('label-dropdown', 'value'),
     Input('metrics-checklist', 'value')],
    [State('upload-image', 'contents'),
     State('model-dropdown', 'value'),
     State('stored-segmentation-map', 'data')],
    prevent_initial_call=True
)
def run_xai(method, label_id, selected_metrics, contents, model_name, stored_data):
    print("DEBUG: method:", method)
    print("DEBUG: contents is None?", contents is None)
    print("DEBUG: model_name:", model_name)
    print("DEBUG: label_id:", label_id)
    print("DEBUG: selected_metrics:", selected_metrics)

    if not method or not contents or not model_name or label_id is None:
        return html.P("Bitte wähle eine XAI-Methode, lade ein Bild hoch, wähle ein Modell und ein Label aus."), ""

    # Load image from uploaded contents
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded)).convert("RGB")
    image_path = "temp_image.jpg"
    image.save(image_path)

    # Load the correct model
    model_loader = {
        'fcn_resnet50': fcn_resnet50,
        'fcn_resnet101': fcn_resnet101,
        'deeplabv3_resnet50': deeplabv3_resnet50,
        'deeplabv3_resnet101': deeplabv3_resnet101,
        'deeplabv3_mobilenetv3_large': deeplabv3_mobilenetv3_large,
        'mask2former': mask2former_model
    }

    if model_name not in model_loader:
        print("DEBUG: Invalid model selected")
        return html.P("Invalid model selected."), ""

    if model_name == 'mask2former':
        model, _ = model_loader[model_name]()  # ignore processor
    else:
        model = model_loader[model_name]()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Prepare input for the model
    input_tensor, normalized_inp = prepare_input(image_path)
    print("DEBUG: input_tensor shape:", input_tensor.shape)

    #Gets the numpy array from store
    input_np = np.array(stored_data['input_np'])
    segmentation_map = np.array(stored_data['segmentation_map'])

    # Apply selected XAI method
    xai_methods = {
        "gradcam": grad_cam,
        "saliency": saliency_maps,
        "lime": lime,
        "ablation": feature_ablation,
        "guided_gradcam": guided_grad_cam,
        "seg_gradcam": seg_grad_cam
    }

    if method not in xai_methods:
        print("DEBUG: Invalid XAI method selected")
        return html.P(" Invalid XAI method selected"), ""

    # Run the explanation
    explanation = xai_methods[method](model, label_id, input_tensor, normalized_inp)

    if explanation is None:
        print("DEBUG: Explanation is None")
        return html.P("Die Erklärung konnte für diese Methode nicht generiert werden."), ""

    # explanation is a PIL.Image
    explanation_np = np.array(explanation) #Explanation as a numpy array for the metrics
    buffer = io.BytesIO()
    explanation.save(buffer, format="PNG")
    encoded_explanation = base64.b64encode(buffer.getvalue()).decode()

    # Calculate metrics, calls the quantus metric functions.  Also uses input_np from store, and explanation
    metrics_output = [] #List for multiple metric output

    if 'irof' in selected_metrics:
        irof = calculate_irof_quantus(input_np, explanation_np, model, device)
        metrics_output.append(html.P(f"IROF: {irof:.4f}")) #Append results
    if 'max_sensitivity' in selected_metrics:
        max_sensitivity = calculate_max_sensitivity_quantus(explanation_np, input_np, model, device, label_id)
        metrics_output.append(html.P(f"Max Sensitivity: {max_sensitivity:.4f}")) #Append results
    if 'focus' in selected_metrics:
        focus = calculate_focus_quantus(explanation_np, segmentation_map)
        metrics_output.append(html.P(f"Focus: {focus:.4f}")) #Append results
    if 'effective_complexity' in selected_metrics:
        effective_complexity = calculate_effective_complexity_quantus(explanation_np)
        metrics_output.append(html.P(f"Effective Complexity: {effective_complexity:.4f}")) #Append results


    print("DEBUG: Successfully generated explanation image")
    print("DEBUG: method:", method)
    print("DEBUG: model_name:", model_name)
    print("DEBUG: label_id:", label_id)

    return html.Img(src=f"data:image/png;base64,{encoded_explanation}", style={"width": "100%", "height": "100%"}), metrics_output

if __name__ == '__main__':
    app.run_server(debug=True)