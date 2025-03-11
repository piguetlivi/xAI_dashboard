import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import base64
import io
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from predict_models import (
    predict_fcn_resnet50, predict_fcn_resnet101, predict_deeplabv3_resnet50, 
    predict_deeplabv3_resnet101, predict_deeplabv3_mobilenetv3_large, predict_oneformer
)
from torchvision.models.segmentation import FCN_ResNet50_Weights
from transformers import OneFormerProcessor

# COCO-Labels automatisch aus PyTorch extrahieren
COCO_LABELS = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.meta["categories"]

# OneFormer-Labels aus dem Hugging Face Processor extrahieren
oneformer_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
ONEFORMER_LABELS = oneformer_processor.tokenizer.vocab.keys()
ONEFORMER_LABELS = list(ONEFORMER_LABELS)

# Dash App initialisieren
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout der App
app.layout = html.Div([
    html.H1("Bild-Upload, Segmentierung & Dynamische Labels"),
    dcc.Upload(id='upload-image', children=html.Button('Bild hochladen'), accept='.png, .jpg, .jpeg'),
    dcc.Dropdown(id='model-dropdown', 
                 options=[
                     {'label': 'FCN ResNet50', 'value': 'fcn_resnet50'},
                     {'label': 'FCN ResNet101', 'value': 'fcn_resnet101'},
                     {'label': 'DeepLabV3 ResNet50', 'value': 'deeplabv3_resnet50'},
                     {'label': 'DeepLabV3 ResNet101', 'value': 'deeplabv3_resnet101'},
                     {'label': 'DeepLabV3 MobileNetV3-Large', 'value': 'deeplabv3_mobilenetv3_large'},
                     {'label': 'OneFormer', 'value': 'oneformer'}
                 ], placeholder="Modell auswählen"),
    html.Div(id='output-image-upload'),
    html.Div(id='output-segmentation'),
    dcc.Dropdown(id='label-dropdown', options=[], placeholder="Segmentierte Klasse auswählen"),
])

# Callback zur Verarbeitung des Bildes & Extraktion der Labels
@app.callback(
    [Output('output-image-upload', 'children'),
     Output('output-segmentation', 'children'),
     Output('label-dropdown', 'options')],
    [Input('upload-image', 'contents'),
     Input('model-dropdown', 'value')]
)
def process_uploaded_image(contents, model_name):
    if contents is None or model_name is None:
        return html.P("Kein Bild hochgeladen oder Modell nicht gewählt."), None, []
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded)).convert("RGB")
    
    image_path = "temp_image.jpg"
    image.save(image_path)
    
    # Modell wählen
    model_predictors = {
        'fcn_resnet50': predict_fcn_resnet50,
        'fcn_resnet101': predict_fcn_resnet101,
        'deeplabv3_resnet50': predict_deeplabv3_resnet50,
        'deeplabv3_resnet101': predict_deeplabv3_resnet101,
        'deeplabv3_mobilenetv3_large': predict_deeplabv3_mobilenetv3_large,
        'oneformer': predict_oneformer
    }
    
    predictor = model_predictors.get(model_name)
    if predictor is None:
        return html.P("Ungültiges Modell."), None, []
    
    input_image, output_predictions = predictor(image_path)
    
    unique_labels = np.unique(output_predictions)
    if model_name == 'oneformer':
        label_options = [{'label': ONEFORMER_LABELS[label], 'value': label} for label in unique_labels if label < len(ONEFORMER_LABELS)]
    else:
        label_options = [{'label': COCO_LABELS[label], 'value': label} for label in unique_labels if label < len(COCO_LABELS)]
    
    # Erstelle eine Farbkodierung für die Segmente
    palette = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [255, 255, 0],
        [255, 165, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 255], [128, 128, 128], [255, 20, 147], [255, 215, 0]
    ], dtype=np.uint8)
    
    segmented_image_color = np.zeros((output_predictions.shape[0], output_predictions.shape[1], 3), dtype=np.uint8)
    for class_idx in unique_labels:
        mask = output_predictions == class_idx
        segmented_image_color[mask] = palette[class_idx % len(palette)]
    
    segmented_image = Image.fromarray(segmented_image_color)
    
    buffered = io.BytesIO()
    segmented_image.save(buffered, format="PNG")
    segmented_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    segmented_img_element = html.Img(src=f'data:image/png;base64,{segmented_base64}', style={'width': '50%', 'marginTop': '10px'})
    
    return html.Img(src=contents, style={'width': '50%', 'marginTop': '10px'}), segmented_img_element, label_options

# App starten
if __name__ == '__main__':
    app.run_server(debug=True)
