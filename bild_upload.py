import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import base64
import io
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from models import fcn_resnet50  # Du kannst hier andere Modelle einfügen
from torchvision.models.segmentation import FCN_ResNet50_Weights

# COCO-Labels automatisch aus PyTorch extrahieren
COCO_LABELS = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.meta["categories"]

# Dash App initialisieren
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout der App
app.layout = html.Div([
    html.H1("Bild-Upload, Segmentierung & Dynamische Labels"),
    
    # Upload-Komponente
    dcc.Upload(
        id='upload-image',
        children=html.Button('Bild hochladen'),
        accept='.png, .jpg, .jpeg',
    ),
    
    # Anzeige des hochgeladenen Bildes
    html.Div(id='output-image-upload'),
    
    # Anzeige der Segmentierung
    html.Div(id='output-segmentation'),
    
    # Dynamisches Label-Dropdown
    dcc.Dropdown(id='label-dropdown', options=[], placeholder="Segmentierte Klasse auswählen"),
])

# Callback zur Verarbeitung des Bildes & Extraktion der Labels
@app.callback(
    [Output('output-image-upload', 'children'),
     Output('output-segmentation', 'children'),
     Output('label-dropdown', 'options')],
    Input('upload-image', 'contents')
)
def process_uploaded_image(contents):
    if contents is None:
        return html.P("Kein Bild hochgeladen."), None, []
    
    # Bild aus Base64-Daten umwandeln
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded)).convert("RGB")
    
    # Bild für das Modell vorbereiten
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    
    # Falls GPU verfügbar, nutze sie
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensor = input_tensor.to(device)
    
    # Modell laden und Segmentierung durchführen
    model = fcn_resnet50().to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    
    # Labels aus der Vorhersage extrahieren
    unique_labels = np.unique(output_predictions)
    label_options = [{"label": COCO_LABELS[label], "value": label} for label in unique_labels if label < len(COCO_LABELS)]
    
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
    
    # Hochgeladenes Bild anzeigen
    uploaded_img_element = html.Img(src=contents, style={'width': '50%', 'marginTop': '10px'})
    
    # Segmentiertes Bild als Base64 umwandeln
    buffered = io.BytesIO()
    segmented_image.save(buffered, format="PNG")
    segmented_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    segmented_img_element = html.Img(src=f'data:image/png;base64,{segmented_base64}', style={'width': '50%', 'marginTop': '10px'})
    
    return uploaded_img_element, segmented_img_element, label_options

# App starten
if __name__ == '__main__':
    app.run_server(debug=True)
