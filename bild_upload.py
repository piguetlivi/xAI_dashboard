import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import base64
import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from models import fcn_resnet50  # Falls du andere Modelle nutzen willst, kannst du sie hier importieren
import numpy as np

# Dash App initialisieren
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout der App
app.layout = html.Div([
    html.H1("Bild-Upload und Segmentierung"),
    
    # Upload-Komponente
    dcc.Upload(
        id='upload-image',
        children=html.Button('Bild hochladen'),
        accept='.png, .jpg, .jpeg',
    ),
    
    # Anzeige des hochgeladenen Bildes
    html.Div(id='output-image-upload'),
    
    # Anzeige der Segmentierung
    html.Div(id='output-segmentation')
])

# Callback zur Verarbeitung des hochgeladenen Bildes
@app.callback(
    [Output('output-image-upload', 'children'),
     Output('output-segmentation', 'children')],
    Input('upload-image', 'contents')
)
def process_uploaded_image(contents):
    if contents is None:
        return html.P("Kein Bild hochgeladen."), None
    
    # Bild aus Base64-Daten umwandeln
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded)).convert("RGB")
    
    # Bild vorbereiten für das Modell
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    
    # Falls GPU verfügbar, Bild auf GPU verschieben
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensor = input_tensor.to(device)
    
    # Modell laden und Segmentierung durchführen
    model = fcn_resnet50().to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()


    
    # Erstelle eine Farbkodierung für die Segmente
    palette = np.array([
        [0, 0, 0],        # 0 = Hintergrund (schwarz)
        [128, 0, 0],      # 1 = Klasse 1 (dunkelrot)
        [0, 128, 0],      # 2 = Klasse 2 (dunkelgrün)
        [128, 128, 0],    # 3 = Klasse 3 (gelb)
        [0, 0, 128],      # 4 = Klasse 4 (dunkelblau)
        [128, 0, 128],    # 5 = Klasse 5 (violett)
        [0, 128, 128],    # 6 = Klasse 6 (türkis)
        [255, 255, 0],    # 7 = Klasse 7 (hellgelb)
        [255, 165, 0],    # 8 = Klasse 8 (orange)
        [255, 0, 0],      # 9 = Klasse 9 (hellrot)
        [0, 255, 0],      # 10 = Klasse 10 (hellgrün)
        [0, 0, 255],      # 11 = Klasse 11 (hellblau)
        [255, 255, 255],  # 12 = Klasse 12 (weiß)
        [128, 128, 128],  # 13 = Klasse 13 (grau)
        [255, 20, 147],   # 14 = Klasse 14 (pink)
        [255, 215, 0],    # 15 = Klasse 15 (goldgelb)
    ], dtype=np.uint8)

    # Wandle die Vorhersage in eine farbige Maske um
    segmented_image_color = np.zeros((output_predictions.shape[0], output_predictions.shape[1], 3), dtype=np.uint8)
    for class_idx in np.unique(output_predictions):
        mask = output_predictions == class_idx
        segmented_image_color[mask] = palette[class_idx]

    # Konvertiere das Farbbild zu einem PIL-Image
    segmented_image = Image.fromarray(segmented_image_color)


    print("Output Shape:", output.shape)
    print("Min Value:", output.min().item(), "Max Value:", output.max().item())
    print("Unique Values:", torch.unique(output.argmax(0)).tolist())

    
    # Hochgeladenes Bild anzeigen
    uploaded_img_element = html.Img(src=contents, style={'width': '50%', 'marginTop': '10px'})
    
    # Segmentiertes Bild als Base64 umwandeln
    buffered = io.BytesIO()
    segmented_image.save(buffered, format="PNG")
    segmented_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    segmented_img_element = html.Img(src=f'data:image/png;base64,{segmented_base64}', style={'width': '50%', 'marginTop': '10px'})
    
    return uploaded_img_element, segmented_img_element

# App starten
if __name__ == '__main__':
    app.run_server(debug=True)
