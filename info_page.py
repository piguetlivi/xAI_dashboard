# This file contains the layout for the information page of the Dash application.

import dash
from dash import html, register_page
import dash_bootstrap_components as dbc

# Info content about how the application works
info_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("How this application works"), width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.H4("1. Upload an image"),
            html.P("You begin by uploading an image that you want to analyze. The image is used as input for the segmentation models.")
        ], width=12),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.H4("2. Select a segmentation model"),
            html.P("Choose from several pre-trained segmentation models, such as FCN, DeepLabV3, and Mask2Former. The model processes the image and produces a segmentation map.")
        ], width=12),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.H4("3. Choose an explanation method"),
            html.P("Pick an xAI (explainable AI) method to understand why the model made its prediction. Available methods include Grad-CAM, LIME, Saliency, and others." \
            "Plese note that some methods may not be compatible with all models. If you select a method that is not compatible, the application will notify you.")
        ], width=12),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.H4("4. Select a class label"),
            html.P("From the predicted segmentation map, choose a target class label you want to explain. The explanation will focus on this label." \
            "The application will automatically select the  class labels. You may have to wait a few seconds for the explanation to be generated, depending on the model and method selected.")
        ], width=12),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.H4("5. (Optional) upload ground truth (GT) mask"),
            html.P("You may upload a binary ground truth mask for your image. This enables additional metrics like IoU and Pointing Game to compare explanations with true labels." \
        )
        ], width=12),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.H4("6. View results and evaluate"),
            html.P("The dashboard displays the original image, predicted segmentation, and explanation heatmap. You can also select metrics to evaluate the explanation quality." \
            "")
        ], width=12),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P([
                "For more details on the implementation, refer to the ",
                html.A("GitHub repo", href="https://github.com/piguetlivi/xAI_dashboard/tree/main", target="_blank"),
            ], className="text-muted")
        ], width=12)
    ])
], fluid=True)

# Expose this layout for import
layout = info_layout