# XAI dashboard for semantic segmentation

## Implementation

1. Clone this directory
2. Install the requirments with `pip install -r requirements.txt`

## Use of the app
1. Let run bild_upload.py (if no IDE is available just let run `python dashboard.py` in command line)
2. The app opens
![PNG](/assets/images/starting_page.png)
3. Upload an image
You begin by uploading an image that you want to analyze. The image is used as input for the segmentation models. You can also upload a ground truth image. This is needed for the metrics IoU and Pointing Game.
4. Select a segmentation model
Choose from several pre-trained segmentation models, such as FCN, DeepLabV3, and Mask2Former. The model processes the image and produces a segmentation map.
5. You see the predicted segmentation
![PNG](/assets/images/predicted_segmentation.png)
6. Select a class label
From the predicted segmentation map, choose a target class label you want to explain. The explanation will focus on this label." \
            "The application will automatically select the  class labels. You may have to wait a few seconds for the explanation to be generated, depending on the model and method selected.
7. Choose an explanation method
Pick an xAI (explainable AI) method to understand why the model made its prediction. Available methods include Grad-CAM, LIME, Saliency, and others." \
            "Plese note that some methods may not be compatible with all models. If you select a method that is not compatible, the application will notify you.
8. View results and evaluate
The dashboard displays the original image, predicted segmentation, and explanation heatmap. You can also select metrics to evaluate the explanation quality.



