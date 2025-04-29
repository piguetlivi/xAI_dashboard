# XAI Dashboard for Semantic Segmentation

This project provides an interactive web application (built with Dash) for visualizing image segmentation models and explaining their predictions using eXplainable AI (XAI) methods. You can select different models and XAI techniques and view the results and associated metrics.

## Features

*   **Model Variety:** Supports popular segmentation models such as FCN, DeepLabV3, and Mask2Former.
*   **XAI Methods:** Implements various XAI techniques including Grad-CAM, Saliency Maps, LIME, Feature Ablation, Guided Grad-CAM, and Segmentation Grad-CAM.
*   **Interactive Visualization:** Displays the original image, the predicted segmentation, and the XAI heatmap side-by-side.
*   **Metric Calculation:** Evaluates the explanations using metrics like Intersection over Union (IoU), Pointing Game, and Effective Complexity.
*   **Ground Truth Support:** Option to upload a Ground Truth mask for specific metrics.
*   **Information Page:** A separate page providing more details about the models, methods, and metrics used.

## Installation

To run the project locally, follow these steps:

1.  Clone this repository to your local machine:
    ```bash
    git clone <YOUR_REPOSITORY_URL>
    ```
    Replace `<YOUR_REPOSITORY_URL>` with the actual URL.
2.  Navigate into the cloned directory.
3.  Install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Using the Application

Follow these steps to start and use the web application:

1.  **Start the Application:**
    Run the main dashboard script. In your terminal, navigate to the project directory and execute:
    ```bash
    python dashboard.py
    ```
    The application will open in your web browser (usually at `http://127.0.0.1:8050/`).
    ![Application Starting Page](/assets/images/starting_page.png)

2.  **Upload Image:**
    Click the "**1. Upload Image**" button. Select an image file (PNG, JPG, JPEG) from your computer. This image will be used as input for the segmentation models.
    Optionally, you can upload a **Ground Truth (GT) mask** by clicking "**2. Upload ground truth (GT) mask (optional)**". A GT mask is required to calculate the **IoU** and **Pointing Game** metrics. It should ideally be a grayscale or binary image.

3.  **Select Segmentation Model:**
    Choose one of the available pre-trained segmentation models (e.g., FCN, DeepLabV3, Mask2Former) from the "**3. Select model**" dropdown. The selected model will perform a prediction on your uploaded image.

4.  **View Predicted Segmentation:**
    Once the model has completed its prediction, the resulting segmentation map will be displayed in the "**Predicted segmentation**" area.
    ![Example Predicted Segmentation](/assets/images/predicted_segmentation.png)

5.  **Select Target Class:**
    In the "**5. Select target class label**" dropdown, choose the specific class whose prediction you want to explain. The available options will be based on the classes predicted by the model in your image.
    *(Note: The dropdown population may take a moment after the segmentation is computed.)*

6.  **Choose Explanation Method:**
    Select an XAI method (e.g., Grad-CAM, LIME) from the "**4. Select xAI method**" dropdown. This method will be used to generate a heatmap showing which image regions were most important for the prediction of the selected target class.
    *(Please note: Not all XAI methods are compatible with all segmentation models. If you select an incompatible combination, an error message will be displayed.)*

7.  **View Results and Evaluate:**
    The generated **Explanation heatmap** will be shown in its respective area. You can now select metrics (e.g., IoU, Pointing Game, Effective Complexity) in the "**Metrics**" section to quantitatively evaluate the quality of the explanation. The results of the selected metrics will be displayed below.

## Further Information

For detailed information about the implemented models, XAI methods, and metrics, click the Info icon (ℹ️) in the top right corner of the dashboard or navigate directly to the `/info` page in your browser.

---
