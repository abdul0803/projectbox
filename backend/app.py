from flask import Flask, request, jsonify
import os
import requests
import cv2
from google.cloud import vision
import io
import numpy as np

app = Flask(__name__)

# Azure Custom Vision API credentials
PREDICTION_URL = "https://customevisionproj-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/861b01aa-d77c-4118-8ca1-617c27a75487/detect/iterations/Iteration6/image"
PREDICTION_KEY = "7LJBYddFtLFqnpnj8tBA41MZeSv0trInQGTSP1874tvFpX6zKP9IJQQJ99AKACYeBjFXJ3w3AAAIACOG5wcU"

# Google Vision OCR setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/abdua/OneDrive/projects/projectbox/backend/vision-ocr-project-448520-47a94d8a53c2.json"
vision_client = vision.ImageAnnotatorClient()


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file locally
    upload_folder = './temp_uploads'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # Send the image to Azure Custom Vision for detection
    with open(file_path, 'rb') as image_file:
        headers = {
            "Prediction-Key": PREDICTION_KEY,
            "Content-Type": "application/octet-stream"
        }
        response = requests.post(PREDICTION_URL, headers=headers, data=image_file)

    if response.status_code != 200:
        return jsonify({"error": "Azure Custom Vision API call failed"}), response.status_code

    # Parse the predictions
    predictions = response.json().get('predictions', [])
    image = cv2.imread(file_path)

    # Ensure debug directories exist
    debug_crops_folder = './debug_crops'
    debug_processed_folder = './debug_preprocessed'
    os.makedirs(debug_crops_folder, exist_ok=True)
    os.makedirs(debug_processed_folder, exist_ok=True)

    ocr_results = []

    # Separate handwriting and noise bounding boxes
    handwriting_bboxes = []
    noise_bboxes = []
    for prediction in predictions:
        bbox = prediction['boundingBox']
        if prediction['tagName'] == 'hand written' and prediction['probability'] > 0.8:
            handwriting_bboxes.append(bbox)
        elif prediction['tagName'] == 'noise' and prediction['probability'] > 0.8:
            noise_bboxes.append(bbox)

    for bbox in handwriting_bboxes:
        # Convert bounding box to absolute coordinates
        original_x, original_y, original_w, original_h = calculate_bounding_box(bbox, image.shape)

        # Adjust the bounding box if it overlaps with noise
        adjusted_x, adjusted_y, adjusted_w, adjusted_h = adjust_bounding_box(
            (original_x, original_y, original_x + original_w, original_y + original_h),
            [
                (int(n['left'] * image.shape[1]), int(n['top'] * image.shape[0]),
                int((n['left'] + n['width']) * image.shape[1]),
                int((n['top'] + n['height']) * image.shape[0]))
                for n in noise_bboxes
            ]
        )

        # Crop the region from the image
        cropped_region = image[adjusted_y:adjusted_h, adjusted_x:adjusted_w]

        # Save the cropped region
        crop_path = os.path.join(debug_crops_folder, f'crop_{adjusted_x}_{adjusted_y}.jpg')
        cv2.imwrite(crop_path, cropped_region)

        # Preprocess image for red handwriting segmentation
        segmented_image = segment_red_handwriting(crop_path)
        enhanced_path = os.path.join(debug_processed_folder, f'segmented_{adjusted_x}_{adjusted_y}.jpg')
        cv2.imwrite(enhanced_path, segmented_image)

        # Perform OCR using Google Vision
        ocr_text = perform_ocr_google(enhanced_path)
        ocr_results.append({
            "bounding_box": bbox,
            "ocr_text": ocr_text
        })

    return jsonify({
        "file": file.filename,
        "ocr_results": ocr_results
    }), 200


# Helper function to calculate bounding box
def calculate_bounding_box(bbox, image_shape, padding_x_factor=0.1, padding_y_factor=0.1):
    height, width = image_shape[:2]
    x = int(bbox['left'] * width)
    y = int(bbox['top'] * height)
    w = int(bbox['width'] * width)
    h = int(bbox['height'] * height)

    # Add padding
    padding_x = int(padding_x_factor * w)
    padding_y = int(padding_y_factor * h)

    x = max(0, x - padding_x)
    y = max(0, y - padding_y)
    w = min(width - x, w + 2 * padding_x)
    h = min(height - y, h + 2 * padding_y)

    return x, y, w, h


# Helper function to adjust bounding box based on noise
def adjust_bounding_box(handwriting_bbox, noise_bboxes, overlap_threshold=0.00):
    x1_hand, y1_hand, x2_hand, y2_hand = handwriting_bbox
    handwriting_area = (x2_hand - x1_hand) * (y2_hand - y1_hand)

    for noise_bbox in noise_bboxes:
        x1_noise, y1_noise, x2_noise, y2_noise = noise_bbox

        # Calculate overlap area
        x_overlap = max(0, min(x2_hand, x2_noise) - max(x1_hand, x1_noise))
        y_overlap = max(0, min(y2_hand, y2_noise) - max(y1_hand, y1_noise))
        overlap_area = x_overlap * y_overlap
        overlap_ratio = overlap_area / handwriting_area

        if overlap_area > overlap_threshold * handwriting_area:
            # Adjust bounding box
            if y1_noise < y2_hand and y2_noise > y1_hand:  # Vertical overlap
                if abs(y1_hand - y1_noise) < abs(y2_hand - y2_noise):  # Overlap at top
                    y1_hand = max(y1_hand, y2_noise)
                else:  # Overlap at bottom
                    y2_hand = min(y2_hand, y1_noise)

            if x1_noise < x2_hand and x2_noise > x1_hand:  # Horizontal overlap
                if abs(x1_hand - x1_noise) < abs(x2_hand - x2_noise):  # Overlap at left
                    x1_hand = max(x1_hand, x2_noise)
                else:  # Overlap at right
                    x2_hand = min(x2_hand, x1_noise)

    return x1_hand, y1_hand, x2_hand, y2_hand


# Function to segment red handwriting
def segment_red_handwriting(image_path):
    # Load the image
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for red
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create red mask
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = red_mask1 | red_mask2

    # Morphological operations to clean the mask
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Apply mask to the original image
    result = cv2.bitwise_and(image, image, mask=cleaned_mask)

    # Convert to grayscale
    grayscale_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    return grayscale_result
# Helper function to perform OCR using Google Vision
def perform_ocr_google(image_path):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = vision_client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f"Google Vision OCR Error: {response.error.message}")

    if texts:
        return texts[0].description.strip()  # Return the full detected text
    return "No text detected"


if __name__ == "__main__":
    app.run(debug=True)
