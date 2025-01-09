from flask import Flask, request, jsonify
import os
import requests
import pytesseract
import cv2

app = Flask(__name__)

PREDICTION_URL = ""
PREDICTION_KEY = ""

@app.route('/upload', methods=['POST'])
def upload_file():
    # Validate incoming file
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Save uploaded file
    upload_folder = './temp_uploads'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # Send the file to Azure Custom Vision
    with open(file_path, 'rb') as image_file:
        headers = {
            "Prediction-Key": PREDICTION_KEY,
            "Content-Type": "application/octet-stream"
        }
        response = requests.post(PREDICTION_URL, headers=headers, data=image_file)

    # Handle response from Azure Custom Vision
    if response.status_code != 200:
        try:
            error_details = response.json()
        except ValueError:
            error_details = {"error": "Non-JSON response from Azure"}
        return jsonify({"error": "Failed to get prediction", "details": error_details}), response.status_code

    predictions = response.json().get('predictions', [])
    hand_written_predictions = []

    # Read the uploaded image for cropping
    image = cv2.imread(file_path)

    for prediction in predictions:
        if prediction['tagName'] == 'hand written' and prediction['probability'] > 0.8:
            # Extract bounding box coordinates
            bbox = prediction['boundingBox']
            x = int(bbox['left'] * image.shape[1])
            y = int(bbox['top'] * image.shape[0])
            w = int(bbox['width'] * image.shape[1])
            h = int(bbox['height'] * image.shape[0])

            # Crop the region
            cropped_region = image[y:y+h, x:x+w]

            # Perform OCR on the cropped region
            text = pytesseract.image_to_string(cropped_region, config='--psm 6')
            hand_written_predictions.append({
                "bounding_box": bbox,
                "text": text.strip()  # Clean up extra whitespace
            })

    # Return extracted text or an error message if no confident predictions
    if not hand_written_predictions:
        return jsonify({
            "file": file.filename,
            "message": "No confident hand written text detected.",
            "extracted_text": []
        }), 200

    return jsonify({
        "file": file.filename,
        "extracted_text": hand_written_predictions
    }), 200

if __name__ == "__main__":
    app.run(debug=True)
