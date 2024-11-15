import cv2
import pytesseract
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image
import time
import os

# Azure subscription credentials
subscription_key="5rOi74swZtG1wwdfL73YqusynDLWlxFMAqpwMvMA3WXCWE3Samg8JQQJ99AKACYeBjFXJ3w3AAAFACOGSRqa"
endpoint="https://boxedvision.cognitiveservices.azure.com/"

client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# Load the original image
image_path = "data/testpic.jpg"
image = cv2.imread(image_path)

# Step 1: Crop the Image (focus on the top half)
height, width, _ = image.shape
cropped_image = image[:height // 2, 0:width]

# Step 2: Convert to Grayscale
grayscale_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

# Step 3: Increase Contrast
alpha = 3.0  # Higher contrast control to make handwritten text more distinct
beta = 0  # Brightness control
contrast_image = cv2.convertScaleAbs(grayscale_image, alpha=alpha, beta=beta)

# Step 4: Apply Adaptive Thresholding
threshold_image = cv2.adaptiveThreshold(
    contrast_image, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2
)

# Step 5: Denoise the Image
noise_removed_image = cv2.medianBlur(threshold_image, 5)

# Step 6: Apply Morphological Operations (Dilation)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated_image = cv2.dilate(noise_removed_image, kernel, iterations=1)

# Save the final preprocessed image
preprocessed_image_path = "data/dilated_image.jpg"
cv2.imwrite(preprocessed_image_path, dilated_image)

# Step 7: Send the Preprocessed Image for OCR
with open(preprocessed_image_path, "rb") as image:
    ocr_result = client.read_in_stream(image, raw=True)
    operation_location = ocr_result.headers["Operation-Location"]
    operation_id = operation_location.split("/")[-1]

# Wait for the read operation to complete
while True:
    result = client.get_read_result(operation_id)
    if result.status not in ['notStarted', 'running']:
        break
    time.sleep(1)

# Extract text if the read operation was successful
if result.status == OperationStatusCodes.succeeded:
    for page in result.analyze_result.read_results:
        for line in page.lines:
            print(line.text)

# Optional: Use Tesseract OCR for Handwriting
handwritten_text = pytesseract.image_to_string(Image.open(preprocessed_image_path))
print("Tesseract OCR Result:")
print(handwritten_text)
