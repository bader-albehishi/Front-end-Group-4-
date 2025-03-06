import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import logging

# Setting Up the Flask App
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)

# Loading the model
try:
    model = tf.keras.models.load_model("VGG16_Model.h5")
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Image Preprocessing 
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")  # Ensure image is in RGB mode
    image = image.resize((224, 224))  # Resize image to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Serving an HTML File
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Defining the Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        logging.error("No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        logging.error("Empty file uploaded")
        return jsonify({'error': 'Empty file uploaded'}), 400

    try:
        logging.info(f"Processing file: {file.filename}")
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class = class_names[np.argmax(predictions[0])]  # Get the highest probability class
        probabilities = {class_names[i]: float(predictions[0][i]) for i in range(10)}  # Convert to JSON-serializable format
        logging.info(f"Prediction successful: {predicted_class}")
        return jsonify({'predicted_class': predicted_class, 'probabilities': probabilities})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)