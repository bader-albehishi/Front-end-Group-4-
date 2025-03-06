import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

# Setting Up the Flask App
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Enable CORS for all routes

# Loading the model
model = tf.keras.models.load_model("VGG16_Model.h5")
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
        return jsonify({'error': 'No file uploaded'}), 400  # Handle missing file

    file = request.files['file']
    try:
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class = class_names[np.argmax(predictions[0])]  # Get the highest probability class
        probabilities = {class_names[i]: float(predictions[0][i]) for i in range(10)}  # Convert to JSON-serializable format
        return jsonify({'predicted_class': predicted_class, 'probabilities': probabilities})
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Handle errors during processing

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)