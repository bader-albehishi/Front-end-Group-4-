from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# إنشاء تطبيق Flask
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # السماح بتبادل البيانات بين السيرفر والـ frontend

# تحميل النموذج
MODEL_PATH = "VGG16_Model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please check the path.")

model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# دالة معالجة الصورة
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")  
    image = image.resize((224, 224))  
    image = np.array(image) / 255.0  
    return np.expand_dims(image, axis=0)  

# صفحة HTML رئيسية
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# مسار التنبؤ
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400  

    file = request.files['file']
    
    try:
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class = class_names[np.argmax(predictions[0])]  
        probabilities = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}  

        return jsonify({'predicted_class': predicted_class, 'probabilities': probabilities})
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500  

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  
