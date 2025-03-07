from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

MODEL_PATH = os.path.join(os.getcwd(), "VGG16_Model.tflite")

# تحميل موديل TFLite
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# الحصول على تفاصيل الإدخال والإخراج
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")  
    image = image.resize((224, 224))  
    image = np.array(image) / 255.0  
    return np.expand_dims(image, axis=0)  

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400  

    file = request.files['file']
    
    try:
        image = Image.open(file.stream)
        processed_image = preprocess_image(image).astype(np.float32)

        # تنفيذ التنبؤ باستخدام TFLite
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        predicted_class = class_names[np.argmax(predictions[0])]  
        probabilities = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}  

        return jsonify({'predicted_class': predicted_class, 'probabilities': probabilities})
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500  

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, threaded=True)
