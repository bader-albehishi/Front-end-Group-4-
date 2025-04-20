
# 🖼️ Image Classification API with TFLite and Flask

This project is a simple and efficient image classification web API built using **Flask** and a **TensorFlow Lite (TFLite)** model. It allows users to upload an image and receive a predicted class label along with class probabilities based on the **CIFAR-10 dataset**.

## 🚀 Features

- 📱 TFLite model for fast and lightweight inference
- 🌐 Flask backend for serving predictions
- 🔐 CORS enabled for cross-origin requests
- 🔄 API endpoint to receive image input and return classification results
- 📊 Returns both predicted class and probability distribution

## 🧠 Model

The model used is based on **VGG16**, converted into **TensorFlow Lite** format for optimized performance. It classifies images into the following 10 CIFAR-10 categories:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## 🗂️ Project Structure

```
.
├── VGG16_Model.tflite        # TFLite model file
├── app.py                    # Flask backend server
├── index.html                # (Optional) UI frontend
└── README.md                 # Project documentation
```

## 📦 Requirements

Make sure you have Python 3.7+ and install the required packages:

```bash
pip install flask flask-cors tensorflow pillow
```

## 🧪 Run the App

```bash
python app.py
```

By default, the server will run at `http://0.0.0.0:5000`.

## 📤 API Usage

### `POST /predict`

**Form-data key:** `file`  
**Value:** Image file to be classified

**Example using `curl`:**

```bash
curl -X POST -F "file=@cat.jpg" http://localhost:5000/predict
```

**Response:**

```json
{
  "predicted_class": "cat",
  "probabilities": {
    "airplane": 0.01,
    "automobile": 0.02,
    "bird": 0.15,
    "cat": 0.65,
    ...
  }
}
```

## 🔧 Preprocessing Details

- Converts image to RGB if not already
- Resizes to 224x224 (as required by VGG16)
- Normalizes pixel values to range [0, 1]

## 📌 Notes

- The TFLite model must be named `VGG16_Model.tflite` and placed in the same directory as `app.py`.
- You can optionally create a simple frontend using `index.html` for image upload.

## 📄 License

This project is provided for educational purposes. Feel free to enhance and customize it as needed!
