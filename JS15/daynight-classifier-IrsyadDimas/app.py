import os
import numpy as np
import cv2
import pickle
import tensorflow as tf

from flask import Flask, request, render_template
from skimage.feature import hog
import keras

app = Flask(__name__)

# Load Model & Scaler
MODEL_PATH = 'day_night_model.h5'
SCALER_PATH = 'scaler.pkl'

try:
    model = keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("✅ System Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading system: {e}")
    model = None
    scaler = None

def preprocess_image(image_bytes):
    """Preprocess image untuk prediksi"""
    # Decode gambar
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gambar tidak valid atau gagal dibaca.")
    
    # Preprocessing (Harus sama persis dengan Training)
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(8,8),
                   cells_per_block=(2,2), block_norm='L2-Hys',
                   visualize=False, feature_vector=True)
    if scaler is None:
        raise ValueError("Scaler gagal dimuat. Silakan cek file scaler.pkl.")
    return scaler.transform(hog_feat.reshape(1, -1))

@app.route('/', methods=['GET'])
def home():
    """Halaman utama"""
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk prediksi"""
    try:
        if model is None:
            return "Error: Model gagal dimuat. Silakan cek file model."
        
        # Ambil file dari request
        file = request.files['file']
        
        # Preprocess gambar
        data = preprocess_image(file.read())
        
        # Prediksi
        prediction = model.predict(data)[0][0]
        
        # Tentukan label dan confidence
        if prediction > 0.5:
            label = "Day (Siang)"
            confidence = round(prediction * 100, 1)  # Confidence untuk Day
        else:
            label = "Night (Malam)"
            confidence = round((1 - prediction) * 100, 1)  # Confidence untuk Night
        
        # Kirim ke template dengan label dan confidence
        return render_template('result.html', label=label, confidence=confidence)
    
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    # Port 7860 wajib untuk Hugging Face Spaces
    app.run(host='0.0.0.0', port=7860, debug=True)