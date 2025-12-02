

from flask import Flask, render_template, request
import os
import traceback
from ml_logic.model_loader import ModelLoader
from ml_logic.predictor import Predictor

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models_ews')
model_loader = ModelLoader(MODEL_DIR)
predictor = None
try:
    model_loader.load_models()
    predictor = Predictor(model_loader)
except Exception as e:
    print(f"❌ Error loading models: {e}")
    traceback.print_exc()

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return render_template('result.html', error='Model belum siap. Hubungi admin.')
    try:
        tanggal = request.form['tanggal']
        nominal = int(request.form['nominal'])
        target_type = request.form['target_type']
        rt_number = request.form.get('rt_number') or None
        result = predictor.predict(tanggal, nominal, target_type, rt_number)
        # Tambahkan input ke result agar bisa ditampilkan di template
        result['tanggal'] = tanggal
        result['nominal'] = nominal
        result['target_type'] = target_type
        result['rt_number'] = rt_number
        return render_template('result.html', result=result)
    except Exception as e:
        return render_template('result.html', error=f'Error: {e}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
