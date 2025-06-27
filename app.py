from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('models/umkm_model.pkl')
scaler = joblib.load('models/umkm_scaler.pkl')

# Routes
@app.route('/')
def landing():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Process form data
        features = [
            float(request.form['pendapatan']),
            float(request.form['pengeluaran']),
            float(request.form['hutang']),
            float(request.form['aset']),
            float(request.form['modal_kerja']),
            float(request.form['arus_kas']),
            int(request.form['karyawan']),
            float(request.form['laba']),
        ]
        
        # Make prediction
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        
        # Map prediction to status
        status_map = {
            'sehat': 'Sehat',
            'kurang': 'Kurang Sehat',
            'kritis': 'Kritis'
        }
        
        return render_template('result.html', 
                             prediction=status_map.get(prediction, 'Tidak Diketahui'),
                             features=request.form)
    
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/features')
def features():
    return render_template('features.html')

if __name__ == '__main__':
    app.run(debug=True)