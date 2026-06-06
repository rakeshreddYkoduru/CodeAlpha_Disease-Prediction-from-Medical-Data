import pickle
import numpy as np
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from functools import lru_cache

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

MODELS_DIR = 'backend/models'

class ModelHandler:
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.diseases = ['breast_cancer', 'diabetes', 'heart_disease']
        self.algorithms = ['LogisticRegression', 'SVM', 'RandomForest', 'XGBoost']
        self.cache = {}
        self.metadata = {}
        self.load_all()

    def load_all(self):
        print("🚀 Loading AI models into memory...")
        for disease in self.diseases:
            disease_cache = {'models': {}}
            try:
                # Load scaler
                scaler_path = f'{self.models_dir}/{disease}_scaler.pkl'
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        disease_cache['scaler'] = pickle.load(f)

                # Load features
                feat_path = f'{self.models_dir}/{disease}_features.pkl'
                if os.path.exists(feat_path):
                    with open(feat_path, 'rb') as f:
                        features = pickle.load(f)
                else:
                    features = []

                # Load metrics
                metric_path = f'{self.models_dir}/{disease}_metrics.pkl'
                metrics = {}
                if os.path.exists(metric_path):
                    with open(metric_path, 'rb') as f:
                        metrics = pickle.load(f)

                self.metadata[disease] = {
                    'features': features,
                    'metrics': metrics
                }

                # Load algorithms
                for algo in self.algorithms:
                    algo_path = f'{self.models_dir}/{disease}_{algo}.pkl'
                    if os.path.exists(algo_path):
                        with open(algo_path, 'rb') as f:
                            disease_cache['models'][algo] = pickle.load(f)
                
                self.cache[disease] = disease_cache
                print(f"✅ Loaded models for {disease}")
            except Exception as e:
                print(f"❌ Error loading {disease}: {e}")

    def predict(self, disease, features):
        if disease not in self.cache:
            return {'error': f'Disease type {disease} not found'}
        
        disease_data = self.cache[disease]
        scaler = disease_data.get('scaler')
        models = disease_data.get('models', {})
        
        if not scaler or not models:
            return {'error': 'Incomplete model data'}
        
        try:
            data_scaled = scaler.transform([features])
            results = {}
            
            for algo, model in models.items():
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(data_scaled)[0][1]
                    pred = int(model.predict(data_scaled)[0])
                else:
                    pred = int(model.predict(data_scaled)[0])
                    prob = float(pred)
                
                results[algo] = {
                    'prediction': pred,
                    'probability': round(float(prob) * 100, 2)
                }
            return results
        except Exception as e:
            return {'error': str(e)}

# Initialize Handler
handler = ModelHandler(MODELS_DIR)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    req_data = request.get_json()
    disease = req_data.get('disease')
    features = req_data.get('features')
    
    if not disease or not features:
        return jsonify({'error': 'Missing disease or features'}), 400
        
    results = handler.predict(disease, features)
    return jsonify(results)

@app.route('/metadata', methods=['GET'])
def get_metadata():
    return jsonify(handler.metadata)

if __name__ == '__main__':
    # Use threaded=True for better response handling
    app.run(debug=True, port=4000, threaded=True)

