from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

MODELS_DIR = 'backend/models'

def get_prediction(disease, data):
    try:
       
        with open(f'{MODELS_DIR}/{disease}_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        algorithms = ['LogisticRegression', 'SVM', 'RandomForest', 'XGBoost']
        predictions = {}
        
        data_scaled = scaler.transform([data])
        
        for algo in algorithms:
            with open(f'{MODELS_DIR}/{disease}_{algo}.pkl', 'rb') as f:
                model = pickle.load(f)
                
            
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(data_scaled)[0][1]
                pred = int(model.predict(data_scaled)[0])
            else:
                pred = int(model.predict(data_scaled)[0])
                prob = float(pred) # Fallback
                
            predictions[algo] = {
                'prediction': pred,
                'probability': round(float(prob) * 100, 2)
            }
            
        return predictions
    except Exception as e:
        return {'error': str(e)}

@app.route('/predict', methods=['POST'])
def predict():
    req_data = request.get_json()
    disease = req_data.get('disease')
    features = req_data.get('features')
    
    if not disease or not features:
        return jsonify({'error': 'Missing disease or features'}), 400
        
    results = get_prediction(disease, features)
    return jsonify(results)

@app.route('/metadata', methods=['GET'])
def get_metadata():
    diseases = ['breast_cancer', 'diabetes', 'heart_disease']
    metadata = {}
    
    for disease in diseases:
        try:
            with open(f'{MODELS_DIR}/{disease}_features.pkl', 'rb') as f:
                features = pickle.load(f)
            
            # Also get accuracy metrics
            metrics = {}
            if os.path.exists(f'{MODELS_DIR}/{disease}_metrics.pkl'):
                with open(f'{MODELS_DIR}/{disease}_metrics.pkl', 'rb') as f_m:
                    metrics = pickle.load(f_m)
                    
            metadata[disease] = {
                'features': features,
                'metrics': metrics
            }
        except Exception as e:
            print(f"Error loading {disease}: {e}")
            continue
            
    return jsonify(metadata)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
