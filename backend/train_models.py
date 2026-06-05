import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_breast_cancer

# Ensure directories exist
os.makedirs('backend/models', exist_ok=True)
os.makedirs('backend/data', exist_ok=True)

def train_and_save(X, y, disease_name, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'SVM': SVC(probability=True),
        'RandomForest': RandomForestClassifier(n_estimators=100),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results[name] = round(acc * 100, 2)
        
        # Save model
        with open(f'backend/models/{disease_name}_{name}.pkl', 'wb') as f:
            pickle.dump(model, f)
            
    # Save results/accuracy
    with open(f'backend/models/{disease_name}_metrics.pkl', 'wb') as f:
        pickle.dump(results, f)
        
    # Save scaler and feature names
    with open(f'backend/models/{disease_name}_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(f'backend/models/{disease_name}_features.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
        
    print(f"Results for {disease_name}: {results}")

# 1. Breast Cancer (using sklearn)
print("Training Breast Cancer models...")
cancer_data = load_breast_cancer()
train_and_save(cancer_data.data, cancer_data.target, 'breast_cancer', cancer_data.feature_names.tolist())

# 2. Diabetes (using URL)
print("Training Diabetes models...")
diabetes_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
diabetes_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
diabetes_df = pd.read_csv(diabetes_url, names=diabetes_cols)
train_and_save(diabetes_df.drop('Outcome', axis=1), diabetes_df['Outcome'], 'diabetes', diabetes_cols[:-1])

# 3. Heart Disease (using URL)
print("Training Heart Disease models...")
heart_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
heart_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
heart_df = pd.read_csv(heart_url, names=heart_cols, na_values='?')
heart_df = heart_df.fillna(heart_df.median()) # Simple imputation
heart_df['target'] = heart_df['target'].apply(lambda x: 1 if x > 0 else 0) # Binary classification
train_and_save(heart_df.drop('target', axis=1), heart_df['target'], 'heart_disease', heart_cols[:-1])

print("All models trained and saved successfully!")
