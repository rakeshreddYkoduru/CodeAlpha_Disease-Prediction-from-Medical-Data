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
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

# Ensure directories exist
os.makedirs('backend/models', exist_ok=True)
os.makedirs('backend/data', exist_ok=True)

def select_top_features(X, y, feature_names, top_n=10):
    """Select top n features based on Random Forest importance."""
    if X.shape[1] <= top_n:
        return X, feature_names
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    selected_features = [feature_names[i] for i in indices]
    X_reduced = X[:, indices]
    
    print(f"Selected {top_n} features: {selected_features}")
    return X_reduced, selected_features

def train_and_save(X, y, disease_name, feature_names, top_n=10):
    print(f"\n--- Training for {disease_name} ---")
    
    # Feature Reduction
    X_selected, selected_feature_names = select_top_features(X, y, feature_names, top_n)
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'SVM': SVC(probability=True),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
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
        pickle.dump(selected_feature_names, f)
        
    print(f"Final results for {disease_name}: {results}")

# 1. Breast Cancer
print("Processing Breast Cancer...")
cancer_data = load_breast_cancer()
# Reduce to top 8 features for a cleaner UI
train_and_save(cancer_data.data, cancer_data.target, 'breast_cancer', cancer_data.feature_names.tolist(), top_n=8)

# 2. Diabetes
print("Processing Diabetes...")
diabetes_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
diabetes_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
diabetes_df = pd.read_csv(diabetes_url, names=diabetes_cols)
# Keep top 6 features
train_and_save(diabetes_df.drop('Outcome', axis=1).values, diabetes_df['Outcome'].values, 'diabetes', diabetes_cols[:-1], top_n=6)

# 3. Heart Disease
print("Processing Heart Disease...")
heart_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
heart_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
heart_df = pd.read_csv(heart_url, names=heart_cols, na_values='?')
heart_df = heart_df.fillna(heart_df.median())
heart_df['target'] = heart_df['target'].apply(lambda x: 1 if x > 0 else 0)
# Keep top 8 features
train_and_save(heart_df.drop('target', axis=1).values, heart_df['target'].values, 'heart_disease', heart_cols[:-1], top_n=8)

print("\nModel training with feature reduction complete!")
