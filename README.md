# Disease Prediction System (HealthPulse AI)

A modern, machine learning-powered application to predict the risk of Heart Disease, Diabetes, and Breast Cancer.

## Features
- **Multi-Algorithm Analysis**: Compare results from Logistic Regression, SVM, Random Forest, and XGBoost.
- **Dynamic Risk Assessment**: Real-time probability calculations based on patient medical data.
- **Interactive Visualization**: Risk levels visualized using Plotly.js.
- **Premium UI**: Glassmorphic design with a dark mode aesthetic and smooth animations.

## Setup & Usage

### 1. Prerequisites
- Python 3.x
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `flask`, `flask-cors`

### 2. Running the System
The system consists of two parts:
- **Backend (Flask)**: Currently running on `http://localhost:5000`
- **Frontend (Static Server)**: Currently running on `http://localhost:8000`

To start them manually:
```bash
# Start Backend
python backend/app.py

# Start Frontend (in a new terminal)
python -m http.server 8000 --directory frontend
```

### 3. Training Models
If you want to re-train the models with fresh data:
```bash
python backend/train_models.py
```

## Algorithms Used
- **Support Vector Machine (SVM)**: Excellent for high-dimensional medical data.
- **Logistic Regression**: Reliable baseline for classification.
- **Random Forest**: Robust ensemble method to prevent overfitting.
- **XGBoost**: State-of-the-art gradient boosting for maximum accuracy.
