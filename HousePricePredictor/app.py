import pickle
import numpy as np
import flask
from flask import Flask, request, jsonify, send_from_directory
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    mean_absolute_percentage_error
)
import os

app = Flask(__name__, static_folder='.')

# Load the pickled model and preprocessing components
with open('regmodel.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

# Load the trained model
model = model_pipeline['model']

# Preprocessing functions (should match the original preprocessing)
def preprocess_input(input_data):
    # Convert input to numpy array
    X = np.array(list(input_data.values())).reshape(1, -1)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Polynomial Features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)
    
    # Feature Selection
    selector = SelectKBest(f_regression, k=10)
    # Note: In a real-world scenario, you'd need to save the selector 
    # or recreate it exactly as in the original preprocessing
    X_selected = selector.fit_transform(X_poly, np.array([0]))
    
    return X_selected

@app.route('/')
def serve_app():
    return send_from_directory('.', 'california-housing-app.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        input_data = request.json
        
        # Preprocess input
        X_processed = preprocess_input(input_data)
        
        # Make prediction
        predicted_price = model.predict(X_processed)[0]
        
        # Optional: Calculate some performance metrics
        # Note: These are example metrics, not actual test set metrics
        metrics = {
            'MSE': mean_squared_error([predicted_price], [predicted_price]),
            'RMSE': np.sqrt(mean_squared_error([predicted_price], [predicted_price])),
            'MAE': mean_absolute_error([predicted_price], [predicted_price]),
            'MAPE': mean_absolute_percentage_error([predicted_price], [predicted_price])
        }
        
        return jsonify({
            'predicted_price': float(predicted_price)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)