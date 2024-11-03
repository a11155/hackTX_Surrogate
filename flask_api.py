# api_service.py

from flask import Flask, request, jsonify
from waitress import serve
import logging
import time
from typing import Dict
import numpy as np

# Import your existing model code
from reservoir_predictor import ReservoirPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model instance
model = None

def load_model():
    """Load the model on startup"""
    global model
    try:
        logger.info("Starting model loading...")
        start_time = time.time()
        model = ReservoirPredictor("oil/models/reservoir_model.pth")
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

def validate_input(data: Dict):
    """Basic input validation"""
    required_fields = {
        "days_on_production": (0, 365),
        "depth": (0, 20000),
        "permeability": (0, 10000),
        "porosity": (0, 1),
        "initial_pressure": (0, 10000),
        "temperature": (0, 500),
        "thickness": (0, 1000),
        "initial_water_saturation": (0, 1),
        "water_cut": (0, 1),
        "flowing_pressure": (0, 10000)
    }
    
    for field, (min_val, max_val) in required_fields.items():
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
        if not min_val <= data[field] <= max_val:
            raise ValueError(f"{field} must be between {min_val} and {max_val}")

@app.route('/')
def root():
    return jsonify({
        "message": "Reservoir Flow Rate Prediction API",
        "status": "active",
        "model_loaded": model is not None
    })

@app.route('/health')
def health_check():
    if model is None:
        return jsonify({"status": "unhealthy", "detail": "Model not loaded"}), 503
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route('/predict', methods=['POST'])
def predict():
    """Make a single prediction for reservoir flow rate"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        input_data = request.get_json()
        validate_input(input_data)
        
        start_time = time.time()
        prediction = model.predict(input_data)
        inference_time = time.time() - start_time
        
        logger.debug(f"Inference completed in {inference_time:.4f} seconds")
        return jsonify(prediction)
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def start_server(host="0.0.0.0", port=8000):
    """Start the server using waitress"""
    logger.info(f"Starting API server on {host}:{port}")
    logger.info("Loading model...")
    load_model()
    logger.info("Starting server...")
    serve(app, host=host, port=port, threads=4)

if __name__ == "__main__":
    start_server()