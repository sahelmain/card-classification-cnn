from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables
models = {}
model_info = {}

# Card class names
CARD_CLASSES = [
    'ace_of_clubs', 'ace_of_diamonds', 'ace_of_hearts', 'ace_of_spades',
    'two_of_clubs', 'two_of_diamonds', 'two_of_hearts', 'two_of_spades',
    'three_of_clubs', 'three_of_diamonds', 'three_of_hearts', 'three_of_spades',
    'four_of_clubs', 'four_of_diamonds', 'four_of_hearts', 'four_of_spades',
    'five_of_clubs', 'five_of_diamonds', 'five_of_hearts', 'five_of_spades',
    'six_of_clubs', 'six_of_diamonds', 'six_of_hearts', 'six_of_spades',
    'seven_of_clubs', 'seven_of_diamonds', 'seven_of_hearts', 'seven_of_spades',
    'eight_of_clubs', 'eight_of_diamonds', 'eight_of_hearts', 'eight_of_spades',
    'nine_of_clubs', 'nine_of_diamonds', 'nine_of_hearts', 'nine_of_spades',
    'ten_of_clubs', 'ten_of_diamonds', 'ten_of_hearts', 'ten_of_spades',
    'jack_of_clubs', 'jack_of_diamonds', 'jack_of_hearts', 'jack_of_spades',
    'queen_of_clubs', 'queen_of_diamonds', 'queen_of_hearts', 'queen_of_spades',
    'king_of_clubs', 'king_of_diamonds', 'king_of_hearts', 'king_of_spades',
    'joker'
]

def load_models():
    """Load all available trained models"""
    global models, model_info
    
    model_paths = {
        'custom_cnn': '../results/custom_cnn.h5',
        'lightweight_cnn': '../results/lightweight_cnn.h5',
        'mobilenet_v2': '../results/mobilenet_v2.h5'
    }
    
    for model_name, path in model_paths.items():
        try:
            if os.path.exists(path):
                models[model_name] = tf.keras.models.load_model(path)
                model_info[model_name] = {
                    'name': model_name,
                    'accuracy': get_model_accuracy(model_name),
                    'loaded_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'ready'
                }
                logger.info(f"Loaded model: {model_name}")
            else:
                logger.warning(f"Model file not found: {path}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")

def get_model_accuracy(model_name):
    """Get pre-computed model accuracy"""
    accuracies = {
        'custom_cnn': 0.814,
        'lightweight_cnn': 0.740,
        'mobilenet_v2': 0.517
    }
    return accuracies.get(model_name, 0.0)

def preprocess_image(image_data):
    """Preprocess base64 image data for model prediction"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Resize and normalize
        image = image.resize((200, 200))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        return None

@app.route('/')
def index():
    """API root endpoint with documentation"""
    return {
        'message': 'Card Classification CNN API',
        'version': '1.0',
        'endpoints': {
            'predict': '/api/v1/predict',
            'models': '/api/v1/models',
            'health': '/api/v1/health'
        },
        'timestamp': datetime.now().isoformat()
    }

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    """
    Predict playing card from uploaded image
    
    Expects JSON: {"image": "base64_encoded_image", "model_name": "custom_cnn"}
    Returns: {"predicted_class": "...", "confidence": 0.XX, "top_predictions": [...]}
    """
    try:
        start_time = datetime.now()
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided', 'timestamp': datetime.now().isoformat()}), 400
        
        model_name = data.get('model_name', 'custom_cnn')
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not available', 'timestamp': datetime.now().isoformat()}), 400
        
        # Preprocess image
        image_array = preprocess_image(data['image'])
        if image_array is None:
            return jsonify({'error': 'Invalid image data', 'timestamp': datetime.now().isoformat()}), 400
        
        # Make prediction
        model = models[model_name]
        predictions = model.predict(image_array, verbose=0)
        
        # Get results
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get top 5 predictions
        top_indices = np.argsort(predictions[0])[-5:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            card_name = CARD_CLASSES[idx] if idx < len(CARD_CLASSES) else f"Class_{idx}"
            formatted_name = card_name.replace('_', ' ').title()
            top_predictions.append({
                'class': formatted_name,
                'confidence': float(predictions[0][idx]),
                'percentage': float(predictions[0][idx] * 100)
            })
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        predicted_card = CARD_CLASSES[predicted_class_idx].replace('_', ' ').title()
        
        return jsonify({
            'predicted_class': predicted_card,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'model_used': model_name,
            'processing_time_ms': processing_time,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error', 'timestamp': datetime.now().isoformat()}), 500

@app.route('/api/v1/models', methods=['GET'])
def get_models():
    """Get information about all available models"""
    return jsonify({
        'models': list(model_info.values()),
        'total_models': len(model_info),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Check API health status"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len([m for m in model_info.values() if m.get('status') == 'ready']),
        'api_version': '1.0',
        'tensorflow_version': tf.__version__,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/v1/metrics', methods=['GET'])
def metrics():
    """Get detailed API metrics"""
    return jsonify({
        'total_models': len(model_info),
        'ready_models': len([m for m in model_info.values() if m.get('status') == 'ready']),
        'supported_classes': len(CARD_CLASSES),
        'api_version': '1.0',
        'timestamp': datetime.now().isoformat()
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found', 'timestamp': datetime.now().isoformat()}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'timestamp': datetime.now().isoformat()}), 500

if __name__ == '__main__':
    # Load models
    logger.info("Loading models...")
    load_models()
    
    logger.info("Starting Card Classification API server...")
    logger.info("API endpoints available at: http://localhost:5000/api/v1/")
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True) 