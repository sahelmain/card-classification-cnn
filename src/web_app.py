from flask import Flask, render_template, request, jsonify
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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for models
models = {}
model_info = {}

# Card class names (53 classes)
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
        'Custom CNN': '../results/custom_cnn.h5',
        'Lightweight CNN': '../results/lightweight_cnn.h5',
        'MobileNetV2': '../results/mobilenet_v2.h5'
    }
    
    for model_name, path in model_paths.items():
        try:
            if os.path.exists(path):
                models[model_name] = tf.keras.models.load_model(path)
                model_info[model_name] = {
                    'accuracy': get_model_accuracy(model_name),
                    'loaded_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'Ready'
                }
                logger.info(f"Loaded model: {model_name}")
            else:
                logger.warning(f"Model file not found: {path}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            model_info[model_name] = {
                'status': 'Error',
                'error': str(e)
            }

def get_model_accuracy(model_name):
    """Get pre-computed model accuracy"""
    accuracies = {
        'Custom CNN': 0.814,
        'Lightweight CNN': 0.740,
        'MobileNetV2': 0.517
    }
    return accuracies.get(model_name, 0.0)

def preprocess_image(image):
    """Preprocess uploaded image for model prediction"""
    try:
        # Resize to model input size
        image = image.resize((200, 200))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        return None

def predict_card(image, model_name):
    """Predict card class using specified model"""
    try:
        if model_name not in models:
            return None, f"Model {model_name} not available"
        
        # Preprocess image
        processed_image = preprocess_image(image)
        if processed_image is None:
            return None, "Image preprocessing failed"
        
        # Make prediction
        model = models[model_name]
        predictions = model.predict(processed_image, verbose=0)
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            confidence = float(predictions[0][idx])
            card_name = CARD_CLASSES[idx] if idx < len(CARD_CLASSES) else f"Class_{idx}"
            
            # Format card name for display
            formatted_name = card_name.replace('_', ' ').title()
            
            top_predictions.append({
                'card': formatted_name,
                'confidence': confidence,
                'percentage': confidence * 100
            })
        
        return top_predictions, None
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None, str(e)

@app.route('/')
def index():
    """Main page with model selection and upload interface"""
    return render_template('index.html', 
                         models=list(models.keys()),
                         model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get selected model
        model_name = request.form.get('model', 'Custom CNN')
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not available'}), 400
        
        # Process image
        image = Image.open(file.stream)
        predictions, error = predict_card(image, model_name)
        
        if error:
            return jsonify({'error': error}), 500
        
        # Convert image to base64 for display
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            'predictions': predictions,
            'model_used': model_name,
            'model_accuracy': model_info[model_name].get('accuracy', 0.0),
            'image_data': img_str,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'timestamp': datetime.now().isoformat()
    })

def create_templates():
    """Create minimal HTML template for the Flask app"""
    templates_dir = 'templates'
    os.makedirs(templates_dir, exist_ok=True)
    
    # Simple responsive template
    index_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Card Classification CNN</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            .card-hover:hover { transform: translateY(-5px); transition: all 0.3s; }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark gradient-bg">
            <div class="container">
                <span class="navbar-brand">🎴 Card Classification CNN</span>
            </div>
        </nav>
        
        <div class="container my-4">
            <div class="row">
                <div class="col-lg-8">
                    <div class="card shadow-sm card-hover">
                        <div class="card-body">
                            <h2>🎯 Card Prediction</h2>
                            <p class="text-muted">Upload a playing card image for classification.</p>
                            
                            <form id="uploadForm" enctype="multipart/form-data">
                                <div class="mb-3">
                                    <label for="modelSelect" class="form-label">Select Model</label>
                                    <select class="form-select" id="modelSelect" name="model">
                                        {% for model in models %}
                                        <option value="{{ model }}">{{ model }}</option>
                                        {% endfor %}
                                    </select>
                                </div>
                                
                                <div class="mb-3">
                                    <label for="imageInput" class="form-label">Upload Card Image</label>
                                    <input type="file" class="form-control" id="imageInput" name="image" 
                                           accept="image/*" required>
                                </div>
                                
                                <button type="submit" class="btn btn-primary">Predict Card</button>
                            </form>
                            
                            <div id="loadingDiv" class="d-none text-center my-4">
                                <div class="spinner-border text-primary"></div>
                                <p class="mt-2">Analyzing...</p>
                            </div>
                        </div>
                    </div>
                    
                    <div id="resultsCard" class="card shadow-sm mt-4 d-none">
                        <div class="card-body">
                            <h3>🎊 Results</h3>
                            <div id="resultsContent"></div>
                        </div>
                    </div>
                </div>
                
                <div class="col-lg-4">
                    <div class="card shadow-sm">
                        <div class="card-body">
                            <h3>📊 Models</h3>
                            {% for model, info in model_info.items() %}
                            <div class="mb-3 p-3 border rounded">
                                <h5>{{ model }}</h5>
                                {% if info.status == 'Ready' %}
                                <span class="badge bg-success">Ready</span>
                                <p class="mb-1"><strong>Accuracy:</strong> {{ "%.1f"|format(info.accuracy * 100) }}%</p>
                                {% else %}
                                <span class="badge bg-danger">{{ info.status }}</span>
                                {% endif %}
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            const imageFile = document.getElementById('imageInput').files[0];
            const model = document.getElementById('modelSelect').value;
            
            if (!imageFile) {
                alert('Please select an image first.');
                return;
            }
            
            formData.append('image', imageFile);
            formData.append('model', model);
            
            document.getElementById('loadingDiv').classList.remove('d-none');
            document.getElementById('resultsCard').classList.add('d-none');
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    displayResults(result);
                } else {
                    throw new Error(result.error || 'Prediction failed');
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('loadingDiv').classList.add('d-none');
            }
        });
        
        function displayResults(result) {
            let html = `
                <div class="row">
                    <div class="col-md-6">
                        <img src="data:image/png;base64,${result.image_data}" 
                             class="img-fluid rounded border" alt="Uploaded image">
                    </div>
                    <div class="col-md-6">
                        <h5>Top Predictions:</h5>
            `;
            
            result.predictions.forEach((pred, index) => {
                const badgeClass = index === 0 ? 'bg-success' : 'bg-secondary';
                html += `
                    <div class="mb-2 p-2 border rounded">
                        <div class="d-flex justify-content-between">
                            <span><strong>${pred.card}</strong></span>
                            <span class="badge ${badgeClass}">${pred.percentage.toFixed(1)}%</span>
                        </div>
                    </div>
                `;
            });
            
            html += `
                    </div>
                </div>
                <hr>
                <div class="text-center">
                    <strong>Model:</strong> ${result.model_used} | 
                    <strong>Accuracy:</strong> ${(result.model_accuracy * 100).toFixed(1)}%
                </div>
            `;
            
            document.getElementById('resultsContent').innerHTML = html;
            document.getElementById('resultsCard').classList.remove('d-none');
        }
        </script>
    </body>
    </html>
    """
    
    with open(f'{templates_dir}/index.html', 'w') as f:
        f.write(index_template)

if __name__ == '__main__':
    # Create templates
    create_templates()
    
    # Load models
    logger.info("Loading models...")
    load_models()
    
    # Run the app
    logger.info("Starting Flask web application on http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=True) 