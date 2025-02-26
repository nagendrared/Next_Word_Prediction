# app.py - Flask application for Next Word Prediction

from flask import Flask, request, jsonify, render_template
from next_word_predictor import NextWordPredictor
import time
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize the predictor
predictor = None

def get_predictor():
    """Lazy initialization of the predictor to avoid loading at import time"""
    global predictor
    if predictor is None:
        # Initialize with distilgpt2 (faster) or gpt2 (more accurate)
        model_name = os.environ.get('MODEL_NAME', 'distilgpt2')
        predictor = NextWordPredictor(model_name=model_name)
    return predictor

@app.route('/')
def home():
    """Render the home page with the prediction interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for next word prediction"""
    # Get data from request
    data = request.get_json()
    
    # Extract parameters with defaults
    text = data.get('text', '').strip()
    num_words = min(max(int(data.get('num_words', 1)), 1), 5)  # Limit between 1-5
    num_predictions = min(max(int(data.get('num_predictions', 5)), 1), 10)  # Limit between 1-10
    temperature = float(data.get('temperature', 1.0))
    
    # Validate input
    if not text:
        return jsonify({'error': 'Text input is required'}), 400
    
    # Log the request 
    print(f"Prediction request: '{text}' (words: {num_words}, predictions: {num_predictions})")
    
    # Time the prediction
    start_time = time.time()
    
    try:
        # Get the predictor
        predictor = get_predictor()
        
        # Get predictions
        predictions = predictor.predict_next_words(
            text, 
            num_words=num_words,
            num_predictions=num_predictions,
            temperature=temperature
        )
        
        # Format predictions for JSON response
        formatted_predictions = []
        for prediction, confidence in predictions:
            # Skip empty predictions or those that are just spaces
            if not prediction or prediction.isspace():
                continue
                
            formatted_predictions.append({
                'text': prediction,
                'confidence': confidence
            })
        
        # If we have fewer predictions than requested, give what we have
        if not formatted_predictions:
            return jsonify({
                'error': 'Could not generate valid predictions. Try a different input text.'
            }), 400
        
        elapsed_time = time.time() - start_time
        
        # Create response
        response = {
            'original_text': text,
            'predictions': formatted_predictions,
            'processing_time': elapsed_time
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': predictor is not None
    })

@app.route('/info')
def info():
    """Information about the app"""
    model_name = os.environ.get('MODEL_NAME', 'distilgpt2')
    return jsonify({
        'name': 'Next Word Prediction API',
        'model': model_name,
        'device': get_predictor().device if predictor else None
    })

if __name__ == '__main__':
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    
    # Preload the model if AUTO_LOAD is set to true
    if os.environ.get('AUTO_LOAD', 'false').lower() == 'true':
        get_predictor()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=port)