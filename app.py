"""
============================================
FIRE & SMOKE DETECTION AI SERVICE
Deploy to: Render.com
============================================
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import base64
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load model at startup
MODEL_PATH = 'fire_smoke_detection_model'
CLASS_NAMES = ['Fire', 'Smoke', 'Normal']
IMG_SIZE = (224, 224)  # Adjust based on your model

try:
    model = load_model(MODEL_PATH)
    logger.info("✓ Model loaded successfully!")
except Exception as e:
    logger.error(f"✗ Model load failed: {e}")
    model = None

# ===== HEALTH CHECK ENDPOINTS =====

@app.route('/', methods=['GET'])
def home():
    """Homepage with API info"""
    return jsonify({
        "status": "online",
        "service": "Fire & Smoke Detection AI",
        "version": "1.0",
        "endpoints": {
            "/": "GET - Service info",
            "/health": "GET - Health check",
            "/predict": "POST - Image prediction"
        },
        "model_loaded": model is not None,
        "classes": CLASS_NAMES
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check for monitoring"""
    return jsonify({
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }), 200 if model is not None else 503

# ===== PREDICTION ENDPOINT =====

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict fire/smoke from base64 image
    
    Request JSON:
    {
        "image": "base64_encoded_image_string"
    }
    
    Response JSON:
    {
        "prediction": {
            "class": "Fire|Smoke|Normal",
            "confidence": 95.5
        },
        "class": "Fire",
        "confidence": 95.5
    }
    """
    try:
        # Validate request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "error": "No image provided",
                "message": "Request must include 'image' field with base64 string"
            }), 400
        
        logger.info("Received prediction request")
        
        # Check if model loaded
        if model is None:
            return jsonify({
                "error": "Model not loaded",
                "message": "AI model failed to load at startup"
            }), 500
        
        # Decode base64 image
        try:
            img_data = base64.b64decode(data['image'])
            img = Image.open(io.BytesIO(img_data))
        except Exception as e:
            return jsonify({
                "error": "Invalid image data",
                "message": f"Failed to decode base64: {str(e)}"
            }), 400
        
        # Preprocess image
        img = img.convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx] * 100)
        predicted_class = CLASS_NAMES[class_idx]
        
        logger.info(f"Prediction: {predicted_class} ({confidence:.2f}%)")
        
        # Return result
        result = {
            "prediction": {
                "class": predicted_class,
                "confidence": round(confidence, 2)
            },
            "class": predicted_class,  # For backward compatibility
            "confidence": round(confidence, 2),
            "all_predictions": {
                CLASS_NAMES[i]: round(float(predictions[0][i] * 100), 2)
                for i in range(len(CLASS_NAMES))
            }
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500

# ===== ERROR HANDLERS =====

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Not found",
        "message": "The requested endpoint does not exist"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "Something went wrong on the server"
    }), 500

# ===== RUN SERVER =====

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)