"""
============================================
FIRE DETECTION AI SERVICE
============================================
Flask API for fire/smoke detection using TensorFlow model
Integrated with Supabase for data storage
Deploy: Render.com
============================================
"""

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
from datetime import datetime

# Supabase client
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("⚠️  Supabase client not available - running in test mode")

# ============================================
# FLASK APP SETUP
# ============================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

# ============================================
# ENVIRONMENT VARIABLES
# ============================================

SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_KEY', '')

# Initialize Supabase client
supabase = None
if SUPABASE_AVAILABLE and SUPABASE_URL and SUPABASE_KEY:
    try:
        # Đã xóa tham số proxy gây lỗi
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client initialized")
    except Exception as e:
        print(f"⚠️  Supabase initialization failed: {e}")

# ============================================
# MODEL LOADING
# ============================================

MODEL_PATH = './fire_smoke_detection_model'
model = None

def load_model():
    """Load TensorFlow model (cached)"""
    global model
    if model is None:
        try:
            print(f"🔥 Loading model from {MODEL_PATH}...")
            model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Model loaded successfully!")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
    return model

# ============================================
# IMAGE PREPROCESSING
# ============================================

def preprocess_image(image_data, source='base64'):
    try:
        # Decode image
        if source == 'base64':
            img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        else:
            img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

# ============================================
# ROUTES
# ============================================

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "service": "Fire Detection AI",
        "model": "MobileNetV2",
        "version": "2.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        },
        "supabase": "connected" if supabase else "not configured"
    })

@app.route('/health')
def health():
    """Detailed health check"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "supabase_connected": supabase is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        if 'image' not in data:
            return jsonify({"success": False, "error": "No image provided in request"}), 400
        
        # Load model
        loaded_model = load_model()
        
        # Preprocess image
        print("📸 Preprocessing image...")
        img_array = preprocess_image(data['image'], source='base64')
        
        # Make prediction
        print("🤖 Running inference...")
        predictions = loaded_model.predict(img_array, verbose=0)[0]
        
        class_names = ['Fire', 'Neutral', 'Smoke']
        
        # Get prediction results
        predicted_idx = int(np.argmax(predictions))
        predicted_class = class_names[predicted_idx]
        confidence = float(predictions[predicted_idx] * 100)
        
        # Determine if dangerous
        is_danger = predicted_class in ['Fire', 'Smoke']
        
        # All predictions
        all_preds = {
            'Fire': float(predictions[0] * 100),
            'Neutral': float(predictions[1] * 100),
            'Smoke': float(predictions[2] * 100)
        }
        
        print(f"✅ Prediction: {predicted_class} ({confidence:.2f}%)")
        
        # Prepare response
        response_data = {
            "success": True,
            "prediction": {
                "class": predicted_class,
                "confidence": round(confidence, 2),
                "is_danger": is_danger,
                "all_predictions": {k: round(v, 2) for k, v in all_preds.items()}
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to Supabase
        if supabase:
            try:
                print("💾 Saving to Supabase...")
                result = supabase.table('predictions').insert({
                    'predicted_class': predicted_class,
                    'confidence': round(confidence, 2),
                    'is_danger': is_danger,
                    'all_predictions': all_preds
                }).execute()
                
                if result.data:
                    response_data['database_id'] = result.data[0]['id']
                    print(f"✅ Saved to database with ID: {result.data[0]['id']}")
            except Exception as e:
                print(f"⚠️  Database save failed: {e}")
                response_data['database_error'] = str(e)
        
        return jsonify(response_data)
    
    except ValueError as e:
        return jsonify({"success": False, "error": f"Image processing error: {str(e)}"}), 400
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500

@app.route('/test', methods=['GET'])
def test():
    try:
        loaded_model = load_model()
        return jsonify({
            "status": "ok",
            "model_loaded": True,
            "input_shape": str(loaded_model.input_shape),
            "output_shape": str(loaded_model.output_shape)
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    try:
        load_model()
        print("✅ Model preloaded successfully")
    except Exception as e:
        print(f"⚠️  Model preload failed: {e}")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)