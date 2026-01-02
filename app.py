"""
============================================
FIRE & SMOKE DETECTION API
============================================
Render + TensorFlow + Supabase (FIXED)
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

# ============================================
# SUPABASE INIT (FIXED VERSION)
# ============================================

supabase = None
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        # Fix: Không truyền proxy parameter
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase connected successfully")
    except Exception as e:
        print(f"❌ Supabase init failed: {e}")
        supabase = None

# ============================================
# FLASK APP
# ============================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max

# ============================================
# MODEL LOADING
# ============================================

MODEL_PATH = "fire_smoke_detection_model"
model = None

def load_model():
    """Load TensorFlow model once"""
    global model
    if model is None:
        try:
            print("🔥 Loading model...")
            model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
    return model

# ============================================
# IMAGE PREPROCESSING
# ============================================

def preprocess_image(base64_image):
    """
    Chuyển đổi base64 image thành tensor cho model
    Input: base64 string (không có prefix)
    Output: numpy array (1, 224, 224, 3)
    """
    try:
        # Decode base64
        img_data = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB và resize
        img = img.convert("RGB").resize((224, 224))
        
        # Convert to array và normalize
        arr = np.asarray(img, dtype=np.float32) / 255.0
        
        # Add batch dimension
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")

# ============================================
# API ROUTES
# ============================================

@app.route("/")
def index():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "service": "Fire & Smoke Detection API",
        "model_loaded": model is not None,
        "supabase": "connected" if supabase else "disabled",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/ (GET)"
        }
    })

@app.route("/health")
def health():
    """Detailed health check"""
    return jsonify({
        "status": "healthy",
        "model": "loaded" if model else "not loaded",
        "supabase": "connected" if supabase else "disabled",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint
    
    Request body (JSON):
    {
        "image": "base64_encoded_image_string"
    }
    
    Response:
    {
        "class": "Fire|Neutral|Smoke",
        "confidence": 95.23,
        "timestamp": "2026-01-03T10:30:00.000000",
        "probabilities": {
            "Fire": 95.23,
            "Neutral": 2.45,
            "Smoke": 2.32
        }
    }
    """
    try:
        # Validate request
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({
                "error": "Missing 'image' field in request body",
                "example": {"image": "base64_string_here"}
            }), 400
        
        # Load model nếu chưa load
        mdl = load_model()
        
        # Preprocess image
        img = preprocess_image(data["image"])
        
        # Predict
        preds = mdl.predict(img, verbose=0)[0]
        
        # Class labels
        labels = ["Fire", "Neutral", "Smoke"]
        idx = int(np.argmax(preds))
        
        # Prepare result
        result = {
            "class": labels[idx],
            "confidence": round(float(preds[idx]) * 100, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "probabilities": {
                labels[i]: round(float(preds[i]) * 100, 2) 
                for i in range(len(labels))
            }
        }
        
        # Save to Supabase (nếu có)
        if supabase:
            try:
                supabase.table("predictions").insert({
                    "class": result["class"],
                    "confidence": result["confidence"],
                    "timestamp": result["timestamp"],
                    "fire_prob": result["probabilities"]["Fire"],
                    "neutral_prob": result["probabilities"]["Neutral"],
                    "smoke_prob": result["probabilities"]["Smoke"]
                }).execute()
                result["saved_to_db"] = True
            except Exception as db_error:
                print(f"⚠️ Database save failed: {db_error}")
                result["saved_to_db"] = False
        
        return jsonify(result)
        
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "error": "File too large",
        "max_size": "16MB"
    }), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/health", "/predict"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error"
    }), 500

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    # Pre-load model khi start
    try:
        load_model()
        print("🚀 Server starting...")
    except Exception as e:
        print(f"⚠️ Warning: Could not pre-load model: {e}")
    
    # Run Flask app
    port = int(os.getenv("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )