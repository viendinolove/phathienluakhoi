"""
============================================
FIRE & SMOKE DETECTION API (FINAL)
============================================
Render + TensorFlow (CPU) + Supabase
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
# SUPABASE INIT
# ============================================

supabase = None
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase connected")
    except Exception as e:
        print(f"⚠️ Supabase disabled: {e}")
        supabase = None

# ============================================
# FLASK APP
# ============================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# ============================================
# MODEL LOADING
# ============================================

MODEL_PATH = "fire_smoke_detection_model"  # thư mục SavedModel
model = None

def load_model():
    global model
    if model is None:
        print("🔥 Loading TensorFlow model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded")
    return model

# Preload model cho Gunicorn
try:
    load_model()
except Exception as e:
    print(f"⚠️ Model preload failed: {e}")

# ============================================
# IMAGE PREPROCESSING
# ============================================

def preprocess_image(base64_image: str):
    try:
        img_bytes = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 224))

        arr = np.asarray(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")

# ============================================
# ROUTES
# ============================================

@app.route("/")
def index():
    return jsonify({
        "status": "online",
        "service": "Fire & Smoke Detection API",
        "model_loaded": model is not None,
        "supabase": "connected" if supabase else "disabled",
        "endpoints": {
            "health": "/health (GET)",
            "predict": "/predict (POST)"
        }
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model": "loaded" if model else "not_loaded",
        "supabase": "connected" if supabase else "disabled",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({
                "error": "Missing image field",
                "example": {"image": "base64_string"}
            }), 400

        mdl = load_model()
        img = preprocess_image(data["image"])

        preds = mdl.predict(img, verbose=0)[0]

        labels = ["Fire", "Neutral", "Smoke"]
        idx = int(np.argmax(preds))

        result = {
            "class": labels[idx],
            "confidence": round(float(preds[idx]) * 100, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "probabilities": {
                labels[i]: round(float(preds[i]) * 100, 2)
                for i in range(len(labels))
            }
        }

        if supabase:
            try:
                supabase.table("predictions").insert({
                    "class": result["class"],
                    "confidence": result["confidence"],
                    "fire_prob": result["probabilities"]["Fire"],
                    "neutral_prob": result["probabilities"]["Neutral"],
                    "smoke_prob": result["probabilities"]["Smoke"],
                    "created_at": result["timestamp"]
                }).execute()
                result["saved_to_db"] = True
            except Exception as db_err:
                print(f"⚠️ DB save failed: {db_err}")
                result["saved_to_db"] = False

        return jsonify(result)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print("❌ Predict error:", e)
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "Payload too large (max 16MB)"}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Not found",
        "available": ["/", "/health", "/predict"]
    }), 404

# ============================================
# MAIN (LOCAL ONLY)
# ============================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
