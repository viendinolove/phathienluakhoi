"""
============================================
FIRE DETECTION AI SERVICE
============================================
Flask API for fire/smoke detection
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

# ============================================
# SUPABASE (SAFE INIT - NO PROXY)
# ============================================

supabase = None
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase connected")
    except Exception as e:
        print(f"❌ Supabase init failed: {e}")

# ============================================
# FLASK APP
# ============================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# ============================================
# MODEL
# ============================================

MODEL_PATH = "fire_smoke_detection_model"
model = None

def load_model():
    global model
    if model is None:
        print("🔥 Loading TensorFlow model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded")
    return model

# ============================================
# IMAGE PROCESSING
# ============================================

def preprocess_image(base64_image):
    img = Image.open(io.BytesIO(base64.b64decode(base64_image)))
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ============================================
# ROUTES
# ============================================

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "service": "Fire Detection AI",
        "model_loaded": model is not None,
        "supabase": "connected" if supabase else "disabled"
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    model = load_model()
    img = preprocess_image(data["image"])
    preds = model.predict(img, verbose=0)[0]

    labels = ["Fire", "Neutral", "Smoke"]
    idx = int(np.argmax(preds))

    result = {
        "class": labels[idx],
        "confidence": round(float(preds[idx]) * 100, 2),
        "all_predictions": {
            labels[i]: round(float(preds[i]) * 100, 2)
            for i in range(3)
        },
        "timestamp": datetime.utcnow().isoformat()
    }

    # Save to Supabase
    if supabase:
        try:
            supabase.table("predictions").insert(result).execute()
        except Exception as e:
            result["db_error"] = str(e)

    return jsonify(result)

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
