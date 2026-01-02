"""
============================================
FIRE & SMOKE DETECTION API
============================================
Deploy: Render
Fix: Supabase proxy crash
============================================
"""

# ============================================
# FORCE REMOVE PROXY ENV (CRITICAL FIX)
# ============================================

import os

for key in [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
]:
    os.environ.pop(key, None)

# ============================================
# IMPORTS
# ============================================

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime

# ============================================
# SUPABASE INIT (SAFE)
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
        print(f"❌ Supabase init failed: {e}")
else:
    print("⚠️ Supabase disabled (missing env)")

# ============================================
# FLASK APP
# ============================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# ============================================
# MODEL LOAD
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
# IMAGE PREPROCESS
# ============================================

def preprocess_image(base64_image: str):
    img_bytes = base64.b64decode(base64_image)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ============================================
# ROUTES
# ============================================

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "online",
        "service": "Fire Smoke Detection AI",
        "model_loaded": model is not None,
        "supabase": "connected" if supabase else "disabled"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing base64 image"}), 400

    mdl = load_model()
    img = preprocess_image(data["image"])
    preds = mdl.predict(img, verbose=0)[0]

    labels = ["Fire", "Neutral", "Smoke"]
    idx = int(np.argmax(preds))

    result = {
        "class": labels[idx],
        "confidence": round(float(preds[idx]) * 100, 2),
        "scores": {
            labels[i]: round(float(preds[i]) * 100, 2)
            for i in range(len(labels))
        },
        "created_at": datetime.utcnow().isoformat()
    }

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
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
