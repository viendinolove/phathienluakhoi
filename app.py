"""
============================================
FIRE & SMOKE DETECTION API
============================================
Render + TensorFlow + Supabase (STABLE)
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
# SUPABASE INIT (STABLE)
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

# ============================================
# FLASK APP
# ============================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# ============================================
# MODEL
# ============================================

MODEL_PATH = "fire_smoke_detection_model"
model = None

def load_model():
    global model
    if model is None:
        print("🔥 Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded")
    return model

# ============================================
# IMAGE
# ============================================

def preprocess_image(base64_image):
    img = Image.open(io.BytesIO(base64.b64decode(base64_image)))
    img = img.convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ============================================
# ROUTES
# ============================================

@app.route("/")
def index():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "supabase": "connected" if supabase else "disabled"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Missing image"}), 400

    mdl = load_model()
    img = preprocess_image(data["image"])
    preds = mdl.predict(img, verbose=0)[0]

    labels = ["Fire", "Neutral", "Smoke"]
    idx = int(np.argmax(preds))

    result = {
        "class": labels[idx],
        "confidence": round(float(preds[idx]) * 100, 2),
        "timestamp": datetime.utcnow().isoformat()
    }

    if supabase:
        supabase.table("predictions").insert(result).execute()

    return jsonify(result)

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
