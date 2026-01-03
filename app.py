"""
============================================
FIRE & SMOKE DETECTION SYSTEM - DEMO VERSION
============================================
Features: API JSON + Visual Dashboard
"""

from flask import Flask, request, jsonify, Response, send_file
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont # Thêm thư viện vẽ
import io
import base64
import os
from datetime import datetime

# ============================================
# CONFIG & GLOBALS
# ============================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Biến toàn cục để lưu ảnh mới nhất phục vụ Demo
latest_visualized_frame = None 

# Supabase (Giữ nguyên của bạn)
supabase = None
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase connected")
    except Exception as e:
        print(f"❌ Supabase failed: {e}")

# ============================================
# MODEL STUFF
# ============================================
MODEL_PATH = "fire_smoke_detection_model"
model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded")
    return model

try:
    load_model()
except:
    pass

# ============================================
# HELPER: VẼ CẢNH BÁO LÊN ẢNH
# ============================================
def visualize_prediction(pil_image, label, confidence):
    """Vẽ khung và chữ lên ảnh để hiển thị Demo"""
    draw = ImageDraw.Draw(pil_image)
    
    # Chọn màu: Đỏ cho Fire, Xám cho Smoke, Xanh cho Neutral
    color = (0, 255, 0) # Green
    if label == "Fire": color = (255, 0, 0) # Red
    elif label == "Smoke": color = (128, 128, 128) # Gray
    
    # Vẽ chữ (Nếu không có font thì dùng default)
    text = f"{label}: {confidence}%"
    
    # Vẽ hình chữ nhật nền cho chữ để dễ đọc
    # Tọa độ (10, 10)
    draw.rectangle([(5, 5), (150, 25)], fill="black")
    draw.text((10, 10), text, fill=color)
    
    # Vẽ khung bao quanh ảnh nếu có cháy
    if label == "Fire":
        draw.rectangle([(0,0), (pil_image.width-1, pil_image.height-1)], outline="red", width=5)
        
    return pil_image

# ============================================
# ROUTES
# ============================================

@app.route("/")
def index():
    """Trang Dashboard để xem Demo"""
    html_dashboard = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🔥 AI Fire Detection System</title>
        <meta http-equiv="refresh" content="2"> <style>
            body { font-family: Arial, sans-serif; text-align: center; background: #222; color: white; }
            .container { margin-top: 50px; }
            img { border: 5px solid #fff; border-radius: 10px; max-width: 100%; }
            h1 { color: #f39c12; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>HỆ THỐNG CẢNH BÁO CHÁY AIOT</h1>
            <p>Trạng thái thời gian thực từ ESP32-CAM</p>
            <br>
            <img src="/latest_frame" alt="Waiting for ESP32 stream..." width="640">
            <p><i>Hệ thống tự động cập nhật mỗi 2 giây</i></p>
        </div>
    </body>
    </html>
    """
    return html_dashboard

@app.route("/latest_frame")
def get_latest_frame():
    """Trả về ảnh đã được AI xử lý gần nhất"""
    global latest_visualized_frame
    if latest_visualized_frame:
        return send_file(latest_visualized_frame, mimetype='image/jpeg')
    else:
        return "No image received yet", 404

@app.route("/predict", methods=["POST"])
def predict():
    global latest_visualized_frame
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image"}), 400
        
        # 1. Decode ảnh
        img_data = base64.b64decode(data["image"])
        img_pil = Image.open(io.BytesIO(img_data)).convert("RGB")
        
        # 2. Xử lý cho AI (Resize)
        img_ai = img_pil.resize((224, 224))
        arr = np.asarray(img_ai, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        
        # 3. Predict
        mdl = load_model()
        preds = mdl.predict(arr, verbose=0)[0]
        labels = ["Fire", "Neutral", "Smoke"]
        idx = int(np.argmax(preds))
        label = labels[idx]
        conf = round(float(preds[idx]) * 100, 2)
        
        # 4. Vẽ kết quả lên ảnh gốc (để hiển thị Dashboard)
        # Resize ảnh gốc to ra chút để xem cho rõ nếu ESP gửi ảnh nhỏ
        img_display = img_pil.resize((640, 480)) 
        img_display = visualize_prediction(img_display, label, conf)
        
        # Lưu vào bộ nhớ RAM để route /latest_frame lấy ra hiển thị
        byte_io = io.BytesIO()
        img_display.save(byte_io, 'JPEG')
        byte_io.seek(0)
        latest_visualized_frame = byte_io

        # 5. Lưu Supabase (Giữ nguyên logic của bạn)
        if supabase:
            # ... (Code lưu Supabase cũ của bạn giữ nguyên ở đây)
            pass

        return jsonify({"class": label, "confidence": conf})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))