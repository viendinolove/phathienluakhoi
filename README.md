# 🔥 Fire Detection AI Service

AI-powered fire and smoke detection service using TensorFlow and Flask.

## Model

- **Architecture:** MobileNetV2 (Transfer Learning)
- **Input:** 224x224 RGB images
- **Output:** Fire, Neutral, Smoke classifications
- **Accuracy:** 90%+

## API Endpoints

### GET /
Health check and service info

### GET /health
Detailed health status

### POST /predict
Main prediction endpoint

**Request:**
```json
{
  "image": "base64_encoded_image_string"
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "class": "Fire",
    "confidence": 95.23,
    "is_danger": true,
    "all_predictions": {
      "Fire": 95.23,
      "Neutral": 2.14,
      "Smoke": 2.63
    }
  },
  "timestamp": "2024-01-01T12:00:00",
  "database_id": 123
}
```

## Deployment

Deployed on Render.com with Supabase integration.

## Environment Variables

- `SUPABASE_URL`: Supabase project URL
- `SUPABASE_SERVICE_KEY`: Supabase service role key
- `PORT`: Server port (default: 5000)

## Local Development
```bash
pip install -r requirements.txt
python app.py
```

## License

MIT