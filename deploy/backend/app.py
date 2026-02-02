"""
Helmet Detection Computer Vision - Backend API
Flask REST API for safety helmet detection using VGG-16 transfer learning
"""

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os

# Flask app initialization
helmet_api = Flask("Helmet Detection API")

# Load the trained model once at startup (not per request)
MODEL_PATH = "helmet_detection_model_v1.keras"
print(f"Loading model from {MODEL_PATH}...")
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Warning: API will start but predictions will fail until model is added")
    model = None

@helmet_api.get('/')
def home():
    """Health check endpoint with API documentation."""
    return jsonify({
        "service": "Helmet Detection Computer Vision API",
        "version": "1.0",
        "model": "VGG-16 Transfer Learning + FFNN",
        "status": "ready" if model is not None else "model_not_loaded",
        "endpoints": {
            "POST /v1/predict": "Detect helmet presence in uploaded image"
        },
        "usage": {
            "method": "POST",
            "content_type": "multipart/form-data",
            "field_name": "image",
            "accepted_formats": ["jpg", "jpeg", "png"],
            "max_file_size": "10MB"
        },
        "example": "curl -X POST https://your-domain/v1/predict -F 'image=@worker.jpg'"
    })

@helmet_api.post('/v1/predict')
def predict_helmet():
    """
    Predict whether a person is wearing a safety helmet.

    Expected input: Image file via multipart/form-data with field name 'image'
    Returns: JSON with classification result and confidence score
    """

    # Check if model is loaded
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please add helmet_detection_model_v1.keras to the backend directory.'
        }), 503

    # Validate request has file
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image file provided. Use field name "image" in multipart/form-data.'
        }), 400

    file = request.files['image']

    # Validate file is not empty
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Validate file format
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else None
    if file_ext not in allowed_extensions:
        return jsonify({
            'error': f'Invalid file format "{file_ext}". Allowed: {", ".join(allowed_extensions)}'
        }), 400

    try:
        # Read image from upload
        image_bytes = file.read()

        # Validate file size (10MB limit)
        if len(image_bytes) > 10 * 1024 * 1024:
            return jsonify({'error': 'File too large. Maximum size: 10MB'}), 400

        # Preprocess image
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to model input size (200x200)
        image = image.resize((200, 200))

        # Convert to numpy array
        image_array = np.array(image)

        # Normalize pixel values to [0, 1]
        image_array = image_array.astype('float32') / 255.0

        # Add batch dimension: (200, 200, 3) -> (1, 200, 200, 3)
        image_array = np.expand_dims(image_array, axis=0)

        # Make prediction
        prediction = model.predict(image_array, verbose=0)
        confidence = float(prediction[0][0])

        # Binary classification: threshold at 0.5
        predicted_class = 1 if confidence >= 0.5 else 0
        label = "Helmet Detected" if predicted_class == 1 else "No Helmet"

        # Return prediction
        return jsonify({
            'class': predicted_class,
            'confidence': round(confidence, 4),
            'label': label,
            'model_version': '1.0',
            'input_size': '200x200 RGB',
            'processing_time_ms': 0  # TODO: Add timing if needed
        })

    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'type': type(e).__name__
        }), 500

if __name__ == '__main__':
    # Run development server
    port = int(os.environ.get('PORT', 7860))
    helmet_api.run(
        debug=True,
        host='0.0.0.0',
        port=port
    )
