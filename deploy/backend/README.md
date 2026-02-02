---
title: Helmet Detection Backend
emoji: ⛑️
colorFrom: red
colorTo: red
sdk: docker
pinned: false
license: mit
---

# Helmet Detection Computer Vision - Backend API

Flask REST API for safety helmet detection using VGG-16 transfer learning.

## API Endpoints

### GET /
Health check and API documentation

### POST /v1/predict
Detect helmet presence in uploaded image

**Input**: multipart/form-data with field name "image"
**Accepts**: JPG, JPEG, PNG (max 10MB)
**Returns**: JSON with class, confidence, and label

**Example**:
```bash
curl -X POST https://rnunez245-helmet-detection-backend.hf.space/v1/predict \
  -F "image=@test_image.jpg"
```

**Response**:
```json
{
  "class": 1,
  "confidence": 0.9987,
  "label": "Helmet Detected",
  "model_version": "1.0"
}
```

## Model Details
- **Architecture**: VGG-16 (ImageNet) + Feed-Forward Neural Network
- **Accuracy**: 100% on test set
- **Input**: 200×200 RGB images
- **Output**: Binary classification (helmet/no helmet)

## Links
- **Frontend**: [helmet-detection-frontend](https://huggingface.co/spaces/rnunez245/helmet-detection-frontend)
- **GitHub**: [helmet-detection-computer-vision](https://github.com/rnunez245/helmet-detection-computer-vision)

## Note
⚠️ Model file will be added soon. API is live but predictions require the trained model file.
