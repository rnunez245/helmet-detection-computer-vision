---
title: Helmet Detection Frontend
emoji: ⛑️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# Helmet Detection Computer Vision - Interactive Demo

AI-powered safety helmet detection for automated workplace safety compliance monitoring.

## How to Use

1. **Try Examples**: Click example buttons in sidebar to test instantly
2. **Upload Your Own**: Upload any image (JPG/PNG) of a worker
3. **Analyze**: Click "Analyze Image" to run detection
4. **View Results**: See prediction with confidence score and visual feedback

## Best Results With
- Clear, well-lit photos
- Construction/industrial settings
- Single person visible
- Standard safety helmets (hard hats)

## Model Details
- **Architecture**: VGG-16 Transfer Learning + Feed-Forward Neural Network
- **Accuracy**: 100% on test set
- **Training Data**: 631 images from construction sites and industrial environments
- **Use Case**: Automated workplace safety compliance monitoring

## Limitations
This model is optimized for workplace safety compliance in construction/industrial environments. Performance may vary on:
- Multiple people in frame
- Extreme angles or poor lighting
- Non-safety helmets (bike, sports helmets)

## Links
- **Backend API**: [helmet-detection-backend](https://huggingface.co/spaces/rnunez245/helmet-detection-backend)
- **GitHub Repository**: [helmet-detection-computer-vision](https://github.com/rnunez245/helmet-detection-computer-vision)

## Author
**Ruben Nunez** - UT Austin AI/ML Program Graduate

## Note
⚠️ Example images will be added soon. You can upload your own images to test!
