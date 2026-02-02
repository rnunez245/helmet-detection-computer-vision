# ⛑️ Helmet Detection Computer Vision

![Live Demo](https://img.shields.io/badge/demo-live-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?style=for-the-badge&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**AI-powered safety helmet detection for automated workplace safety compliance monitoring**

🔗 **[Live Demo (Frontend)](https://huggingface.co/spaces/rnunez245/helmet-detection-frontend)** | **[API Documentation](https://huggingface.co/spaces/rnunez245/helmet-detection-backend)**

---

## 📋 Business Problem

**Client**: SafeGuard Corp (Industrial Safety Solutions Provider)

**Challenge**: Manual monitoring of safety helmet compliance is inefficient, error-prone, and cannot scale across large construction sites and manufacturing facilities.

**Impact of Problem**:
- ⚠️ Head injuries account for 10% of workplace fatalities
- 📉 Manual monitoring coverage limited to 20-30% of site areas
- 💰 OSHA violations result in $14,502 per serious incident
- 👥 Safety managers overwhelmed with monitoring responsibilities

**Business Opportunity**: Automated computer vision system for real-time helmet detection with 24/7 monitoring capability.

---

## 🎯 Solution

An **AI-powered helmet detection system** using VGG-16 transfer learning to automatically identify safety helmet compliance in workplace environments.

### Model Performance

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **Accuracy** | 100% | 100% | 100% |
| **Precision** | 100% | 100% | 100% |
| **Recall** | 100% | 100% | 100% |

**Model**: VGG-16 Transfer Learning (ImageNet pre-trained weights) + Feed-Forward Neural Network with data augmentation

**Dataset**: 631 images (311 with helmet, 320 without helmet) from construction sites and industrial environments

---

## 🔧 Technical Architecture

### Model Architecture
```
Input Image (200×200×3 RGB)
    ↓
VGG-16 Base (Frozen, ImageNet Weights)
    ↓
Flatten
    ↓
Dense(128, ReLU) → Dropout(0.5) → Dense(32, ReLU) → Dense(1, Sigmoid)
    ↓
Binary Classification: Helmet (1) / No Helmet (0)
```

### Deployment Architecture
```
┌─────────────────────┐      ┌─────────────────────┐
│   Streamlit UI      │──────│    Flask API        │
│   (Frontend)        │ HTTP │    (Backend)        │
│   Image Upload      │ POST │    VGG-16 Model     │
└─────────────────────┘      └─────────────────────┘
        │                             │
        │                             ▼
        │                    ┌─────────────────────┐
        └───────────────────▶│  helmet_detection   │
          Visual Results     │  _model_v1.keras    │
                             └─────────────────────┘
```

---

## 🛠️ Tech Stack

**Machine Learning**:
- TensorFlow 2.19 (Keras API)
- VGG-16 (Transfer Learning)
- ImageDataGenerator (Data Augmentation)

**Backend API**:
- Flask 2.2.2 (REST API)
- Gunicorn (WSGI Server)
- Pillow (Image Processing)

**Frontend UI**:
- Streamlit 1.45.0
- Requests (HTTP Client)

**Deployment**:
- HuggingFace Spaces (Docker + Streamlit)
- Docker (Containerization)

---

## 📁 Project Structure

```
helmet-detection-computer-vision/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── .gitignore                         # Python/ML exclusions
├── notebooks/
│   └── project_5_helmet_detection_computer_vision_business_case.ipynb
├── model/
│   └── helmet_detection_model_v1.keras
├── test_images/                       # Example images for testing
│   ├── helmet_0.jpg
│   ├── helmet_1.jpg
│   ├── no_helmet_0.jpg
│   └── no_helmet_1.jpg
└── deploy/
    ├── backend/                       # Flask API
    │   ├── app.py                     # API endpoints
    │   ├── Dockerfile                 # Container config
    │   ├── requirements.txt           # Dependencies
    │   └── helmet_detection_model_v1.keras
    └── frontend/                      # Streamlit UI
        ├── app.py                     # Interactive interface
        ├── requirements.txt           # Dependencies
        └── examples/                  # Click-to-test images
            ├── helmet_0.jpg
            ├── helmet_1.jpg
            ├── no_helmet_0.jpg
            └── no_helmet_1.jpg
```

---

## 🚀 Deployment

### Live Demo

**Frontend**: [https://huggingface.co/spaces/rnunez245/helmet-detection-frontend](https://huggingface.co/spaces/rnunez245/helmet-detection-frontend)

**Backend API**: [https://huggingface.co/spaces/rnunez245/helmet-detection-backend](https://huggingface.co/spaces/rnunez245/helmet-detection-backend)

### Local Development

#### Backend (Flask API)

```bash
cd deploy/backend

# Install dependencies
pip install -r requirements.txt

# Run API server
python app.py

# API will be available at http://localhost:7860
```

**Test the API**:
```bash
curl -X POST http://localhost:7860/v1/predict \
  -F "image=@path/to/your/image.jpg"
```

#### Frontend (Streamlit UI)

```bash
cd deploy/frontend

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

# UI will open at http://localhost:8501
```

---

## 📊 Key Insights

1. **Transfer Learning is Highly Effective**: VGG-16 pre-trained on ImageNet achieved 100% accuracy with minimal training epochs
2. **Data Augmentation Adds Robustness**: Rotation, shift, shear, and zoom transformations improve real-world generalization
3. **Binary Classification Simplifies Deployment**: Clear threshold (0.5) for helmet/no-helmet decision-making
4. **Balanced Dataset Eliminates Bias**: Nearly equal class distribution (311 vs 320 images) prevents model bias
5. **Lightweight Preprocessing**: Only resize (200×200) and normalize (/255) required for inference

---

## 📈 Business Recommendations

### Deployment Strategy
1. **CCTV Integration**: Connect to existing security camera systems for continuous monitoring
2. **Real-time Alerts**: Automatically notify safety managers when violations detected
3. **Entrance Gate Deployment**: Verify compliance before site entry
4. **Mobile App Integration**: Enable spot-checks and field inspections

### ROI Projections
| Benefit | Annual Impact |
|---------|--------------|
| Reduced Head Injuries | $500K+ (avoided medical/legal costs) |
| OSHA Compliance | $150K+ (avoided fines) |
| Labor Savings | $120K+ (automated monitoring vs. manual) |
| Insurance Premium Reduction | $80K+ (demonstrated safety measures) |
| **Total Annual Savings** | **$850K+** |

### Continuous Improvement
1. **Expand Dataset**: Collect edge cases (poor lighting, partial occlusions, unusual angles)
2. **Multi-class Detection**: Extend to other PPE (safety vests, goggles, gloves)
3. **Anomaly Detection**: Flag unusual site activities beyond PPE compliance
4. **Integration Pipelines**: Connect to incident reporting and compliance systems

---

## 🎯 Model Scope & Limitations

### Best Results With:
✅ Clear, well-lit photos
✅ Construction/industrial settings
✅ Single person visible
✅ Head/upper body in frame
✅ Standard safety helmets (hard hats)

### May Not Work Well With:
❌ Multiple people in frame
❌ Extreme angles (top-down, back view)
❌ Poor lighting/heavy blur
❌ Non-safety helmets (bike, sports helmets)
❌ Artistic/heavily filtered images

**Note**: This model is optimized for workplace safety compliance in construction and industrial environments. It's trained on 631 images from similar settings. Performance may vary on edge cases outside this scope.

---

## 👨‍💻 Author

**Ruben Nunez**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/rnunez245)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=for-the-badge&logo=github)](https://github.com/rnunez245)

*Graduate Certificate in Artificial Intelligence and Machine Learning*
**The University of Texas at Austin**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset: Public helmet detection dataset (631 images)
- Pre-trained Model: VGG-16 (ImageNet weights)
- Framework: TensorFlow/Keras
- Deployment Platform: HuggingFace Spaces
