"""
Helmet Detection Computer Vision - Streamlit Frontend
Interactive UI for safety helmet detection with example images
"""

import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="⛑️",
    layout="wide"
)

st.title("⛑️ Safety Helmet Detection System")
st.markdown("*AI-powered workplace safety compliance monitoring using VGG-16 computer vision*")
st.markdown("---")

# Sidebar with info and examples
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model**: VGG-16 Transfer Learning + FFNN

    **Accuracy**: 100% on test set

    **Use Case**: Automated detection of safety helmet compliance in workplace environments

    **How to use**:
    1. Try example images OR upload your own
    2. Click "Analyze Image"
    3. View prediction results
    """)

    st.markdown("---")

    st.header("🎯 Try Examples")
    st.caption("Click any image to analyze")

    example_images = {
        "👷 Construction Worker (With Helmet)": "examples/helmet_0.jpg",
        "🏗️ Factory Worker (With Helmet)": "examples/helmet_1.jpg",
        "⚠️ Worker (No Helmet - Violation)": "examples/no_helmet_0.jpg",
        "🚧 Site Worker (No Helmet)": "examples/no_helmet_1.jpg"
    }

    for label, path in example_images.items():
        # Display thumbnail image
        try:
            image = Image.open(path)
            st.image(image, caption=label, use_container_width=True)

            # Clickable button below thumbnail
            short_label = "With Helmet ✓" if "With Helmet" in label else "No Helmet ⚠️"
            if st.button(f"Analyze: {short_label}", key=path, use_container_width=True):
                st.session_state.example_image = path
                st.session_state.uploaded_file = None  # Clear uploaded file
        except FileNotFoundError:
            # Fallback to text button if image missing
            if st.button(label, key=path, use_container_width=True):
                st.session_state.example_image = path
                st.session_state.uploaded_file = None

    st.markdown("---")

    st.markdown("""
    ### 📸 Best Results With:
    ✅ Clear, well-lit photos

    ✅ Construction/industrial settings

    ✅ Single person visible

    ✅ Head/upper body in frame

    ✅ Standard safety helmets (hard hats)

    ### ⚠️ May Not Work Well With:
    ❌ Multiple people in frame

    ❌ Extreme angles (top-down, back view)

    ❌ Poor lighting/heavy blur

    ❌ Non-safety helmets (bike, sports)

    ❌ Artistic/heavily filtered images

    ### ℹ️ Model Scope
    This model is optimized for workplace safety compliance in construction and industrial environments. It's trained on 631 images of workers in similar settings.
    """)

    st.markdown("---")

    st.markdown("""
    **Technical Details**:
    - Input: 200×200 RGB images
    - Architecture: VGG-16 (ImageNet) + Dense layers
    - Threshold: 0.5 for binary classification
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image showing a person's head area",
        key='file_uploader'
    )

    # Check for example image or uploaded file
    image_to_display = None
    image_source = None

    if 'example_image' in st.session_state and st.session_state.example_image:
        try:
            image_to_display = Image.open(st.session_state.example_image)
            image_source = "example"
        except:
            st.warning(f"Example image not found. Please upload an image instead.")
    elif uploaded_file is not None:
        image_to_display = Image.open(uploaded_file)
        image_source = "uploaded"
        st.session_state.uploaded_file = uploaded_file

    if image_to_display is not None:
        # Display uploaded/example image
        st.image(image_to_display, caption="Selected Image" if image_source == "example" else "Uploaded Image", use_container_width=True)

        # Show file details
        if image_source == "uploaded":
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
                "Image dimensions": f"{image_to_display.size[0]} × {image_to_display.size[1]} pixels"
            }
            st.json(file_details)

with col2:
    st.subheader("🔍 Detection Results")

    if image_to_display is not None:
        # Backend API URL (update this when deployed)
        API_URL = "https://rnunez245-helmet-detection-backend.hf.space/v1/predict"
        # For local testing: API_URL = "http://localhost:7860/v1/predict"

        if st.button("🚀 Analyze Image", type='primary', use_container_width=True):
            with st.spinner("Analyzing image..."):
                try:
                    # Prepare image for upload
                    buf = io.BytesIO()

                    # Get image data
                    if image_source == "example":
                        # Read example image file
                        with open(st.session_state.example_image, 'rb') as f:
                            image_data = f.read()
                        image_name = st.session_state.example_image.split('/')[-1]
                    else:
                        # Use uploaded file
                        uploaded_file.seek(0)
                        image_data = uploaded_file.read()
                        image_name = uploaded_file.name

                    # Call backend API
                    files = {'image': (image_name, io.BytesIO(image_data), 'image/jpeg')}
                    response = requests.post(API_URL, files=files, timeout=30)

                    if response.status_code == 200:
                        result = response.json()

                        # Extract prediction data
                        predicted_class = result['class']
                        confidence = result['confidence']
                        label = result['label']

                        # Display results with color coding
                        st.success("✅ Analysis Complete!")

                        # Color-coded result
                        if predicted_class == 1:
                            st.markdown(f"""
                            <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; border: 2px solid #28a745;">
                                <h2 style="color: #155724; margin: 0;">✓ HELMET DETECTED</h2>
                                <p style="color: #155724; font-size: 1.2em; margin: 10px 0 0 0;">Workplace safety compliant</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; border: 2px solid #dc3545;">
                                <h2 style="color: #721c24; margin: 0;">⚠ NO HELMET DETECTED</h2>
                                <p style="color: #721c24; font-size: 1.2em; margin: 10px 0 0 0;">Safety violation - immediate action required</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Metrics
                        st.markdown("### 📊 Confidence Metrics")

                        col_a, col_b, col_c = st.columns(3)

                        with col_a:
                            st.metric("Prediction", label)

                        with col_b:
                            st.metric("Confidence Score", f"{confidence:.2%}")

                        with col_c:
                            st.metric("Model Version", result.get('model_version', '1.0'))

                        # Confidence visualization
                        st.progress(confidence, text=f"Model Confidence: {confidence:.1%}")

                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        try:
                            error_data = response.json()
                            st.json(error_data)
                        except:
                            st.text(response.text)

                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timeout. Please try again.")

                except requests.exceptions.RequestException as e:
                    st.error(f"🔌 Connection error: {str(e)}")
                    st.info("Make sure the backend API is running at: " + API_URL)

                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
    else:
        st.info("👆 Please upload an image or select an example to begin analysis")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem;">
    <p><strong>Built by Ruben Nunez</strong> | UT Austin AI/ML Program</p>
    <p>Model: VGG-16 Transfer Learning | Accuracy: 100% on test set</p>
    <p>Use Case: Automated workplace safety compliance monitoring</p>
</div>
""", unsafe_allow_html=True)
