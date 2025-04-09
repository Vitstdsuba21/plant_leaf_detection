import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# -------------------------------
# Download and Load Pre-trained Model
# -------------------------------

# Google Drive file ID and URL
# -------------------------------
# Download and Load Pre-trained Model
# -------------------------------

# Google Drive file ID and URL
file_id = "1KU4h7ztVsEmhbZj_g8GaRBK6S9q8fTag"
url = f"https://drive.google.com/uc?id={file_id}"
model_file = "trained_model.keras"

# Download the model if not already present
if not os.path.exists(model_file):
    with st.spinner("Downloading model..."):
        result = gdown.download(url, model_file, quiet=False, use_cookies=False)
        if result is None or not os.path.exists(model_file):
            st.error("❌ Model download failed. Please check the Google Drive link and file permissions.")
            st.stop()

# Try loading the model safely
try:
    model = tf.keras.models.load_model(model_file)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# -------------------------------
# Prediction Function
# -------------------------------
def model_prediction(test_image):
    image = Image.open(test_image).convert("RGB")
    image = image.resize((224, 224))  # Resize to model input size
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    input_arr = input_arr / 255.0  # Normalize
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    return result_index, confidence

# -------------------------------
# Class Labels
# -------------------------------
class_names = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
    'Blueberry__healthy', 'Cherry(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)__healthy', 'Corn(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)__Common_rust', 'Corn_(maize)__Northern_Leaf_Blight', 'Corn(maize)___healthy', 
    'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 
    'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy', 
    'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy', 
    'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew', 
    'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot', 
    'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold', 
    'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite', 
    'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato___healthy'
]

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Plant Disease Recognition", layout="wide")
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("🌱 PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpg"
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    st.markdown(""" 
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help identify plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases.

    ### How It Works
    1. **Upload Image:** Go to the *Disease Recognition* page.
    2. **Analysis:** The system processes the image.
    3. **Results:** View predictions and disease info.

    ### Why Choose Us?
    - ✅ Accurate
    - ✅ Fast
    - ✅ Easy to use

    ### Get Started
    Click *Disease Recognition* in the sidebar and upload an image!
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    ### Dataset Info
    - 87K RGB images (healthy + diseased leaves)
    - 38 categories
    - Train: 80% | Validation: 20% | Test: 33 samples

    ### Source
    Originally hosted on a public GitHub repository with offline augmentation applied.

    ### Use Case
    Helps farmers and agronomists detect diseases early using mobile devices or laptops.
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("🩺 Disease Recognition")
    test_image = st.file_uploader("📤 Upload a Leaf Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("Analyzing the image...")
            result_index, confidence = model_prediction(test_image)
            prediction = class_names[result_index]
            st.success(f"🌿 The model predicts: **{prediction}**")
            st.info(f"🔍 Confidence: **{confidence * 100:.2f}%**")
