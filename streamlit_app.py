import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Loading Keras model
@st.cache_resource
def get_model():
    return load_model("face_mask_detector.h5")

model = get_model()

# Image preprocessing
def prepare_image(image):
    image = image.convert("RGB").resize((150, 150))     # Convert and resize
    array = np.asarray(image) / 255.0                   # Normalize to [0,1]
    return np.expand_dims(array, axis=0)                # Shape: (1, 150, 150, 3)

# Streamlit UI
st.title("Face Mask Detection Project")
st.write("Upload an image to detect whether the person is wearing a mask.")

file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed = prepare_image(image)
    prob = model.predict(processed)[0][0]

    # Applying threshold
    result_label = "😷 Mask" if prob <= 0.5 else "❌ No Mask"
    conf = (1 - prob) * 100 if prob <= 0.5 else prob * 100

    st.markdown(f"### Prediction: *{result_label}*")
    st.write(f"Confidence: {conf:.2f}%")