import streamlit as st
import numpy as np
import cv2
import os
import urllib.request
import tensorflow as tf
from .streamlit_preproc import streamlit_processing


MODEL_PATH = "streamlit/ODM.keras"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1CRSOojRW3jqdKb9cJtJm-DQZnPej-Hwr"


if not os.path.exists(MODEL_PATH):
    try:
        print("Downloading model file...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")
    except Exception as e:
        st.error(f"Error downloading the model: {e}")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")
except Exception as e:
    model = None
    st.error(f"Failed to load model: {e}")

classes = {
    0: "Normal",
    1: "Diabetes",
    2: "Glaucoma",
    3: "Cataract",
    4: "Age related Macular Degeneration",
    5: "Hypertension",
    6: "Pathological Myopia",
    7: "Other diseases/abnormalities"
}

st.title("Eye Disease Detection")
uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    if image is None:
        st.error("Error loading the image.")
    else:
        processed_image = streamlit_processing(image, size=128)
        image_normalized = processed_image / 255.0
        image_input = np.expand_dims(image_normalized, axis=0)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        if model:
            try:
                prediction = model.predict(image_input)
                st.write(f"Prediction shape: {prediction.shape}")
                prediction = prediction.squeeze()
                st.write(f"Prediction after squeeze: {prediction}")
                predicted_class_index = np.argmax(prediction)
                predicted_class = classes.get(predicted_class_index, "Unknown")
                st.markdown(f"### Prediction: {predicted_class}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.error("Model is not loaded. Cannot make predictions.")
