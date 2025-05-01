import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from keras import models


class_names = ["Normal", "Diabetes", "Glaucoma", "Cataract",
               "Age related Macular Degeneration", "Hypertension",
               "Pathological Myopia", "Other diseases/abnormalities"]

model = models.load_model("eye_disease_model.h5")

st.title("Eye Disease Detection")
uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_resized = cv2.resize(image, (128, 128))
    image_normalized = image_resized / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    prediction = model.predict(image_input)
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f"### Prediction: {predicted_class}")