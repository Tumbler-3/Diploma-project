import streamlit as st
import numpy as np
import cv2
from keras import models

class_names = ["Normal", "Diabetes", "Glaucoma", "Cataract",
               "Age related Macular Degeneration", "Hypertension",
               "Pathological Myopia", "Other diseases/abnormalities"]

try:
    model = models.load_model("eye_disease_model.h5")
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None  

st.title("Eye Disease Detection")
uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    if image is None:
        st.error("Error loading the image.")
    else:
        image_resized = cv2.resize(image, (128, 128)) 
        image_normalized = image_resized / 255.0  
        image_input = np.expand_dims(image_normalized, axis=0)  

        st.image(image, caption="Uploaded Image", use_column_width=True)

        try:
            prediction = model.predict(image_input)
            print(f"Prediction: {prediction}")
            print(f"Prediction shape: {prediction.shape}")

            prediction = prediction.squeeze()  
            predicted_class = class_names[np.argmax(prediction)]
            st.markdown(f"### Prediction: {predicted_class}")
        
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    if model is None:
        st.error("The model could not be loaded. Please check the model file.")
