import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Class names for predictions
class_names = ["Normal", "Diabetes", "Glaucoma", "Cataract",
               "Age related Macular Degeneration", "Hypertension",
               "Pathological Myopia", "Other diseases/abnormalities"]

# Load the model once, and ensure proper error handling
try:
    model = tf.keras.models.load_model("eye_disease_model.h5")
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None  # Set model to None if loading failed

# Title and file uploader in Streamlit
st.title("Eye Disease Detection")
uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "png", "jpeg"])

# Check if file is uploaded
if uploaded_file is not None and model is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    if image is None:
        st.error("Error loading the image.")
    else:
        # Preprocess the image
        image_resized = cv2.resize(image, (128, 128))  # Resize to match model input
        image_normalized = image_resized / 255.0  # Normalize the pixel values
        image_input = np.expand_dims(image_normalized, axis=0)  # Expand dimensions for batch size

        # Show the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction using the model
        try:
            prediction = model.predict(image_input)
            predicted_class = class_names[np.argmax(prediction)]  # Get the class with the highest probability
            st.markdown(f"### Prediction: {predicted_class}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    if model is None:
        st.error("The model could not be loaded. Please check the model file.")
