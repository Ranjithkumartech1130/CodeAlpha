import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("digit_model.h5")

st.title("✍️ Handwritten Character Recognition Dashboard")
st.write("Upload a handwritten digit image (0–9)")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=200)

    # Preprocess image
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.subheader(f"🎯 Predicted Digit: {predicted_class}")
