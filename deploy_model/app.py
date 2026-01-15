import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image

# --- Load Your Model and Class Names ---
@st.cache_resource
def load_my_model():
    # Load the new, modern .keras file
    # (Assuming your file is named 'resnet50_dryfruits.keras')
    model = load_model('resnet50_dryfruits.keras') 
    return model

class_names = {
    0: 'AlmondGrade_A',
    1: 'AlmondGrade_B',
    2: 'CashewGrade_A',
    3: 'CashewGrade_B',
    4: 'CashewGrade_C',
    5: 'PistachioGrade_A',
    6: 'RaisinGrade_A',
    7: 'RaisinGrade_B',
    8: 'WalnutGrade_A'
}
# --------------------------------------------------------

model = load_my_model()

# --- App Interface ---
st.title("Dry Fruit Quality Grader")
st.write("Upload an image of a dry fruit, and the model will predict its grade.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Preprocess the image
    img = Image.open(uploaded_file).convert('RGB') # Ensure 3 channels
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    
    # 2. Make prediction
    prediction = model.predict(img_preprocessed)
    predicted_index = np.argmax(prediction[0])
    predicted_class_name = class_names[predicted_index]
    confidence = np.max(prediction[0])
    
    # 3. Display results with confidence threshold
    CONFIDENCE_THRESHOLD = 0.85  # Set your threshold (e.g., 90%)
    
    # Always display the uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if confidence < CONFIDENCE_THRESHOLD:
        # If confidence is low, show a "Not Sure" message
        st.markdown(f"## Prediction: Not Sure")
        st.write(f"This doesn't look like a dry fruit from my dataset. Please upload a clearer image.")
    else:
        # If confidence is high, show the prediction
        st.markdown(f"## Prediction: **{predicted_class_name}**")
        st.markdown(f"### Confidence: **{confidence * 100:.2f}%**")