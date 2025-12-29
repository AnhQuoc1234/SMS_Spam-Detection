import streamlit as st
import tensorflow as tf
import numpy as np

# Load the trained model
# We use caching so the model only loads once, making the app faster
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('load_your_model_here.keras')
    return model

model = load_model()

# Build the App Interface
st.title("SMS Spam Detection Demo")
st.write("Enter an SMS message below to check if it's Spam or Ham.")

# Text input area
user_input = st.text_area("Message Content", placeholder="Type your message here")

# Prediction logic
if st.button("Analyze Message"):
    if not user_input.strip():
        st.warning("Please enter a message first.")
    else:
        # The model expects a list/batch of strings
        prediction_prob = model.predict([user_input])[0][0]
        
        # Display results
        # Threshold is usually 0.5 for sigmoid output
        if prediction_prob > 0.5:
            st.error(f" SPAM DETECTED ({prediction_prob:.2%} confidence)")
        else:
            st.success(f"LEGITIMATE MESSAGE (Ham) ({1 - prediction_prob:.2%} confidence)")
            
        st.info(f"Raw Model Score: {prediction_prob:.4f}")