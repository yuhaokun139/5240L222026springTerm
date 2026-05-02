import streamlit as st
from PIL import Image
import time

#function part
def main():
        # App title
        st.title("Streamlit Demo on Hugging Face")
        
        # Write some text
        st.write("Welcome to a demo app showcasing basic Streamlit components!")
        
        # File uploader for image and audio
        uploaded_image = st.file_uploader("Upload an image",
                                          type=["jpg", "jpeg", "png"])
        
        # Display image with spinner
        if uploaded_image is not None:
            with st.spinner("Loading image..."):
                time.sleep(1)  # Simulate a delay
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Button interaction
        if st.button("Click Me"):
            st.write("🎉 You clicked the button!")


from transformers import pipeline
from PIL import Image

# Streamlit UI
print("Title: Age Classification using ViT")

# Load the age classification pipeline
# The code below should be placed in the main part of the program
age_classifier = pipeline("image-classification",
                          model="nateraw/vit-age-classifier")

image_name = "middleagedMan.jpg"
image_name = Image.open(image_name).convert("RGB")

# Classify age
age_predictions = age_classifier(image_name)
print(age_predictions)
age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)

# Display results
print("Predicted Age Range:")
print(f"Age range: {age_predictions[0]['label']}")
