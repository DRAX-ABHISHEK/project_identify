import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import sys
import os

print("Python Path:", sys.path)

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.genai import generate_description

# Load the trained model
model = torch.load('models/resnet_model.pth')
model.eval()

# Define class names (example for Animals-10 dataset)
class_names = ["dog", "cat", "horse", "spider", "butterfly", "chicken", "cow", "sheep", "elephant", "squirrel"]

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("AI Wildlife Identifier")
uploaded_file = st.file_uploader("Upload an image of an animal", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image and predict
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        species = class_names[predicted.item()]

    st.write(f"Predicted Species: {species}")

    # Generate description
    description = generate_description(species)
    st.write(f"Description: {description}")