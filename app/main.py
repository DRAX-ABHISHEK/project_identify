from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms
from utils.genai import generate_description
import io

app = FastAPI()

# Load the trained model
model = torch.load('models/resnet_model.pth')
model.eval()

# Define class names
class_names = ["dog", "cat", "horse", "spider", "butterfly", "chicken", "cow", "sheep", "elephant", "squirrel"]

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        species = class_names[predicted.item()]
    description = generate_description(species)
    return {"species": species, "description": description}