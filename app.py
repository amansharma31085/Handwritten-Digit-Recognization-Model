import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from model import CNN
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import base64
import io
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load model
model = CNN()
model.load_state_dict(torch.load("checkpoint/mnist_cnn.pth", map_location="cpu"))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    # Decode the image from base64
    image_bytes = base64.b64decode(data.split(",")[1])
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        # Apply softmax to get probabilities
        probs = F.softmax(output, dim=1).numpy()[0]
        # Get the predicted digit
        pred = int(np.argmax(probs))

    # Save probability bar chart with highlighting
    plt.figure()
    colors = ['#1f77b4'] * 10  # Default color
    colors[pred] = '#2ca02c'   # Highlight color for the predicted digit
    
    plt.bar(range(10), probs, color=colors)
    plt.title("Prediction Probabilities")
    plt.xlabel("Digit")
    plt.ylabel("Probability")
    plt.xticks(range(10))
    plt.ylim(0, 1) # Ensure y-axis is from 0 to 1
    plt.savefig("static/prob.png")
    plt.close()

    return jsonify({"digit": pred, "prob_img": "prob.png"})

if __name__ == '__main__':
    # Ensure the static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)