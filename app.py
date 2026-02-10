import os
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from torchvision import transforms
from PIL import Image
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wikipedia  

from model import CNN, AlphabetModel

app = Flask(__name__)

digit_model = CNN()
digit_model.load_state_dict(torch.load("checkpoint/digit.pth", map_location="cpu"))
digit_model.eval()

alpha_model = AlphabetModel(num_classes=26)
alpha_model.load_state_dict(torch.load("checkpoint/letter.pth", map_location="cpu"))
alpha_model.eval()

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
    mode = request.json.get('mode', 'digits')

    image_bytes = base64.b64decode(data.split(",")[1])
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image_tensor = transform(image).unsqueeze(0)

    if mode == 'letters':
        model, num_classes = alpha_model, 26
        labels = [chr(ord('A') + i) for i in range(26)]
    else:
        model, num_classes = digit_model, 10
        labels = [str(i) for i in range(10)]

    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1).numpy()[0]
        pred_idx = int(np.argmax(probs))
        result_text = labels[pred_idx]

    plt.figure(figsize=(10, 4))
    colors = ['#1f77b4'] * num_classes
    colors[pred_idx] = '#2ca02c'
    plt.bar(range(num_classes), probs, color=colors)
    plt.xticks(range(num_classes), labels)
    plt.title(f"Detection: {result_text}")
    plt.ylim(0, 1)
    plt.savefig("static/prob.png")
    plt.close()

    return jsonify({"result": result_text, "prob_img": "prob.png"})

@app.route('/get_info', methods=['POST'])
def get_info():
    word = request.json.get('word', '').strip().lower()
    if not word:
        return jsonify({"info": ""})

    try:
        summary = wikipedia.summary(word, sentences=2, auto_suggest=False)
        return jsonify({"info": f"{word.upper()}: {summary}"})

    except wikipedia.exceptions.DisambiguationError as e:
        return jsonify({"info": f"{word.upper()}: Multiple meanings found. Did you mean: {', '.join(e.options[:3])}?"})

    except wikipedia.exceptions.PageError:
        return jsonify({"info": f"{word.upper()}: No exact Wikipedia page found."})

    except Exception:
        return jsonify({"info": "Error connecting to Wikipedia service."})

if __name__ == '__main__':
    if not os.path.exists('static'): os.makedirs('static')
    app.run(debug=True)