# ✍️ Handwritten Digit Recognition Model

A web-based application for recognizing handwritten digits (0–9) using a machine learning model. Users can draw a digit on the interface, and the application predicts the number using a trained neural network.

---

## Features

* **Interactive drawing interface**: Draw digits directly in your browser using a canvas-based web UI (`app.py`).
* **Predictive model**: Fast and accurate digit recognition powered by a trained neural network (`model.py`).
* **Flexible training**: Retrain the model using your own dataset or standard datasets such as MNIST with `train.py`.
* **Web deployment ready**: Easily run locally or deploy using Flask or similar frameworks.

---

## Repository Contents

| File / Directory | Description                                                              |
| ---------------- | ------------------------------------------------------------------------ |
| `app.py`         | Flask web application that serves the drawing UI and handles predictions |
| `model.py`       | Neural network architecture and model loading scripts                    |
| `train.py`       | Script to train the model using datasets (e.g., MNIST)                   |
| `templates/`     | HTML templates for the web interface                                     |
| `static/`        | Static assets such as CSS, JavaScript, and images                        |
| `.gitignore`     | Common files to ignore (cache, checkpoints, etc.)                        |
| `README.md`      | Project documentation                                                    |

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/amansharma31085/Handwritten-Digit-Recognization-Model.git
cd Handwritten-Digit-Recognization-Model
```

---

### 2. Install Dependencies

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
```

Activate the virtual environment:

* **Windows**

```bash
venv\Scripts\activate
```

* **Linux / macOS**

```bash
source venv/bin/activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not available, install manually:
> `flask`, `tensorflow` or `keras`, `numpy`, `Pillow`, `opencv-python`

---

## How to Use

### 1. Train the Model

To train or retrain the neural network:

```bash
python train.py
```

This script loads the dataset (e.g., MNIST), trains the model, evaluates performance, and saves the trained model (e.g., `model.h5`).

---

### 2. Launch the Web App

Run the application:

```bash
python app.py
```

Open your browser at:

```
http://127.0.0.1:5000
```

---

### 3. Draw and Recognize

Draw a digit in the web interface and submit it. The backend processes the image, runs the model, and returns the predicted digit (0–9).

---

## Project Workflow

1. **Training** – Use `train.py` to build and save the model
2. **Serving** – `app.py` loads the trained model and hosts the UI
3. **Prediction** – User draws digit → UI captures it → Backend predicts → Result displayed

---

## Requirements

* Python 3.7+
* Flask (or equivalent web framework)
* TensorFlow or Keras
* NumPy
* Pillow
* OpenCV (optional, for image preprocessing)
