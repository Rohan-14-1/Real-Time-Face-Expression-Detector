<div align="center">

# 🎭 Real-Time Face Expression Detector

**AI-powered facial emotion recognition using deep learning and live webcam analysis**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

A full-stack application that detects and classifies human facial expressions in real time through webcam input. The system captures video frames from the browser, sends them to a Flask backend for inference via a custom-trained CNN model, and displays live emotion predictions with confidence scores.

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Model Details](#-model-details) · [API Reference](#-api-reference) · [Contributing](#-contributing)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎥 **Live Webcam Feed** | Real-time video capture directly in the browser via WebRTC |
| 🧠 **CNN Emotion Model** | Custom-trained Convolutional Neural Network with BatchNormalization & Dropout |
| ⚡ **Fast Inference** | Frame-by-frame prediction at ~1.25 FPS (800 ms interval) |
| 📊 **Confidence Meter** | Animated progress bar showing model confidence per prediction |
| 📝 **Prediction Log** | Rolling log of recent predictions for session tracking |
| 🌙 **Dark UI** | Sleek, modern dark-themed interface |
| 🔌 **REST API** | Clean JSON API endpoint for external integrations |

### 🏷️ Supported Emotions

```
😠 Angry  ·  🤢 Disgust  ·  😨 Fear  ·  😊 Happy  ·  😢 Sad  ·  😲 Surprise  ·  😐 Neutral
```

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        BROWSER (Frontend)                        │
│                                                                  │
│   ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐   │
│   │  Webcam Feed │───▶│ Canvas Frame │───▶│  POST /api/predict│   │
│   │  (WebRTC)   │    │  Capture     │    │  -expression      │   │
│   └─────────────┘    └──────────────┘    └────────┬──────────┘   │
│                                                   │              │
│   ┌──────────────────────────────────────────────┐│              │
│   │  📊 Expression Label + Confidence Bar + Log  ││              │
│   └──────────────────────────────────────────────┘│              │
└───────────────────────────────────────────────────┼──────────────┘
                                                    │ HTTP (JPEG Blob)
┌───────────────────────────────────────────────────┼──────────────┐
│                     FLASK SERVER (Backend)         │              │
│                                                    ▼              │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│   │  Receive  │───▶│  Grayscale   │───▶│  CNN Model Predict   │   │
│   │  Image    │    │  48×48 Resize│    │  (emotion_model.hdf5)│   │
│   └──────────┘    └──────────────┘    └──────────┬───────────┘   │
│                                                   │              │
│                                    { expression, confidence }    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Real-Time-Face-Expression-Detector/
│
├── backend/
│   ├── app.py                 # Flask server & prediction API
│   ├── model.py               # CNN model architecture & training script
│   ├── emotion_model.hdf5     # Pre-trained model weights (~7.2 MB)
│   └── requirements.txt       # Python dependencies
│
├── frontend/
│   ├── index.html             # Main UI layout
│   ├── script.js              # Webcam capture & API integration
│   └── style.css              # Dark-themed responsive styles
│
├── dataset/
│   ├── train/                 # Training images (7 emotion classes)
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── test/                  # Test/validation images (same structure)
│
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.8 or higher
- **pip** (Python package manager)
- A modern web browser with webcam support (Chrome, Firefox, Edge)

### 1. Clone the Repository

```bash
git clone https://github.com/Rohan-14-1/Real-Time-Face-Expression-Detector.git
cd Real-Time-Face-Expression-Detector
```

### 2. Create a Virtual Environment *(recommended)*

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Launch the Server

```bash
python backend/app.py
```

The server starts at **`http://localhost:8000`**. Open this URL in your browser, allow camera access, and click **Start Camera** to begin real-time expression detection.

---

## ▶️ How to Run

### Running the Application

**Step 1 — Start the Flask backend:**

```bash
# Make sure your virtual environment is activated
python backend/app.py
```

You should see output similar to:

```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8000
 * Running on http://<your-ip>:8000
```

**Step 2 — Open the app in your browser:**

Navigate to **http://localhost:8000** in Chrome, Firefox, or Edge.

**Step 3 — Start detecting expressions:**

1. Click the **"Start Camera"** button
2. Allow camera/webcam access when prompted by the browser
3. Face the camera — the model will begin predicting your expression in real time
4. Watch the **Expression Label**, **Confidence Bar**, and **Prediction Log** update live
5. Click **"Stop"** to end the session

> [!TIP]
> Ensure good lighting and face the camera directly for the best prediction accuracy.

### Re-training the Model

If you want to train the CNN from scratch with your own dataset:

```bash
python backend/model.py
```

> [!IMPORTANT]
> The `dataset/train/` and `dataset/test/` directories must each contain 7 subdirectories named:
> `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise` — each filled with their respective face images.

The trained model will be saved as `backend/emotion_model.hdf5`, automatically replacing the existing weights.

### Troubleshooting

| Issue | Solution |
|-------|----------|
| **Camera not starting** | Ensure browser has camera permissions enabled. Check `chrome://settings/content/camera` |
| **"Network error: Failed to fetch"** | Verify the Flask server is running on port `8000`. Check terminal for errors. |
| **Black/blank video feed** | Another application may be using the camera. Close other video apps and retry. |
| **Low prediction accuracy** | Ensure your face is well-lit, centered, and unobstructed. |
| **Module not found errors** | Re-run `pip install -r backend/requirements.txt` inside your virtual environment. |
| **Port already in use** | Change the port in `backend/app.py` (line 79) or kill the process using port 8000. |

---

## 🧠 Model Details

### Architecture

The emotion classifier is a **Sequential CNN** designed for efficient inference on 48×48 grayscale face images:

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Conv2D (32 filters, 3×3) + ReLU | 46×46×32 | 320 |
| BatchNormalization | 46×46×32 | 128 |
| MaxPooling2D (2×2) | 23×23×32 | 0 |
| Dropout (0.25) | 23×23×32 | 0 |
| Conv2D (64 filters, 3×3) + ReLU | 21×21×64 | 18,496 |
| BatchNormalization | 21×21×64 | 256 |
| MaxPooling2D (2×2) | 10×10×64 | 0 |
| Dropout (0.25) | 10×10×64 | 0 |
| Conv2D (128 filters, 3×3) + ReLU | 8×8×128 | 73,856 |
| BatchNormalization | 8×8×128 | 512 |
| MaxPooling2D (2×2) | 4×4×128 | 0 |
| Dropout (0.25) | 4×4×128 | 0 |
| Flatten | 2048 | 0 |
| Dense (256) + ReLU | 256 | 524,544 |
| BatchNormalization | 256 | 1,024 |
| Dropout (0.5) | 256 | 0 |
| Dense (7) + Softmax | 7 | 1,799 |

> **Total Parameters:** ~620K · **Model Size:** ~7.2 MB

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Input Size | 48 × 48 × 1 (grayscale) |
| Optimizer | Adam |
| Loss Function | Categorical Cross-Entropy |
| Batch Size | 64 |
| Epochs | 25 |
| Data Augmentation | Rotation (±10°), Zoom (±10%), Horizontal Flip |

### Re-training the Model

To train the model from scratch using your own dataset:

```bash
python backend/model.py
```

> Ensure the `dataset/train/` and `dataset/test/` directories contain subdirectories for each of the 7 emotion classes with their respective images.

---

## 📡 API Reference

### `POST /api/predict-expression`

Accepts a face image and returns the predicted emotion with confidence score.

**Request:**

```
Content-Type: multipart/form-data
Body: file=<image_file.jpg>
```

**Response (200 OK):**

```json
{
  "expression": "Happy",
  "confidence": 0.9423
}
```

**Error Response (400 / 500):**

```json
{
  "error": "No file"
}
```

#### Quick Test with cURL

```bash
curl -X POST http://localhost:8000/api/predict-expression \
  -F "file=@test_face.jpg"
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | HTML5, CSS3, JavaScript (ES6+) | UI, webcam capture, API calls |
| **Backend** | Flask, Flask-CORS | REST API server |
| **ML/DL** | TensorFlow / Keras | CNN model training & inference |
| **Computer Vision** | OpenCV, Pillow | Image preprocessing |
| **Data Format** | NumPy, HDF5 | Array operations & model storage |

</div>

---

## 🔧 Configuration

| Variable | Location | Default | Description |
|----------|----------|---------|-------------|
| `BACKEND_URL` | `frontend/script.js` | `http://localhost:8000/api/predict-expression` | API endpoint URL |
| `host` | `backend/app.py` | `0.0.0.0` | Server bind address |
| `port` | `backend/app.py` | `8000` | Server port |
| Capture interval | `frontend/script.js` | `800` ms | Time between frame captures |

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Ideas for Improvement

- [ ] Add face detection bounding boxes (Haar Cascade / MTCNN)
- [ ] Support multi-face detection in a single frame
- [ ] Display emotion probability distribution chart
- [ ] Add model accuracy metrics dashboard
- [ ] Implement WebSocket for faster streaming
- [ ] Dockerize the application
- [ ] Deploy to cloud (AWS / GCP / Heroku)

---

<div align="center">

**Built with using TensorFlow & Flask**

⭐ Star this repo if you found it helpful!

</div>
