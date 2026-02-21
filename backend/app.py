# ============================================================
# app.py – Real Emotion Prediction Backend
# ============================================================

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
from PIL import Image
import os
from tensorflow.keras.models import load_model

app = Flask(
    __name__,
    static_folder="../frontend",
    static_url_path=""
)
CORS(app)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "emotion_model.hdf5")
model = load_model(MODEL_PATH)

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# -----------------------------
# Health Check
# -----------------------------
@app.route("/")
def serve_frontend():
    return send_from_directory("../frontend", "index.html")

# -----------------------------
# Prediction API
# -----------------------------
@app.route("/api/predict-expression", methods=["POST"])
def predict_expression():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file"}), 400

        file = request.files["file"]

        # Read image
        image = Image.open(file.stream).convert("RGB")
        image = np.array(image)

        # Preprocess
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48))
        gray = gray / 255.0
        gray = gray.reshape(1, 48, 48, 1)

        # Predict
        preds = model.predict(gray)
        confidence = float(np.max(preds))
        expression = EMOTIONS[np.argmax(preds)]

        return jsonify({
            "expression": expression,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)