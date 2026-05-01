from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import json

app = Flask(__name__)

IMG_SIZE = 48
MODEL_PATH = Path("emotion_model.keras")
CLASS_NAMES_PATH = Path("class_names.json")

# Load trained model once at startup
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH.resolve()}")
model = tf.keras.models.load_model(str(MODEL_PATH))

# Labels
DEFAULT_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
if CLASS_NAMES_PATH.exists():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        loaded_labels = json.load(f)
    if isinstance(loaded_labels, list) and all(isinstance(x, str) for x in loaded_labels):
        labels = loaded_labels
    else:
        labels = DEFAULT_LABELS
else:
    labels = DEFAULT_LABELS

# Emojis
emoji = {
    "angry": "😠",
    "fear": "😨",
    "happy": "😊",
    "neutral": "😐",
    "surprise": "😲",
    "sad": "😢",
    "disgust": "🤢"
}

# Face detector
face = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)
if face.empty():
    raise RuntimeError("Failed to load OpenCV Haar cascade for face detection.")


def detect_largest_face(gray_image):
    """
    Detect face with a fallback configuration if the first pass fails.
    """
    faces = face.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        faces = face.detectMultiScale(
            gray_image,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(60, 60)
        )

    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def preprocess_face(gray_image, box):
    x, y, w, h = box
    roi = gray_image[y:y + h, x:x + w]
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    roi = roi.astype(np.float32) / 255.0
    return roi.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def top_k_predictions(pred_vector, k=3):
    top_idx = np.argsort(pred_vector)[-k:][::-1]
    return [
        {
            "label": labels[i].title(),
            "emoji": emoji.get(labels[i], "🙂"),
            "confidence": round(float(pred_vector[i]) * 100, 2)
        }
        for i in top_idx
    ]

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "Missing image file"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"success": False, "error": "Empty filename"}), 400

        raw_bytes = file.read()
        if not raw_bytes:
            return jsonify({"success": False, "error": "Uploaded file is empty"}), 400

        img = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"success": False, "error": "Invalid image"}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        box = detect_largest_face(gray)
        if box is None:
            return jsonify({"success": False, "error": "No face detected"}), 200

        x, y, w, h = map(int, box)
        roi = preprocess_face(gray, (x, y, w, h))
        pred = model.predict(roi, verbose=0)[0]
        if pred.shape[0] != len(labels):
            return jsonify({
                "success": False,
                "error": (
                    f"Label mismatch: model outputs {pred.shape[0]} classes "
                    f"but app has {len(labels)} labels. Retrain and regenerate class_names.json."
                )
            }), 500

        pred_idx = int(np.argmax(pred))
        label = labels[pred_idx]
        conf = float(pred[pred_idx])

        return jsonify({
            "success": True,
            "emotion": label.title(),
            "emoji": emoji.get(label, "🙂"),
            "confidence": round(conf * 100, 2),
            "top_predictions": top_k_predictions(pred, k=3),
            "box": {
                "x": x,
                "y": y,
                "w": w,
                "h": h
            }
        })
    except Exception as exc:
        return jsonify({"success": False, "error": f"Prediction failed: {str(exc)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)