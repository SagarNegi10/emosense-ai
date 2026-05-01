# 🧠 EmoSenseAI

EmoSenseAI is a real-time facial emotion recognition system built using TensorFlow (CNN) and Flask.  
It detects human emotions from facial images and serves predictions through a simple web interface.

---

## ✨ Features

- 🧠 Deep Learning-based emotion classification (CNN)
- 📷 Face detection using OpenCV
- 🌐 Web interface using Flask
- 📊 Top-3 emotion predictions with confidence scores
- ⚡ Lightweight and fast inference

### 🎯 Supported Emotions

- Angry 😠  
- Disgust 🤢  
- Fear 😨  
- Happy 😄  
- Neutral 😐  
- Sad 😢  
- Surprise 😲  

---

## 🧱 Tech Stack

| Category         | Technology |
|-----------------|-----------|
| Backend         | Flask |
| ML Framework    | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Frontend        | HTML, CSS, JavaScript |
| Data Handling   | NumPy |
| Visualization   | Matplotlib |

---

## 📂 Project Structure

```bash
EmoSenseAI/
│
├── app.py                  # Flask web application
├── train.py               # Model training script
├── requirements.txt       # Dependencies
├── emotion_model.keras    # Trained model (generated)
├── class_names.json       # Label mapping (generated)
│
├── templates/
│   └── index.html         # Frontend UI
│
├── static/
│   ├── css/
│   └── js/
│
└── dataset/               # Dataset directory (user-provided)
```
---
## 📊 Dataset
Download the dataset from Kaggle:
https://www.kaggle.com/datasets/fahadullaha/facial-emotion-recognition-dataset
Expected Dataset Format:
```bash
dataset/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

---
## ⚙️ Installation & Setup
1. Clone Repository
```bash
git clone https://github.com/your-username/EmoSenseAI.git
cd EmoSenseAI
```
2. Create Virtual Environment
```bash
python -m venv venv
```
Activate it:
Windows
```bash
venv\Scripts\activate
```
Mac/Linux
```bash
source venv/bin/activate
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```
---
🧠 Train the Model
```bash
python train.py
```
### Output Files

- `emotion_model.keras` → Trained CNN model  
- `class_names.json` → Label mapping for inference  

---

## 🌐 Run the Application

```bash
python app.py
```
---
## 📸 How It Works

    1. Upload an image via the web UI  
    2. OpenCV detects the face  
    3. Image is preprocessed (grayscale, resized)  
    4. CNN model predicts emotion  

### Results Displayed

- Predicted emotion  
- Confidence score  
- Top 3 predictions  

---

## ⚠️ Important Notes

- Ensure `emotion_model.keras` exists before running `app.py`  
- `class_names.json` must match training labels  
- Only images with clear faces will produce results  
- If no face is detected → `"No face detected"`  

---

## 🔧 Future Improvements

- Real-time webcam emotion detection  
- Emotion-based AI recommendations  
- Mobile-friendly UI  
- Deployment (AWS / Render / Vercel backend)  
- Model accuracy improvements with augmentation  

---

## 🤝 Contributing

    1. Fork the repository  
    2. Create a feature branch  
    3. Commit your changes  
    4. Open a Pull Request  

---

## 👨‍💻 Author

**Sagar Negi**  
MCA Student | Full-Stack Developer | AI Enthusiast  

