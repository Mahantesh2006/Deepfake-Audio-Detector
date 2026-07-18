# 🎙️ Deepfake Audio Detector

An AI-powered web application that detects whether an uploaded voice recording is a **real human voice** or an **AI-generated (deepfake) voice** using **MFCC feature extraction**, a **1D CNN (PyTorch)**, and a **Gradio** web interface.

---

## 📌 Features

- 🎤 Upload audio files (.wav, .mp3, .m4a)
- 🧠 Detects Real vs AI-generated voice
- 📊 Displays prediction confidence
- ⚡ Fast inference using a trained CNN model
- 🌐 Interactive Gradio web interface

---

## 🛠️ Tech Stack

- Python 3
- PyTorch
- Librosa
- NumPy
- Gradio

---

## 📂 Project Structure

```
Deepfake-Audio-Detector/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── test_audio/
│
├── models/
│   └── deepfake_audio_model.pth
│
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   └── model.py
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Deepfake-Audio-Detector.git

cd Deepfake-Audio-Detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Web Application

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:7860
```

---

## 🧪 Command Line Prediction

```bash
python predict.py path/to/audio.wav
```

Example:

```bash
python predict.py data/test_audio/sample.wav
```

---

## 🧠 Model Architecture

The model is a **1D Convolutional Neural Network (CNN)**.

Pipeline:

```
Audio File
      │
      ▼
Librosa
      │
      ▼
MFCC Feature Extraction
      │
      ▼
1D CNN (PyTorch)
      │
      ▼
Softmax
      │
      ▼
REAL / FAKE
```

---

## 📊 Sample Output

```
Prediction

🟢 REAL VOICE

Model Confidence : 99.96%

AI Voice Probability : 0.04%
```

---

## 📸 Application Screenshot

Add a screenshot here after uploading it.

Example:

```
images/demo.png
```

Then insert:

```markdown
![App Screenshot](images/demo.png)
```

---

## 📈 Future Improvements

- Support longer audio clips
- Add spectrogram visualization
- Improve accuracy using larger datasets
- Deploy on Hugging Face Spaces
- Add confidence graph
- Support multiple languages


---
