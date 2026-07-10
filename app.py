import gradio as gr
import torch

from src.preprocess import extract_mfcc
from src.model import DeepfakeAudioCNN

# -----------------------
# Load Model
# -----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeepfakeAudioCNN()
model.load_state_dict(
    torch.load(
        "models/deepfake_audio_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()


# -----------------------
# Prediction Function
# -----------------------

def predict_audio(audio_path):

    if audio_path is None:
        return "Please upload an audio file.", ""

    try:

        # Extract MFCC
        mfcc = extract_mfcc(audio_path)

        feature = torch.tensor(
            mfcc,
            dtype=torch.float32
        )

        # (batch, channel, features)
        feature = feature.unsqueeze(0)
        feature = feature.unsqueeze(0)

        feature = feature.to(device)

        with torch.no_grad():

            output = model(feature)

            probability = torch.softmax(
                output,
                dim=1
            )

            confidence, predicted = torch.max(
                probability,
                1
            )

        confidence = confidence.item() * 100
        predicted = predicted.item()

        # IMPORTANT:
        # Same mapping as predict.py

        if predicted == 0:

            result = "🟢 REAL VOICE"
            ai_probability = 100 - confidence

        else:

            result = "🔴 FAKE VOICE"
            ai_probability = confidence

        analysis = f"""
Model Confidence : {confidence:.2f} %

AI Voice Probability : {ai_probability:.2f} %
"""

        return result, analysis

    except Exception as e:

        return "❌ Error", str(e)


# -----------------------
# UI
# -----------------------

with gr.Blocks(title="Deepfake Audio Detector") as demo:

    gr.Markdown(
        """
# 🎙️ Deepfake Audio Detector

Upload a voice sample and the AI model will determine whether it is:

🟢 **Real Human Voice**

or

🔴 **AI Generated Voice**
"""
    )

    with gr.Row():

        audio = gr.Audio(
            type="filepath",
            label="🎤 Upload Audio"
        )

        with gr.Column():

            prediction = gr.Textbox(
                label="Prediction"
            )

            analysis = gr.Textbox(
                label="Analysis",
                lines=4
            )

    button = gr.Button(
        "🚀 Analyze Audio",
        variant="primary"
    )

    button.click(
        predict_audio,
        inputs=audio,
        outputs=[
            prediction,
            analysis
        ]
    )

    gr.Examples(
        examples=[
            ["data/test_audio/sample.wav"]
        ],
        inputs=audio
    )


demo.launch()