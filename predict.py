import torch
import numpy as np
import sys

from src.preprocess import extract_mfcc
from src.model import DeepfakeAudioCNN



def predict(audio_path):


    # Load model

    model = DeepfakeAudioCNN()

    model.load_state_dict(
        torch.load(
            "models/deepfake_audio_model.pth"
        )
    )


    model.eval()



    # Extract MFCC

    mfcc = extract_mfcc(
        audio_path
    )


    # Convert to tensor

    feature = torch.tensor(
        mfcc,
        dtype=torch.float32
    )


    # CNN input:
    # (batch, channel, features)

    feature = feature.unsqueeze(0)
    feature = feature.unsqueeze(0)



    # Prediction

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



    if predicted.item() == 0:

        result = "REAL VOICE"

    else:

        result = "FAKE VOICE"



    print("\nPrediction:", result)

    print(
        "Confidence:",
        round(confidence.item()*100,2),
        "%"
    )




if __name__ == "__main__":


    if len(sys.argv) < 2:

        print(
            "Usage: python predict.py audio_file"
        )

    else:

        predict(
            sys.argv[1]
        )