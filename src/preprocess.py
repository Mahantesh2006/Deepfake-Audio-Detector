import librosa
import numpy as np


def load_audio(file_path, sample_rate=22050):
    audio, sr = librosa.load(
        file_path,
        sr=sample_rate,
        mono=True
    )

    return audio, sr


def extract_mfcc(file_path, n_mfcc=40):
    audio, sr = load_audio(file_path)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc
    )

    # Average over time axis
    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc