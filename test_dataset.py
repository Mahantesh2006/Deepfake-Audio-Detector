from src.dataset import load_dataset
import numpy as np


X, y = load_dataset(
    "data/raw/real-vs-fake-human-voice-deepfake-audio"
)


print("\nFeature shape:", X.shape)
print("Labels shape:", y.shape)


print("\nReal samples:", np.sum(y==0))
print("Fake samples:", np.sum(y==1))