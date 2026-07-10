from src.dataset import AudioDataset

dataset = AudioDataset(
    "data/raw/real-vs-fake-human-voice-deepfake-audio"
)

X, y = dataset.load_dataset()

print("Feature shape:", X.shape)
print("Labels shape:", y.shape)

print("\nFirst feature vector:")
print(X[0])

print("\nFirst label:")
print(y[0])