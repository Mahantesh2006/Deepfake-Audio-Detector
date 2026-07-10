from src.preprocess import extract_mfcc

audio_path = "data/test_audio/sample.wav"

features = extract_mfcc(audio_path)

print("MFCC Shape:", features.shape)
print(features)