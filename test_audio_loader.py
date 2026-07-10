import librosa

audio_path = "data/raw/real-vs-fake-human-voice-deepfake-audio/UK/female/1/original.m4a"

audio, sr = librosa.load(audio_path, sr=22050)

print("Sample Rate:", sr)
print("Audio Length:", len(audio))