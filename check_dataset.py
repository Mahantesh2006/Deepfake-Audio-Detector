import pandas as pd

file_path = "data/raw/real-vs-fake-human-voice-deepfake-audio/dataset_voice_changing_metainfo.xlsx"

df = pd.read_excel(file_path)

print(df.head())
print("\nColumns:")
print(df.columns.tolist())
print("\nShape:", df.shape)