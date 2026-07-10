import os
import numpy as np

from src.preprocess import extract_mfcc


def load_dataset(data_folder):

    X = []
    y = []

    real_count = 0
    fake_count = 0

    for root, dirs, files in os.walk(data_folder):

        for audio_name in files:

            # Supported audio formats
            if not audio_name.lower().endswith(
                (".wav", ".mp3", ".flac", ".m4a")
            ):
                continue

            audio_path = os.path.join(
                root,
                audio_name
            )

            name = audio_name.lower()


            # -----------------------------
            # Label assignment
            # REAL = 0
            # FAKE = 1
            # -----------------------------

            if (
                "original" in name
                or "real" in name
                or "human" in name
            ):

                label = 0
                real_count += 1


            elif (
                "synthetic" in name
                or "fake" in name
                or "ai" in name
                or "generated" in name
            ):

                label = 1
                fake_count += 1


            else:

                print(
                    "Skipping unknown label:",
                    audio_path
                )

                continue


            try:

                # Extract MFCC features
                mfcc = extract_mfcc(audio_path)

                X.append(mfcc)
                y.append(label)


                print(
                    f"Loaded: {audio_name} | Label: "
                    f"{'REAL' if label == 0 else 'FAKE'}"
                )


            except Exception as e:

                print(
                    f"Skipping {audio_path}: {e}"
                )


    print("\n========== DATASET SUMMARY ==========")
    print("REAL samples:", real_count)
    print("FAKE samples:", fake_count)
    print("Total samples:", real_count + fake_count)
    print("=====================================\n")


    return np.array(X), np.array(y)