#!/usr/bin/env python3

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def generate_spectrogram(wav_path, start_time_sec=60, end_time_sec=120, output_image='spectrogram.png'):
    # Load only the desired portion of the audio
    duration = end_time_sec - start_time_sec
    y, sr = librosa.load(wav_path, offset=start_time_sec, duration=duration)

    # Create Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram from {start_time_sec}s to {end_time_sec}s')
    plt.tight_layout()
    plt.savefig("./images/" + os.path.basename(wav_path) + ".png")
    plt.show()


# Example usage
generate_spectrogram("./audio/" + sys.argv[1])
