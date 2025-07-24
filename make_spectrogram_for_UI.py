import os
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image


def process_one_mp3(file):
    # Step 1: Load the audio
    audio = AudioSegment.from_file(file, format="mp3")

    # Step 2: Extract 30s to 60s (in milliseconds)
    start_ms = 30 * 1000
    end_ms = 60 * 1000
    audio_segment = audio[start_ms:end_ms]

    # Step 3: Convert to mono and resample
    audio_segment = audio_segment.set_channels(1).set_frame_rate(22050)

    # Step 4: Convert to NumPy array and normalize
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0

    # Step 5: Generate spectrogram
    D = librosa.stft(samples)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Step 6: Plot to a buffer instead of showing
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=22050, x_axis='time', y_axis='log', cmap='magma', ax=ax)
    ax.set(title="Spectrogram (30s–60s)")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)

    # Step 7: Return as PIL image (you can convert to other formats as needed)
    image = Image.open(buf)
    return image

