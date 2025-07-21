import os
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Input and output folders
music_folder = 'music'
spectrogram_folder = 'spectrograms'
os.makedirs(spectrogram_folder, exist_ok=True)

def process_mp3(file_path, output_folder):
    print(file_path)
    name = file_path[6:]
    try:
        # Load audio with pydub
        audio = AudioSegment.from_file(file_path)

        # Extract the 30s to 60s segment
        start_ms = 30 * 1000
        end_ms = 60 * 1000
        if len(audio) < end_ms:
            print(f"Skipping {file_path}: shorter than 60 seconds")
            return
        segment = audio[start_ms:end_ms]

        # Export to WAV for librosa
        temp_wav = "temp.wav"
        segment.export(temp_wav, format="wav")

        # Load audio segment with librosa
        y, sr = librosa.load(temp_wav, sr=None)

        # Generate spectrogram
        S = librosa.stft(y)
        S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)

        # Plot and save spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram {name} (30s–60s)')
        plt.tight_layout()

        # Save plot
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_folder, base_name + '_spectrogram.png')
        plt.savefig(output_path)
        plt.close()

        print(f"Saved spectrogram: {output_path}")

        # Clean up temp file
        os.remove(temp_wav)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


# Process all mp3 files in the music folder
for filename in os.listdir(music_folder):
   if filename.lower().endswith('.mp3'):
        file_path = os.path.join(music_folder, filename)
        process_mp3(file_path, spectrogram_folder)
