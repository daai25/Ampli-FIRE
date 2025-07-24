#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity


# Structure of our CNN:
class AudioFingerprintNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
	# First layer is 2D image.  1 channel (grayscale values).  3x3 filter. 1px border.
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # Professor B. talked about pooling layers.  We use one here:
        self.pool = nn.MaxPool2d(2, 2)
        # Now, feed that into another 2D layer and search for features:
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # Pooling again.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Output embedding.
        self.fc1 = nn.Linear(64, embedding_dim)

    def forward(self, x):
        # Using RELU as activation functions on both conv layers.
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalize

# Make a MEL Spectrogram.  MELs are better for how humans hear,
# as our ears amplify certain frequency ranges. Using 30 seconds of audio.
def audio_to_mel_spectrogram(file_path, sr=22050, n_mels=128, duration=30):
    try:
	# Load file
        y, _ = librosa.load(file_path, sr=sr, duration=duration)
	# Make sure each sample is the same size
        y = librosa.util.fix_length(y, size=sr*duration)
	# Create a spectrogram.  Math is normalizing values.
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel_spec_norm = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-9)
        return log_mel_spec_norm
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# We have a spectrogram.  We need to convert to form we want (usually a vector or matrix).
def prepare_input(mel_spec):
    # You can print this out if you want to see what it looks like.
    return torch.tensor(mel_spec).unsqueeze(0).unsqueeze(0).float()

# I put all files in ./audio/GENRE.  We aren't really using the genre info,
# but we could add it in.  This mostly helps verify it is working.  We usually
# wouldn't expect piano music recommenders to include a song from the metal
# directory, for instance.
def list_all_files_recursive(root_dir):
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            file_paths.append(full_path)
    return file_paths

# --- Main block ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <query_audio_file>")
        sys.exit(1)

    # We want to find recomendations of songs like this one:
    query_file = sys.argv[1]

    # Create an instance of our model
    model = AudioFingerprintNet()
    model.eval()

    audio_dir = './audio'
    supported_formats = ('.wav', '.mp3', '.flac', '.ogg')
    database_files = [f for f in list_all_files_recursive(audio_dir) if f.endswith(supported_formats)]

    if not database_files:
        print("No audio files found in './audio'.")
        sys.exit(1)

    print(f"Found {len(database_files)} audio files...")
    database_embeddings = []
    valid_files = []

    # Let's process all the files we found.
    for file in database_files:
        # Using 256 mels.  Try to vary the value and see how it affects
        # the output.
        mel = audio_to_mel_spectrogram(file, n_mels=256)
        if mel is None:
            continue
        inp = prepare_input(mel)
        with torch.no_grad():
            # The output from our model will be an embedding that represents
            # each song.
            emb = model(inp).squeeze().numpy()
            # Print if you want to see them.  But beware, this will print each
            # one.
            #print(emb)
        # Add the embedding to our DB (right now just a list).
        database_embeddings.append(emb)
        valid_files.append(file)

    if not database_embeddings:
        print("No valid audio embeddings generated.")
        sys.exit(1)

    # This is our "database" (just a list).
    database_embeddings = np.stack(database_embeddings)  # (N, embedding_dim)

    # Query embeddings to find files like our given file:
    mel_query = audio_to_mel_spectrogram(query_file)
    if mel_query is None:
        print(f"Query file '{query_file}' could not be processed.")
        sys.exit(1)

    # Create an embedding for our input file:
    inp_query = prepare_input(mel_query)
    with torch.no_grad():
        emb_query = model(inp_query).squeeze().numpy()

    # Compute the cosine similarity between the input file
    # and all the other embeddings.  The most similar *should* be
    # similar.
    similarities = cosine_similarity([emb_query], database_embeddings)[0]

    # Top match
    best_match_idx = np.argmax(similarities)
    print(f"\nBest match: {valid_files[best_match_idx]} (Similarity: {similarities[best_match_idx]:.4f})")

    # Top 10 matches
    top_indices = np.argsort(similarities)[::-1][:10]
    print("\nTop 10 Matches:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"{rank}. {valid_files[idx]} - Similarity: {similarities[idx]:.4f}")

