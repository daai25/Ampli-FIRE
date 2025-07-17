#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import sys

# --- Define a simple CNN for audio embedding ---
class AudioFingerprintNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, embedding_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.global_pool(x)  # output shape: (batch, 64, 1, 1)
        x = x.view(x.size(0), -1)  # shape: (batch, 64)
        x = self.fc1(x)
        return F.normalize(x, p=2, dim=1)


# --- Helper: load audio and compute mel-spectrogram ---
def audio_to_mel_spectrogram(file_path, sr=22050, n_mels=128, duration=10):
    y, _ = librosa.load(file_path, sr=sr, duration=duration)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    # Normalize to 0-1
    log_mel_spec_norm = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
    return log_mel_spec_norm

# --- Prepare a batch tensor from mel spectrogram ---
def prepare_input(mel_spec):
    tensor = torch.tensor(mel_spec).unsqueeze(0).unsqueeze(0).float()  # (batch, channel, freq, time)
    return tensor

# --- Example usage ---

# Initialize model (random weights here, usually you'd load trained weights)
model = AudioFingerprintNet()

# Create a small "database" of audio files and their embeddings
database_files = os.listdir('./audio')
print("Found " + str(len(database_files)) + " audio files...")
database_embeddings = []

for file in database_files:
    mel = audio_to_mel_spectrogram('./audio/' + file)
    inp = prepare_input(mel)
    with torch.no_grad():
        emb = model(inp)
    database_embeddings.append(emb.numpy())

database_embeddings = np.vstack(database_embeddings)  # shape (N, embedding_dim)

# Given a query audio snippet, find closest match
query_file = sys.argv[1]

mel_query = audio_to_mel_spectrogram(query_file)
inp_query = prepare_input(mel_query)

with torch.no_grad():
    emb_query = model(inp_query).numpy()

# Compute cosine similarity between query and database embeddings
similarities = cosine_similarity(emb_query, database_embeddings)[0]
print(similarities)

best_match_idx = np.argmax(similarities)
print(f'Best match: {database_files[best_match_idx]} with similarity {similarities[best_match_idx]:.4f}')

top_indices = np.argsort(similarities)[::-1][:10]

# Print top 10 matches
print("\nTop 10 Matches:")
for rank, idx in enumerate(top_indices, 1):
    print(f"{rank}. {database_files[idx]} - Similarity: {similarities[idx]:.4f}")
