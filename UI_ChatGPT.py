import os
import torch
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
import time
from genre_rec_model import recommend_by_genre

# --- CONFIG ---
model_file = "mobilenetv3_genre_classifier_script.pt"
img_size = 224
parent_genres = ["Rock", "Pop", "Hip-Hop & Rap", "Electronic", "R&B & Soul",
                 "Jazz", "Classical", "Country & Folk", "Latin", "Metal",
                 "Punk & Hardcore", "Reggae & Ska", "World & International", "Blues", "Other"]

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = torch.jit.load(model_file, map_location="cpu")
    model.eval()
    return model

model = load_model()

# --- TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# --- AUDIO TO SPECTROGRAM ---
def audio_to_spectrogram(audio_file, output_path="temp_spectrogram.png"):
    y, sr = librosa.load(audio_file, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(5, 5))
    librosa.display.specshow(S_dB, sr=sr, cmap='magma')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return output_path

# --- PREDICT FUNCTION ---
def predict_genre(spectrogram_path):
    img = Image.open(spectrogram_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_idx = torch.argmax(probs).item()
        confidence = probs[top_idx].item()
    return parent_genres[top_idx], confidence

# --- STREAMLIT APP ---
coll1, coll2, coll3 = st.columns([3, 2, 3])
with coll2:
    st.image("Amplifire_logo.png", width=150)

st.set_page_config(page_title="Ampli-FIRE", layout="centered")

if 'step' not in st.session_state:
    st.session_state.step = 1
if 'count' not in st.session_state:
    st.session_state.count = 0
if 'show_recommend' not in st.session_state:
    st.session_state.show_recommend = False

# Styling
st.markdown("""
    <style>
    body {background-color: black; color: white;}
    .stApp {background-color: black; color: white;}
    .stTextInput>div>div>input {background-color: #222; color: white;}
    .stFileUploader>div>div {color: white;}
    .or-divider {text-align: center; font-size: 1.2em; font-weight: bold; margin: 10px 0;}
    div.stButton > button {color: black !important;}
    </style>
""", unsafe_allow_html=True)

# Upload section
st.title("Upload or Enter Text")

uploaded_file = st.file_uploader("Upload a file", type=["mp3"], disabled=st.session_state.count != 0)
st.markdown('<div class="or-divider">OR</div>', unsafe_allow_html=True)

# Text inputs
genre_options = parent_genres
song_name = st.text_input("Song Name:", disabled=uploaded_file is not None)
if song_name:
    st.session_state.count = 1

col1, col2 = st.columns([1, 1])
with col1:
    artist_name = st.text_input("Artist Name:", disabled=uploaded_file is not None)
with col2:
    initial_genre = st.selectbox("Genre:", genre_options, disabled=uploaded_file is not None)

# Next button logic
if st.session_state.step < 3:
    if st.button("Next"):
        st.session_state.show_recommend = True
        st.write("# Recommended Songs:")

        # If MP3 uploaded → predict genre
        if uploaded_file:
            with st.spinner("🎶 Processing audio and predicting genre..."):
                spectrogram_path = audio_to_spectrogram(uploaded_file)
                predicted_genre, confidence = predict_genre(spectrogram_path)
                st.image(spectrogram_path, caption="Generated Spectrogram", use_column_width=True)
                st.success(f"🎵 Predicted Genre: *{predicted_genre}* ({confidence*100:.2f}%)")
                st.write(recommend_by_genre(predicted_genre))
        else:
            # Text input mode
            st.write(recommend_by_genre(initial_genre))

if st.session_state.show_recommend:
    if st.button("Recommend Again"):
        if uploaded_file:
            st.write("# Recommended Songs:")
            st.write(recommend_by_genre(predicted_genre))
        else:
            st.write("# Recommended Songs:")
            st.write(recommend_by_genre(initial_genre))
