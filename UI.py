import streamlit as st
from genre_rec_model import recommend_by_genre
import torch
from torchvision import transforms
from PIL import Image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
import io
import os

# Configuring Page
st.set_page_config(page_title="Ampli-FIRE", layout="centered")

model_file = "mobilenetv3_genre_classifier_script.pt"
img_size = 224
genre_options = ["Rock", "Pop", "Hip-Hop & Rap", "Electronic", "R&B & Soul", "Jazz", "Classical",
                 "Country & Folk", "Latin", "Metal", "Punk & Hardcore", "Reggae & Ska",
                 "World & International", "Blues", "Other"]

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


@st.cache_resource
def load_model():
    model = torch.jit.load(model_file, map_location="cpu")
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def predict_genre(spectrogram_path):
    img = Image.open(spectrogram_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_idx = torch.argmax(probs).item()
        confidence = probs[top_idx].item()
    return genre_options[top_idx], confidence


coll1, coll2, coll3 = st.columns([3, 2, 3])
with coll2:
    st.image("Amplifire_logo.png", width=150)

if 'step' not in st.session_state:
    st.session_state.step = 1
if 'count' not in st.session_state:
    st.session_state.count = 0
if 'show_recommend' not in st.session_state:
    st.session_state.show_recommend = False

# Background as black
st.markdown("""
    <style>
    body {
        background-color: black;
        color: white;
    }
    .stApp {
        background-color: black;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #222;
        color: white;
    }
    .stFileUploader>div>div {
        color: white;
    }
    .or-divider {
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
        margin: 10px 0;
    }
    /* Make button text black */
    div.stButton > button {
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

# Upload section
st.title("Upload or Enter Text")

uploaded_file = st.file_uploader("Upload a file", type=["mp3"], disabled=st.session_state.count is not 0)

st.markdown('<div class="or-divider">OR</div>', unsafe_allow_html=True)

# Text boxes
# Disables text box if user uploads a file.

song_name = st.text_input("Song Name:", disabled=uploaded_file is not None)
if song_name:
    st.session_state.count = 1

col1, col2 = st.columns([1, 1])
with col1:
    artist_name = st.text_input("Artist Name:", disabled=uploaded_file is not None)
with col2:
    initial_genre = st.selectbox("Genre:",genre_options, disabled=uploaded_file is not None)


model_file = "mobilenetv3_genre_classifier_script.pt"


def load_model():
    model = torch.jit.load(model_file, map_location="cpu")
    model.eval()
    return model


if st.session_state.step < 3:
    if st.button("Next"):
        st.session_state.show_recommend = True
        st.write("# Recommended Songs: ")
        if uploaded_file is not None:
            # Spectrogram
            spec = process_one_mp3(uploaded_file)
            predicted_genre, confidence = predict_genre(spec)
            st.write(recommend_by_genre(predicted_genre))
        if uploaded_file is None:
            st.write(recommend_by_genre(initial_genre))

if st.session_state.show_recommend:
    if st.button("Recommend Again"):
        st.write("# Recommended Songs: ")
        st.write(recommend_by_genre(initial_genre))
