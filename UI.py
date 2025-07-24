import uuid
import streamlit as st
from genre_rec_model import recommend_by_genre
import torch
from torchvision import transforms
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
import os
import glob
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Configuring Page
st.set_page_config(page_title="Ampli-FIRE", layout="centered")

genre_model_file = "mobilenetv3_genre_classifier_script.pt"
similar_model_file = "resnet18_feature_extractor_full.pt"
img_size = 224
genre_options = ["Pick a Genre", "Rock", "Pop", "Hip-Hop & Rap", "Electronic", "R&B & Soul", "Jazz", "Classical",
                 "Country & Folk", "Latin", "Metal", "Punk & Hardcore", "Reggae & Ska",
                 "World & International", "Blues", "Other"]
metadata_df = pd.read_csv("labeled_song_artists_grouped.csv")
metadata_df = metadata_df.drop_duplicates(subset="Filename", keep="first")
metadata_dict = metadata_df.set_index("Filename").to_dict(orient="index")

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

    # Create unique filename
    output_dir = "output_spectrograms"
    os.makedirs(output_dir, exist_ok=True)
    image_filename = f"{uuid.uuid4().hex}.png"
    image_path = os.path.join(output_dir, image_filename)

    fig.savefig(image_path, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)

    # Step 7: Return the path
    return image_path


@st.cache_resource
def load_genre_model():
    genre_model = torch.jit.load(genre_model_file, map_location="cpu")
    genre_model.eval()
    return genre_model


@st.cache_resource
def load_similar_model():
    similar_model = torch.load(similar_model_file, map_location="cpu", weights_only=False)
    similar_model.eval()
    return similar_model

genre_model = load_genre_model()
similar_model = load_similar_model()

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


def predict_genre(spectrogram_path):
    img = Image.open(spectrogram_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = genre_model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_idx = torch.argmax(probs).item()
        confidence = probs[top_idx].item()
    return genre_options[top_idx], confidence


coll1, coll2, coll3 = st.columns([3, 2, 3])
with coll2:
    st.image("Amplifire_logo.png", width=150, caption="Amplifire logo.\nA speaker with fire on it.")

if 'step' not in st.session_state:
    st.session_state.step = 1
if 'count' not in st.session_state:
    st.session_state.count = 0
if 'show_recommend' not in st.session_state:
    st.session_state.show_recommend = False
if "show_restart" not in st.session_state:
    st.session_state.show_restart = False
if "reset_form" not in st.session_state:
    st.session_state.reset_form = False

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
    </style>
""", unsafe_allow_html=True)

# Upload section
st.title("Upload or Enter Text")
st.write("Uploading a file creates a more accurate recommendation and allows you to listen to the recommended songs.")
file_key = "file_uploader" if not st.session_state.reset_form else "file_uploader_reset"
song_key = "song"
artist_key = "artist"
genre_key = "genre"
uploaded_file = st.file_uploader("Upload a file", type=["mp3"], key=file_key)

st.markdown('<div class="or-divider">OR</div>', unsafe_allow_html=True)

# Text boxes
# Disables text box if user uploads a file.

song_name = st.text_input("Song Name:", key=song_key, disabled=uploaded_file is not None)
if song_name:
    st.session_state.count = 1

col1, col2 = st.columns([1, 1])
with col1:
    artist_name = st.text_input("Artist Name:", key=artist_key, disabled=uploaded_file is not None)
with col2:
    initial_genre = st.selectbox("Genre:",genre_options, key=genre_key, disabled=uploaded_file is not None)


if st.session_state.step < 3:
    if st.button("Next"):
        st.write("We are finding recommendations for you.")
        st.write("This may take a moment...")
        st.session_state.show_restart = True
        if uploaded_file is not None:
            # Spectrogram
            spec_path = process_one_mp3(uploaded_file)
            predicted_genre, confidence = predict_genre(spec_path)
            # Set path to spectrogram dataset
            dataset_dir = "spectrograms_64"  # adjust path if needed
            image_paths = glob.glob(os.path.join(dataset_dir, "*.png"))


            def get_embedding(img_path):
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    embedding = similar_model(img_tensor)
                return embedding.squeeze().cpu().numpy().flatten()


            # Get embedding for the uploaded spectrogram
            query_embedding = get_embedding(spec_path).reshape(1, -1)

            # Compute embeddings for the dataset
            dataset_embeddings = []
            valid_paths = []

            for path in image_paths:
                try:
                    emb = get_embedding(path)
                    dataset_embeddings.append(emb)
                    valid_paths.append(path)
                except Exception:
                    continue  # Skip unreadable images

            # Compute similarity
            if dataset_embeddings:
                dataset_embeddings = np.array(dataset_embeddings)
                similarities = cosine_similarity(query_embedding, dataset_embeddings).flatten()
                top_indices = similarities.argsort()[::-1][:4]
                top_files = [valid_paths[i] for i in top_indices]

                st.write(f"### Your song genre was {predicted_genre}.")
                st.write("### Here are some other songs you might like:")
                mp3_folder = "music"  # Adjust this if your mp3s are stored elsewhere

                for i, path in enumerate(top_files, 1):
                    filename = os.path.basename(path)
                    song_info = metadata_dict.get(filename, {"Song Title": "Unknown", "Artist": "Unknown"})
                    st.write(
                        f"{i}. **{song_info['Song Title']}** by *{song_info['Artist']}* (Score: {similarities[top_indices[i - 1]]:.4f})"
                    )

                    # Play the corresponding MP3
                    mp3_file = os.path.join(mp3_folder, filename.replace("_spectrogram.png", ".mp3"))

                    if os.path.exists(mp3_file):
                        with open(mp3_file, "rb") as audio_file:
                            st.audio(audio_file.read(), format="audio/mp3")
                    else:
                        st.warning(f"Audio not found for {filename.replace('.png', '')}")
            else:
                st.warning("⚠️ No embeddings found in dataset for similarity search.")
        if uploaded_file is None:
            st.session_state.show_recommend = True
            if (song_name is None) and (artist_name is None):
                st.write("Please indicate a song name and artist.")
            if initial_genre != "Pick a Genre":
                st.write(recommend_by_genre(initial_genre))
            else:
                st.write("Please pick a genre.")
if st.session_state.show_recommend:
    if st.button("Recommend Again"):
        st.write("This may take a moment...")
        if uploaded_file:
            spec_path = process_one_mp3(uploaded_file)
            predicted_genre, confidence = predict_genre(spec_path)
            # Set path to spectrogram dataset
            dataset_dir = "spectrograms_64"  # adjust path if needed
            image_paths = glob.glob(os.path.join(dataset_dir, "*.png"))


            def get_embedding(img_path):
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    embedding = similar_model(img_tensor)
                return embedding.squeeze().cpu().numpy().flatten()


            # Get embedding for the uploaded spectrogram
            query_embedding = get_embedding(spec_path).reshape(1, -1)

            # Compute embeddings for the dataset
            dataset_embeddings = []
            valid_paths = []

            for path in image_paths:
                try:
                    emb = get_embedding(path)
                    dataset_embeddings.append(emb)
                    valid_paths.append(path)
                except Exception:
                    continue  # Skip unreadable images

            # Compute similarity
            if dataset_embeddings:
                dataset_embeddings = np.array(dataset_embeddings)
                similarities = cosine_similarity(query_embedding, dataset_embeddings).flatten()
                top_indices = similarities.argsort()[::-1][:4]
                top_files = [valid_paths[i] for i in top_indices]

                st.write("### Here are some other songs you might like:")
                mp3_folder = "music"  # Adjust this if your mp3s are stored elsewhere

                for i, path in enumerate(top_files, 1):
                    filename = os.path.basename(path)
                    song_info = metadata_dict.get(filename, {"Song Title": "Unknown", "Artist": "Unknown"})
                    st.write(
                        f"{i}. **{song_info['Song Title']}** by *{song_info['Artist']}* (Score: {similarities[top_indices[i - 1]]:.4f})"
                    )

                    # Play the corresponding MP3
                    mp3_file = os.path.join(mp3_folder, filename.replace(".png", ".mp3"))

                    if os.path.exists(mp3_file):
                        with open(mp3_file, "rb") as audio_file:
                            st.audio(audio_file.read(), format="audio/mp3")
                    else:
                        st.warning(f"Audio not found for {filename.replace('.png', '')}")
            else:
                st.warning("⚠️ No embeddings found in dataset for similarity search.")
        if uploaded_file is None:
            if (song_name is None) and (artist_name is None):
                st.write("Please indicate a song name and artist.")
            if initial_genre != "Pick a Genre":
                st.write(recommend_by_genre(initial_genre))
            else:
                st.write("Please pick a genre.")

def reset():
    st.session_state.show_recommend = False
    st.session_state.reset_form = not st.session_state.reset_form
    st.session_state[song_key] = ""
    st.session_state[artist_key] = ""
    st.session_state[genre_key] = genre_options[0]
    st.session_state.show_restart = False

if st.session_state.show_restart:
    st.button("Restart", on_click=reset)
