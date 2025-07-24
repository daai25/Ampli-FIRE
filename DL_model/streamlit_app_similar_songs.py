import streamlit as st
import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models import resnet18, ResNet18_Weights

# --- CONFIG ---
DATA_DIR = "/mnt/c/zhaw/Ampli-FIRE/spectrograms_64"  # Folder with your spectrogram PNGs
MODEL_PATH = "/mnt/c/zhaw/Ampli-FIRE/DL_model/resnet18_feature_extractor.pt"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        # Rebuild model architecture
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model = nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
        # Load weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        st.success("✅ Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None


# --- IMAGE TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# --- LOAD MODEL INSTANCE ---
model = load_model()

# --- FUNCTION: Get feature embedding ---
def get_embedding(pil_image):
    img = transform(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model(img)
    return embedding.squeeze().cpu().numpy().flatten()

# --- LOAD DATASET ---
@st.cache_data
def load_dataset_embeddings():
    image_paths = glob.glob(os.path.join(DATA_DIR, "*.png"))
    if len(image_paths) == 0:
        st.error(f"❌ No PNG files found in '{DATA_DIR}'")
        return [], np.array([])

    embeddings = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            emb = get_embedding(img)
            embeddings.append(emb)
        except Exception as e:
            st.warning(f"⚠️ Skipping {path}: {e}")
    if len(embeddings) == 0:
        st.error("❌ No valid embeddings found in dataset.")
        return [], np.array([])

    embeddings = np.array(embeddings)
    st.success(f"✅ Loaded {len(embeddings)} spectrogram embeddings.")
    return image_paths, embeddings

image_paths, dataset_embeddings = load_dataset_embeddings()
if len(dataset_embeddings) == 0:
    st.stop()

# --- STREAMLIT UI ---
st.title("🎵 Spectrogram Similarity Finder")
st.write("Upload a spectrogram PNG to find visually similar songs from the dataset.")

uploaded_file = st.file_uploader("📂 Upload your spectrogram", type=["png"])

if uploaded_file is not None:
    try:
        # Load and show uploaded image
        query_img = Image.open(uploaded_file).convert("RGB")
        st.image(query_img, caption="🎵 Uploaded Spectrogram", use_column_width=True)

        # Compute query embedding
        query_embedding = get_embedding(query_img)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Compute cosine similarity
        cos_sim = cosine_similarity(query_embedding, dataset_embeddings).flatten()
        top_indices = cos_sim.argsort()[::-1][:4]
        top_scores = [cos_sim[i] for i in top_indices]
        top_paths = [image_paths[i] for i in top_indices]

        # Show results
        st.subheader("🔝 Top 4 Visually Similar Spectrograms")
        cols = st.columns(4)
        for i, (col, path, score) in enumerate(zip(cols, top_paths, top_scores)):
            img = Image.open(path)
            col.image(img, caption=f"Similarity: {score:.4f}", use_column_width=True)

        # Collage
        st.subheader("📸 Collage of Results")
        collage_width = 800
        collage_img = Image.new("RGB", (collage_width, collage_width // 5), color=(255, 255, 255))
        images = [query_img] + [Image.open(p) for p in top_paths]
        for idx, img in enumerate(images):
            img = img.resize((collage_width // 5, collage_width // 5))
            collage_img.paste(img, (idx * collage_width // 5, 0))
        st.image(collage_img, caption="Query + Top 4 Similar", use_column_width=True)

    except Exception as e:
        st.error(f"❌ Failed to process uploaded image: {e}")
