import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os

# --- CONFIG ---
model_file = "mobilenetv3_genre_classifier_script.pt"
test_dir = os.path.expanduser('~/Switzerland/project/Ampli-FIRE/spectrograms_64_split/test')
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

# --- PREDICT FUNCTION ---
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_idx = torch.argmax(probs).item()
        top_prob = probs[top_idx].item()
        return parent_genres[top_idx], top_prob

# --- STREAMLIT UI ---
st.title("🎵 Spectrogram Genre Classifier")
st.write("Upload or select a spectrogram image to predict its genre.")

# Option 1: Upload image
uploaded_file = st.file_uploader("Upload Spectrogram (.png)", type=["png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Spectrogram", use_column_width=True)
    genre, confidence = predict_image(uploaded_file)
    st.success(f"🎶 Predicted Genre: **{genre}** ({confidence*100:.2f}% confidence)")

st.markdown("---")

# Option 2: Select from test folder
st.subheader("Or select from test folder")
test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".png")]
selected_file = st.selectbox("Select spectrogram:", [""] + test_files)

if selected_file:
    img_path = os.path.join(test_dir, selected_file)
    img = Image.open(img_path).convert("RGB")
    st.image(img, caption=selected_file, use_column_width=True)
    genre, confidence = predict_image(img_path)
    st.success(f"🎶 Predicted Genre: **{genre}** ({confidence*100:.2f}% confidence)")

st.markdown("---")

# Option 3: Predict all
if st.button("🎯 Predict all spectrograms in test folder"):
    st.info("Running predictions for all test images...")
    results = []
    for f in test_files:
        img_path = os.path.join(test_dir, f)
        genre, confidence = predict_image(img_path)
        results.append((f, genre, confidence))
    for f, genre, confidence in results:
        st.write(f"🖼 **{f}** → 🎵 {genre} ({confidence*100:.2f}%)")
