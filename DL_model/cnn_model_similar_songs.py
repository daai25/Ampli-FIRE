import os
import glob
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIG ---
data_dir = "/mnt/c/zhaw/Ampli-FIRE/spectrograms_32"  # <-- Change to your dataset path
img_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# --- LOAD MODEL (ResNet18 without classifier head) ---
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model = nn.Sequential(*list(model.children())[:-1])  # Remove final FC layer
model = model.to(device)
model.eval()

# --- FUNCTION: Get feature embedding ---
def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img)
    return embedding.squeeze().cpu().numpy().flatten()  # Flatten (512, 1, 1) → (512,)

# --- LOAD ALL IMAGES ---
image_paths = glob.glob(os.path.join(data_dir, "*.png"))
if len(image_paths) == 0:
    raise FileNotFoundError(f"❌ No PNG files found in {data_dir}")

print(f"✅ Found {len(image_paths)} spectrograms.")

# --- COMPUTE EMBEDDINGS ---
print("🔄 Computing embeddings...")
embeddings = np.array([get_embedding(p) for p in image_paths])

# --- CHOOSE QUERY SPECTROGRAM ---
query_filename = input("🎵 Enter spectrogram filename (e.g., SongA_spectrogram.png): ").strip()
query_path = os.path.join(data_dir, query_filename)
if not os.path.exists(query_path):
    raise FileNotFoundError(f"❌ Spectrogram '{query_filename}' not found in {data_dir}")

query_embedding = get_embedding(query_path).reshape(1, -1)

# --- FIND VISUALLY SIMILAR SONGS ---
cos_sim = cosine_similarity(query_embedding, embeddings).flatten()
visually_similar_indices = cos_sim.argsort()[::-1][1:5]  # Skip self at [0]
visually_similar_paths = [image_paths[i] for i in visually_similar_indices]
visually_similar_scores = [cos_sim[i] for i in visually_similar_indices]

# --- SAVE RESULTS ---
def save_similar_songs(query, visual, scores, out_dir="similar_songs_results"):
    os.makedirs(out_dir, exist_ok=True)

    # Save the query spectrogram
    shutil.copy(query, os.path.join(out_dir, "query.png"))

    # Save visually similar spectrograms
    for idx, img_path in enumerate(visual, start=1):
        dst_path = os.path.join(out_dir, f"visual_{idx}.png")
        shutil.copy(img_path, dst_path)

    # Save list of similar songs to a text file
    txt_path = os.path.join(out_dir, "similar_songs.txt")
    with open(txt_path, "w") as f:
        f.write("Query Song:\n")
        f.write(f"{os.path.basename(query)}\n\n")
        f.write("Top 4 Visually Similar Songs:\n")
        for idx, (path, score) in enumerate(zip(visual, scores), start=1):
            f.write(f"{idx}. {os.path.basename(path)} (Similarity: {score:.4f})\n")
    print(f"📁 Saved all results in '{out_dir}'")

# --- CREATE COLLAGE ---
def save_collage(query, visual, out_file="similar_songs_results/collage.png"):
    images = [Image.open(query)] + [Image.open(p) for p in visual]
    titles = ["Query"] + [f"Visual {i}" for i in range(1, 5)]

    plt.figure(figsize=(20, 4))
    for i, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(1, 5, i)
        plt.imshow(img)
        plt.title(title, fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"📸 Saved collage to '{out_file}'")

# --- RUN ---
output_folder = "similar_songs_results"
save_similar_songs(query_path, visually_similar_paths, visually_similar_scores, out_dir=output_folder)
save_collage(query_path, visually_similar_paths, out_file=os.path.join(output_folder, "collage.png"))
