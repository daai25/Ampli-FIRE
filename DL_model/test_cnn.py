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

# --- FUNCTION: Extract label from filename ---
def get_label(filename):
    base = os.path.basename(filename).lower()
    if "_" in base:
        return base.split("_")[0]  # e.g., 'dog_bark_001.png' → 'dog'
    else:
        return base.split("-")[0]  # fallback for 'dog-bark-001.png'

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
labels = [get_label(p) for p in image_paths]

# --- COMPUTE EMBEDDINGS ---
print("🔄 Computing embeddings...")
embeddings = np.array([get_embedding(p) for p in image_paths])

# --- CHOOSE QUERY SPECTROGRAM ---
query_filename = input("Enter spectrogram filename (e.g., siren_12.png): ").strip()
query_path = os.path.join(data_dir, query_filename)
if not os.path.exists(query_path):
    raise FileNotFoundError(f"❌ Spectrogram '{query_filename}' not found in {data_dir}")

query_embedding = get_embedding(query_path).reshape(1, -1)
query_label = get_label(query_path)

# --- FIND VISUALLY SIMILAR (ResNet embeddings) ---
cos_sim = cosine_similarity(query_embedding, embeddings).flatten()
visually_similar_indices = cos_sim.argsort()[::-1][1:5]  # Skip self at [0]
visually_similar_paths = [image_paths[i] for i in visually_similar_indices]

# --- FIND SEMANTICALLY SIMILAR (same label) ---
semantically_similar_paths = [p for p, l in zip(image_paths, labels)
                              if l == query_label and p != query_path][:4]

# --- SAVE RESULTS ---
def save_similar_spectrograms(query, visual, semantic, out_dir="similar_results"):
    os.makedirs(out_dir, exist_ok=True)

    # Save the query image
    shutil.copy(query, os.path.join(out_dir, "query.png"))

    # Save visually similar spectrograms
    for idx, img_path in enumerate(visual, start=1):
        dst_path = os.path.join(out_dir, f"visual_{idx}.png")
        shutil.copy(img_path, dst_path)

    # Save semantically similar spectrograms
    for idx, img_path in enumerate(semantic, start=1):
        dst_path = os.path.join(out_dir, f"semantic_{idx}.png")
        shutil.copy(img_path, dst_path)

    print(f"📁 Saved all individual spectrograms in '{out_dir}'")

# --- CREATE COLLAGE ---
def save_collage(query, visual, semantic, out_file="similar_results/collage.png"):
    images = [Image.open(query)] + [Image.open(p) for p in visual] + [Image.open(p) for p in semantic]
    titles = ["Query"] + [f"Visual {i}" for i in range(1, 5)] + [f"Semantic {i}" for i in range(1, 5)]

    plt.figure(figsize=(20, 4))
    for i, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(1, 9, i)
        plt.imshow(img)
        plt.title(title, fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"📸 Saved collage of results to '{out_file}'")

# --- SAVE EVERYTHING ---
output_folder = "similar_results"
save_similar_spectrograms(query_path, visually_similar_paths, semantically_similar_paths, out_dir=output_folder)
save_collage(query_path, visually_similar_paths, semantically_similar_paths, out_file=os.path.join(output_folder, "collage.png"))
