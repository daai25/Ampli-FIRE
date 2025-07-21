import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
base_dir = "/mnt/c/zhaw/Ampli-FIRE/spectrograms_64_split"
test_dir = os.path.join(base_dir, "test")
img_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "genre_classifier_with_keywords.pth"

# --- GENRE MAPPING ---
keyword_map = {
    "Rock": ["rock", "alternative rock", "indie rock", "garage"],
    "Pop": ["pop", "electropop", "dance pop"],
    "Hip-Hop & Rap": ["hip hop", "rap", "trap", "drill"],
    "Electronic": ["electronic", "edm", "techno", "house", "trance", "dubstep"],
    "R&B & Soul": ["r&b", "soul", "neo soul", "contemporary r&b"],
    "Jazz": ["jazz", "bebop", "smooth jazz", "fusion"],
    "Classical": ["classical", "orchestral", "baroque", "symphony"],
    "Country & Folk": ["country", "folk", "americana", "bluegrass"],
    "Latin": ["latin", "reggaeton", "salsa", "bachata", "cumbia"],
    "Metal": ["metal", "heavy metal", "black metal", "thrash", "death metal"],
    "Punk & Hardcore": ["punk", "hardcore", "emo", "post-punk"],
    "Reggae & Ska": ["reggae", "ska", "dub"],
    "World & International": ["afro", "world", "k-pop", "j-pop", "bhangra", "afrobeats"],
    "Blues": ["blues", "delta blues", "electric blues"],
    "Other": []
}

# --- TRANSFORMS ---
eval_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# --- MODEL ---
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, len(keyword_map))
model = model.to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
class_to_idx = checkpoint['class_to_idx']
idx_to_class = checkpoint['idx_to_class']

# --- HELPER FUNCTIONS ---
def predict_image_by_name(image_name):
    img_path = os.path.join(test_dir, image_name)
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return None, 0
    img = Image.open(img_path).convert("RGB")
    img = eval_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        top_idx = np.argmax(probs)
        genre = idx_to_class[top_idx]
        confidence = probs[top_idx] * 100
    print(f"🎨 {image_name}: {genre} ({confidence:.2f}% confidence)")
    return genre, confidence

def predict_all_in_test_folder():
    print(f"📂 Predicting all spectrograms in {test_dir}...")
    for file in os.listdir(test_dir):
        if file.lower().endswith('.png'):
            predict_image_by_name(file)

# --- INTERACTIVE MENU ---
while True:
    print("\n📖 Menu:")
    print("1. Predict genre by spectrogram image name")
    print("2. Predict genres for all spectrograms in test folder")
    print("3. Exit")
    choice = input("👉 Enter your choice (1/2/3): ").strip()

    if choice == '1':
        image_name = input("🖼 Enter spectrogram image file name (with extension): ").strip()
        predict_image_by_name(image_name)
    elif choice == '2':
        predict_all_in_test_folder()
    elif choice == '3':
        print("👋 Exiting...")
        break
    else:
        print("⚠️ Invalid choice. Please enter 1, 2, or 3.")
