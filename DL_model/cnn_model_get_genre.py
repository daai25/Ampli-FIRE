import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- CONFIG ---
base_dir = "/mnt/c/zhaw/Ampli-FIRE/spectrograms_64_split"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")
labels_file = "/mnt/c/zhaw/Ampli-FIRE/labeled_song_artists"
batch_size = 32
num_epochs = 5
learning_rate = 0.0005
img_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "genre_classifier_adapted.pth"

# --- LOAD LABELS ---
sep = "\t" if labels_file.endswith(".txt") else ","
label_df = pd.read_csv(labels_file, sep=sep)
print("📑 Columns in labeled_song_artists:", label_df.columns.tolist())

filename_col = "Filename"
genre_col = "Genre"

# Build filename → genre mapping
filename_to_genre = {
    row[filename_col].strip().lower(): row[genre_col].strip()
    for _, row in label_df.iterrows()
}

# Get unique genres
unique_genres = sorted(label_df[genre_col].unique())
if "Other" not in unique_genres:
    unique_genres.append("Other")
print(f"🎵 Found genres: {unique_genres}")

class_to_idx = {g: i for i, g in enumerate(unique_genres)}
idx_to_class = {i: g for g, i in class_to_idx.items()}

# --- TRANSFORMS ---
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

eval_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- CUSTOM DATASET ---
class SpectrogramDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
        self.labels = []
        for f in self.files:
            genre = filename_to_genre.get(f.strip().lower(), "Other")
            if genre not in class_to_idx:
                print(f"⚠️ Genre '{genre}' not in class_to_idx, assigning as 'Other'")
                genre = "Other"
            self.labels.append(class_to_idx[genre])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, self.files[idx])
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# --- LOAD DATASETS ---
train_dataset = SpectrogramDataset(train_dir, transform=train_transform)
val_dataset = SpectrogramDataset(val_dir, transform=eval_transform)
test_dataset = SpectrogramDataset(test_dir, transform=eval_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"✅ Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
print(f"✅ Classes: {unique_genres}")

# --- MODEL (MobileNetV3) ---
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(unique_genres))
model = model.to(device)

# --- LOSS & OPTIMIZER ---
y_train = train_dataset.labels
present_classes = np.unique(y_train)
computed_weights = compute_class_weight('balanced', classes=present_classes, y=y_train)

# Full weight tensor for all classes
weights_tensor = torch.ones(len(unique_genres), dtype=torch.float)
for cls, w in zip(present_classes, computed_weights):
    weights_tensor[cls] = w
weights_tensor = weights_tensor.to(device)

criterion = nn.CrossEntropyLoss(weight=weights_tensor)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- TRAINING LOOP ---
best_val_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
    train_acc = 100. * correct / total
    print(f"📈 Epoch {epoch+1}: Loss={running_loss/total:.4f}, Train Acc={train_acc:.2f}%")

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    val_acc = accuracy_score(val_labels, val_preds) * 100
    print(f"📊 Val Accuracy: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class
        }, model_path)
        print(f"💾 Saved best model at epoch {epoch+1}")

# --- PREDICT FUNCTIONS ---
def predict_image(image_name):
    img_path = os.path.join(test_dir, image_name)
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return
    img = Image.open(img_path).convert("RGB")
    img = eval_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        top_idx = np.argmax(probs)
        genre = idx_to_class[top_idx]
        confidence = probs[top_idx] * 100
    print(f"🎵 {image_name}: {genre} ({confidence:.2f}%)")

def predict_all():
    print("📂 Predicting all test spectrograms...")
    for f in os.listdir(test_dir):
        if f.lower().endswith('.png'):
            predict_image(f)

# --- AUTO PREDICT TEST ---
predict_all()

# --- INTERACTIVE MENU ---
while True:
    print("\n📖 Menu:")
    print("1. Predict genre by spectrogram image name")
    print("2. Predict all test spectrograms again")
    print("3. Exit")
    choice = input("👉 Enter choice (1/2/3): ").strip()
    if choice == '1':
        img_name = input("🖼 Enter image filename: ").strip()
        predict_image(img_name)
    elif choice == '2':
        predict_all()
    elif choice == '3':
        print("👋 Exiting.")
        break
    else:
        print("⚠️ Invalid choice.")
