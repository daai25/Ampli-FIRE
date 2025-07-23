import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

# --- CONFIG ---
base_dir =os.path.expanduser('~/Switzerland/project/Ampli-FIRE/spectrograms_64_split')
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")
label_file = os.path.expanduser('~/Switzerland/project/Ampli-FIRE/labeled_song_artists_grouped.csv')
batch_size = 32
num_epochs = 50  # Increase slightly but let early stopping control
learning_rate = 0.0003
patience = 5  # Early stopping patience (stop if no val improvement for 5 epochs)
img_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "genre_classifier_mobilenetv3_fixed.pth"

# --- GENRES ---
parent_genres = ["Rock", "Pop", "Hip-Hop & Rap", "Electronic", "R&B & Soul",
                 "Jazz", "Classical", "Country & Folk", "Latin", "Metal",
                 "Punk & Hardcore", "Reggae & Ska", "World & International", "Blues", "Other"]
class_to_idx = {g: i for i, g in enumerate(parent_genres)}
idx_to_class = {i: g for g, i in class_to_idx.items()}

# --- COLLAPSE SUBGENRES ---
def collapse_genre(subgenre):
    for parent in parent_genres[:-1]:
        if parent.lower() in subgenre.lower():
            return parent
    return "Other"

# --- LOAD LABELS ---
label_df = pd.read_csv(label_file)
label_df.columns = [col.strip().lower() for col in label_df.columns]
label_df = label_df.rename(columns={"filename": "song_name", "genre": "genres"})
label_df["parent_genre"] = label_df["genres"].apply(collapse_genre)
filename_to_genre = {
    os.path.splitext(row["song_name"].lower().strip())[0]: row["parent_genre"]
    for _, row in label_df.iterrows()
}
print(f"✅ Loaded labels. Parent genres: {sorted(set(label_df['parent_genre']))}")

# --- TRANSFORMS ---
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

eval_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# --- CUSTOM DATASET ---
class SpectrogramDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
        self.labels = []
        for f in self.files:
            f_base = os.path.splitext(f.lower().strip())[0]
            genre = filename_to_genre.get(f_base, "Other")
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

# --- DATASETS & LOADERS ---
train_dataset = SpectrogramDataset(train_dir, transform=train_transform)
val_dataset = SpectrogramDataset(val_dir, transform=eval_transform)
test_dataset = SpectrogramDataset(test_dir, transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 🔥 Check distribution
train_dist = Counter(train_dataset.labels)
print(f"📊 Train distribution: { {idx_to_class[k]: v for k, v in train_dist.items()} }")

# --- MODEL ---
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = True
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(parent_genres))
model = model.to(device)

# --- LOSS & OPTIMIZER ---
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# --- TRAINING LOOP WITH EARLY STOPPING ---
best_val_acc = 0.0
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    train_loss = running_loss / len(train_dataset)
    train_acc = 100. * correct / total
    print(f"📈 Epoch {epoch+1}: Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%")

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    val_acc = accuracy_score(val_labels, val_preds) * 100
    print(f"📊 Val Accuracy: {val_acc:.2f}%")

    scheduler.step()

    # Check for improvement
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class
        }, model_path)
        print(f"💾 Saved best model at Epoch {epoch+1} (Val Acc: {val_acc:.2f}%)")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"⚠️ No improvement for {epochs_no_improve} epochs")
        if epochs_no_improve >= patience:
            print(f"⏹ Early stopping at epoch {epoch+1}")
            break

# --- LOAD BEST MODEL ---
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# --- TEST ACCURACY ---
print("\n🔍 Evaluating on test set...")
test_preds, test_labels = [], []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = outputs.max(1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(test_labels, test_preds) * 100
print(f"✅ Overall Test Accuracy: {test_acc:.2f}%")

# --- PREDICT FUNCTIONS ---
def predict_image(image_name):
    img_path = os.path.join(test_dir, image_name)
    if not os.path.exists(img_path):
        print(f"❌ File not found: {image_name}")
        return
    img = Image.open(img_path).convert("RGB")
    img = eval_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        top_idx = np.argmax(probs)
        print(f"🎵 {image_name}: {idx_to_class[top_idx]} ({probs[top_idx]*100:.2f}%)")

def predict_all():
    print(f"📂 Predicting all spectrograms in {test_dir}...")
    for f in os.listdir(test_dir):
        if f.lower().endswith('.png'):
            predict_image(f)

# --- INTERACTIVE MENU ---
while True:
    print("\n📖 Menu:")
    print("1. Predict genre for a spectrogram")
    print("2. Predict all spectrograms in test folder")
    print("3. Exit")
    choice = input("👉 Enter choice (1/2/3): ").strip()
    if choice == '1':
        img_name = input("🖼 Enter spectrogram file name: ").strip()
        predict_image(img_name)
    elif choice == '2':
        predict_all()
    elif choice == '3':
        print("👋 Exiting...")
        break
    else:
        print("⚠️ Invalid choice. Please enter 1, 2, or 3.")
