import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score

# --- CONFIG ---
data_dir = "/mnt/c/zhaw/Ampli-FIRE/spectrograms_32"  # <-- Path to spectrograms
batch_size = 32
num_epochs = 15
learning_rate = 0.001
img_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "genre_classifier_fixed15.pth"

# --- FIXED GENRE LIST ---
fixed_genres = ['rock', 'pop', 'hip-hop & rap', 'electronic', 'r&b & soul',
                'jazz', 'classical', 'country & folk', 'latin', 'metal',
                'punk & hardcore', 'reggae & ska', 'world & international', 'blues', 'other']
print(f"🎯 Using fixed genres: {[g.title() for g in fixed_genres]}")

# --- TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# --- DATASET ---
class SpectrogramDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.files = glob.glob(os.path.join(folder, "*.png"))
        if not self.files:
            raise FileNotFoundError(f"❌ No PNG files found in {folder}")
        self.transform = transform

        self.labels = []
        for f in self.files:
            genre_name = os.path.basename(f).split('_')[0].lower()  # e.g., 'rock_song1.png' → 'rock'
            if genre_name in fixed_genres:
                self.labels.append(genre_name)
            else:
                self.labels.append('other')  # Unknown genres → 'Other'

        self.class_to_idx = {cls: idx for idx, cls in enumerate(fixed_genres)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        label_name = self.labels[idx]
        label = self.class_to_idx[label_name]
        if self.transform:
            img = self.transform(img)
        return img, label

# --- LOAD DATASET ---
dataset = SpectrogramDataset(data_dir, transform=transform)
print(f"✅ Total spectrograms: {len(dataset)}")

# --- SPLIT DATASET ---
total_size = len(dataset)
val_size = int(0.1 * total_size)
test_size = int(0.1 * total_size)
train_size = total_size - val_size - test_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size],
                                            generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# --- MODEL ---
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(fixed_genres))  # 15 genres
model = model.to(device)

# --- LOSS & OPTIMIZER ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- TRAINING LOOP ---
print("🚀 Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / train_size
    train_acc = 100. * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {train_loss:.4f} Acc: {train_acc:.2f}%")

# --- VALIDATION ---
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
print(f"📊 Validation Accuracy: {val_acc:.2f}%")

# --- TEST ---
test_preds, test_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = outputs.max(1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
test_acc = accuracy_score(test_labels, test_preds) * 100
print(f"✅ Test Accuracy: {test_acc:.2f}%")

# --- SAVE MODEL ---
torch.save({
    'model_state_dict': model.state_dict(),
    'class_to_idx': dataset.class_to_idx,
    'idx_to_class': dataset.idx_to_class
}, model_path)
print(f"💾 Saved model to '{model_path}'")

# --- PREDICT FUNCTION ---
def predict_genre(image_path, model_path="genre_classifier_fixed15.pth"):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    idx_to_class = checkpoint['idx_to_class']

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        top_idx = np.argmax(probs)
        genre = idx_to_class[top_idx]
        confidence = probs[top_idx] * 100
    print(f"🎶 Predicted Genre: {genre.title()} ({confidence:.2f}% confidence)")
    return genre.title(), confidence

# --- TEST PREDICTION ---
test_image = input("🎵 Enter spectrogram filename to predict genre: ").strip()
test_image_path = os.path.join(data_dir, test_image)
if os.path.exists(test_image_path):
    predict_genre(test_image_path)
else:
    print(f"❌ Spectrogram '{test_image}' not found in {data_dir}")
