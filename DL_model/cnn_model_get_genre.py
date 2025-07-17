import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
from PIL import Image
import shutil
import csv
import numpy as np

# --- CONFIG ---
train_dir = "/mnt/c/zhaw/Ampli-FIRE/spectrograms_32/train"  # <-- Path to train folder
test_dir = "/mnt/c/zhaw/Ampli-FIRE/spectrograms_32/test"    # <-- Path to test folder
batch_size = 32
num_epochs = 15
learning_rate = 0.001
img_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "genre_classifier_subfolders.pth"
output_csv = "predictions.csv"
output_folder = "predicted"

# --- FIXED GENRE LIST ---
fixed_genres = ['Rock', 'Pop', 'Hip-Hop & Rap', 'Electronic', 'R&B & Soul',
                'Jazz', 'Classical', 'Country & Folk', 'Latin', 'Metal',
                'Punk & Hardcore', 'Reggae & Ska', 'World & International', 'Blues', 'Other']
fixed_genres_lower = [g.lower() for g in fixed_genres]

print("🎯 Fixed genres:")
for g in fixed_genres:
    print(f"  - {g}")

# --- TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# --- LOAD TRAIN DATASET ---
dataset = datasets.ImageFolder(root=train_dir, transform=transform)
dataset.class_to_idx = {g: i for i, g in enumerate(fixed_genres_lower)}
dataset.idx_to_class = {i: g for g, i in dataset.class_to_idx.items()}

print(f"✅ Training genres (fixed to 15): {[g.title() for g in dataset.idx_to_class.values()]}")

# --- SPLIT DATASET ---
total_size = len(dataset)
val_size = int(0.1 * total_size)
test_size_split = int(0.1 * total_size)
train_size = total_size - val_size - test_size_split
train_set, val_set, test_set_split = random_split(dataset, [train_size, val_size, test_size_split],
                                                  generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# --- MODEL ---
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(fixed_genres))
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

# --- SAVE MODEL ---
torch.save({
    'model_state_dict': model.state_dict(),
    'class_to_idx': dataset.class_to_idx,
    'idx_to_class': dataset.idx_to_class
}, model_path)
print(f"💾 Saved model to '{model_path}'")

# --- PREDICT FUNCTIONS ---
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return checkpoint['idx_to_class']

def predict_single_image(image_path, idx_to_class):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        top_idx = np.argmax(probs)
        genre = idx_to_class[top_idx]
        confidence = probs[top_idx] * 100
    print(f"🎶 {os.path.basename(image_path)}: {genre.title()} ({confidence:.2f}% confidence)")
    return genre.title(), confidence

def predict_all_in_folder(test_dir, idx_to_class, output_csv, organize_folder=None):
    results = []
    test_images = [f for f in os.listdir(test_dir) if f.lower().endswith('.png')]

    if not test_images:
        print(f"❌ No PNG images found in '{test_dir}'")
        return

    print(f"📂 Found {len(test_images)} images in '{test_dir}'. Predicting genres...")

    if organize_folder:
        os.makedirs(organize_folder, exist_ok=True)
        for genre in fixed_genres:
            os.makedirs(os.path.join(organize_folder, genre), exist_ok=True)

    for img_name in test_images:
        img_path = os.path.join(test_dir, img_name)
        genre, confidence = predict_single_image(img_path, idx_to_class)
        results.append((img_name, genre, confidence))

        if organize_folder:
            dest_dir = os.path.join(organize_folder, genre)
            shutil.copy(img_path, os.path.join(dest_dir, img_name))

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Predicted Genre', 'Confidence (%)'])
        writer.writerows(results)
    print(f"💾 Predictions saved to '{output_csv}'")
    if organize_folder:
        print(f"📦 Images organized by genre in '{organize_folder}/'")

# --- MAIN LOOP ---
idx_to_class = load_model(model_path)

while True:
    print("\n📖 Menu:")
    print("1. Predict genres for all images in test folder")
    print("2. Predict genre for a single spectrogram")
    print("3. Exit")
    choice = input("👉 Enter your choice (1/2/3): ").strip()

    if choice == '1':
        predict_all_in_folder(test_dir, idx_to_class, output_csv, organize_folder=output_folder)
    elif choice == '2':
        file_name = input("🎵 Enter spectrogram filename (must be in test folder): ").strip()
        img_path = os.path.join(test_dir, file_name)
        if os.path.exists(img_path):
            predict_single_image(img_path, idx_to_class)
        else:
            print(f"❌ File '{file_name}' not found in test folder.")
    elif choice == '3':
        print("👋 Exiting...")
        break
    else:
        print("⚠️ Invalid choice. Please enter 1, 2, or 3.")
